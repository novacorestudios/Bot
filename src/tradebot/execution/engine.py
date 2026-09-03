"""Execution engine.

The only component that sends order actions to the exchange. It takes an
``OrderIntent`` — which only the risk engine can construct — and is responsible
for the gap between "we decided to trade" and "we know what we own".

Four properties matter more than anything else here:

**1. Idempotency.** Every order carries a deterministic client order id derived
from the intent. If a submission times out we do NOT re-send; we query by that
id and adopt whatever the exchange actually has. A blind retry is how a bot ends
up with two positions where it believes it has one.

**2. No position without a stop.** After a fill, a protective stop is placed
immediately. If it cannot be placed, the position is closed at once. A position
whose stop silently failed looks identical to a protected one right up until it
does not.

**3. Serialised per symbol.** A per-symbol lock plus an in-flight registry means
two signals in the same cycle cannot open two positions in the same market.

**4. Actual fills, not requested ones.** Position size, stop distance and risk
are all recomputed from what was actually filled. A partial fill that is tracked
at the requested size produces a stop sized for a position that does not exist.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.config import TunableConfig
from tradebot.core.errors import (
    ExchangeError,
    FilterViolationError,
    NetworkError,
    TimeoutError_,
)
from tradebot.core.events import EventBus, EventType
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import (
    ExitReason,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    RiskEvent,
    RiskEventType,
    Trade,
    new_id,
)
from tradebot.execution.state import OrderState, OrderTracker

log = get_logger(__name__)


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of attempting to open a position."""

    success: bool
    position: Position | None = None
    order: Order | None = None
    reason: str = ""
    slippage: float = 0.0
    protected: bool = False

    @classmethod
    def failure(cls, reason: str, order: Order | None = None) -> ExecutionResult:
        return cls(success=False, reason=reason, order=order)


class ExecutionEngine:
    """Turns approved intents into positions, and manages their protection."""

    def __init__(
        self,
        config: TunableConfig,
        gateway: Any,
        event_bus: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.gateway = gateway
        self.events = event_bus or EventBus()
        self.clock = clock or SystemClock()

        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.trades: list[Trade] = []
        # Every order's lifecycle, from before it is sent to a terminal state.
        # The user stream, the REST response and reconciliation all write here,
        # and the tracker is what makes their interleaving deterministic.
        self.tracker = OrderTracker(self.clock)

        self._locks: dict[str, asyncio.Lock] = {}
        self._in_flight: set[str] = set()

        self.entries_blocked = False
        self.entries_blocked_reason = ""

        self.submitted = 0
        self.filled = 0
        self.rejected = 0
        self.unprotected_closures = 0

    # ------------------------------------------------------------------ #
    def lock_for(self, symbol: str) -> asyncio.Lock:
        found = self._locks.get(symbol)
        if found is None:
            found = asyncio.Lock()
            self._locks[symbol] = found
        return found

    @property
    def in_flight(self) -> set[str]:
        return set(self._in_flight)

    def block_entries(self, reason: str) -> None:
        """Stop opening positions. Exits remain permitted, always."""
        self.entries_blocked = True
        self.entries_blocked_reason = reason
        log.warning("entries_blocked", reason=reason)

    def unblock_entries(self) -> None:
        if self.entries_blocked:
            log.info("entries_unblocked", previous_reason=self.entries_blocked_reason)
        self.entries_blocked = False
        self.entries_blocked_reason = ""

    # ------------------------------------------------------------------ #
    # Entry
    # ------------------------------------------------------------------ #
    async def open_position(self, intent: OrderIntent) -> ExecutionResult:
        """Submit an entry, confirm the fill, and attach protection."""
        symbol = intent.symbol

        if self.entries_blocked:
            return ExecutionResult.failure(f"entries are blocked: {self.entries_blocked_reason}")

        async with self.lock_for(symbol):
            if symbol in self.positions:
                return ExecutionResult.failure("already holding this symbol")
            if symbol in self._in_flight:
                return ExecutionResult.failure("an order is already in flight")

            self._in_flight.add(symbol)
            try:
                return await self._open(intent)
            finally:
                self._in_flight.discard(symbol)

    async def _open(self, intent: OrderIntent) -> ExecutionResult:
        symbol = intent.symbol

        # Leverage must be set BEFORE the order, and the exchange may cap it.
        # Sizing computed against a leverage the exchange refused would be wrong,
        # so the applied value is what gets recorded.
        applied_leverage = intent.leverage
        try:
            applied_leverage = await self.gateway.set_leverage(symbol, intent.leverage)
        except ExchangeError as exc:
            log.warning("set_leverage_failed", symbol=symbol, error=str(exc))

        # Registered BEFORE the request leaves: if the process dies between
        # here and the exchange acknowledging, the client order id still exists
        # for reconciliation to look up.
        tracked = self.tracker.create(
            intent.client_order_id,
            symbol,
            intent.direction.name,
            intent.quantity,
            purpose="ENTRY",
        )
        self.tracker.transition(tracked.client_order_id, OrderState.SUBMITTED, "execution")

        self.submitted += 1
        try:
            order = await self.gateway.place_order(intent)
        except FilterViolationError as exc:
            self.rejected += 1
            self.tracker.transition(
                tracked.client_order_id, OrderState.FAILED, "execution", str(exc)[:120]
            )
            await self._emit_risk_event(
                RiskEventType.ORDER_REJECTED,
                "WARNING",
                f"{symbol}: order rejected by filters: {exc}",
                symbol,
            )
            return ExecutionResult.failure(f"filter violation: {exc}")
        except (TimeoutError_, NetworkError) as exc:
            # The gateway already attempted resolution by client order id. If it
            # raised, we genuinely do not know the state — block entries and let
            # reconciliation settle it rather than guessing.
            self.block_entries(f"indeterminate order state for {symbol}")
            self.tracker.transition(
                tracked.client_order_id, OrderState.INDETERMINATE, "execution", str(exc)[:120]
            )
            await self._emit_risk_event(
                RiskEventType.RECONCILIATION_MISMATCH,
                "CRITICAL",
                f"{symbol}: order state is INDETERMINATE ({exc}); entries "
                f"blocked until reconciliation",
                symbol,
            )
            return ExecutionResult.failure(f"indeterminate order state: {exc}")
        except ExchangeError as exc:
            self.rejected += 1
            self.tracker.transition(
                tracked.client_order_id, OrderState.REJECTED, "execution", str(exc)[:120]
            )
            await self._emit_risk_event(
                RiskEventType.ORDER_REJECTED,
                "WARNING",
                f"{symbol}: {exc}",
                symbol,
            )
            return ExecutionResult.failure(str(exc))

        self.orders[order.client_order_id] = order
        self._sync_tracker(order, "rest_response")

        if order.status is OrderStatus.UNKNOWN:
            self.block_entries(f"unknown order state for {symbol}")
            return ExecutionResult.failure("order state unknown after submission", order)
        if order.status is OrderStatus.REJECTED:
            self.rejected += 1
            await self._emit_risk_event(
                RiskEventType.ORDER_REJECTED,
                "WARNING",
                f"{symbol}: {order.error or 'rejected'}",
                symbol,
            )
            return ExecutionResult.failure(order.error or "rejected", order)

        if order.filled_quantity <= 0:
            return ExecutionResult.failure("order did not fill", order)

        # -- everything below uses the ACTUAL fill, never the request -------- #
        fill_price = order.average_price or intent.price or 0.0
        if fill_price <= 0:
            return ExecutionResult.failure("filled with no average price", order)

        reference = intent.metadata.get("reference_price", fill_price)
        slippage = safe_div(abs(fill_price - reference), reference, 0.0)

        if slippage > self.config.execution.max_entry_slippage:
            # Filled far from where the decision was made: the edge that
            # justified the trade no longer exists. Close immediately rather
            # than hold a position whose thesis was priced away.
            log.warning(
                "entry_slippage_exceeded",
                symbol=symbol,
                slippage=round(slippage, 6),
                limit=self.config.execution.max_entry_slippage,
            )
            await self._emergency_close(
                symbol,
                order,
                fill_price,
                intent,
                ExitReason.RISK_EVENT,
                "entry slippage exceeded the limit",
            )
            return ExecutionResult.failure(
                f"entry slippage {slippage * 100:.3f}% exceeded the limit", order
            )

        filled_margin = order.filled_quantity * fill_price / max(1, applied_leverage)
        margin_cap = intent.metadata.get("margin_per_trade_cap")
        if margin_cap is not None and filled_margin > float(margin_cap) + 1e-12:
            log.error(
                "filled_margin_cap_exceeded",
                symbol=symbol,
                filled_margin=round(filled_margin, 8),
                limit=float(margin_cap),
            )
            await self._emergency_close(
                symbol,
                order,
                fill_price,
                intent,
                ExitReason.RISK_EVENT,
                "filled initial margin exceeded the per-trade cap",
            )
            return ExecutionResult.failure(
                "filled initial margin exceeded the per-trade cap", order
            )

        position = Position(
            position_id=new_id("p_"),
            symbol=symbol,
            direction=intent.direction,
            quantity=order.filled_quantity,
            entry_price=fill_price,
            leverage=applied_leverage,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            strategy=intent.strategy,
            regime=intent.regime,
            opened_at=self.clock.now_ms(),
            entry_notional=order.filled_quantity * fill_price,
            allocated_initial_margin=filled_margin,
            entry_fee=order.total_commission,
            entry_slippage=slippage * order.filled_quantity * fill_price,
            initial_stop=intent.stop_loss,
            initial_risk=abs(fill_price - intent.stop_loss) * order.filled_quantity,
            highest_price=fill_price,
            lowest_price=fill_price,
            entry_order_id=order.client_order_id,
            opportunity_score=intent.opportunity_score,
            expected_net_edge=intent.expected_net_edge,
            metadata={
                "intent_id": intent.intent_id,
                "requested_quantity": intent.quantity,
                "partial_fill": order.filled_quantity < intent.quantity,
                **intent.metadata,
            },
        )
        self.positions[symbol] = position
        self.filled += 1

        protected = await self._attach_protection(position)
        if not protected:
            # A position we cannot protect is worse than no position.
            self.unprotected_closures += 1
            await self._emergency_close(
                symbol,
                order,
                fill_price,
                intent,
                ExitReason.RISK_EVENT,
                "protective stop could not be placed",
            )
            return ExecutionResult.failure(
                "could not place a protective stop; position was closed", order
            )

        await self.events.emit(EventType.POSITION_OPENED, position, source="execution")
        log.info(
            "position_opened",
            symbol=symbol,
            direction=intent.direction.value,
            quantity=position.quantity,
            entry=fill_price,
            stop=position.stop_loss,
            target=position.take_profit,
            leverage=applied_leverage,
            slippage_bps=round(slippage * 10_000, 2),
            partial=position.metadata["partial_fill"],
        )

        return ExecutionResult(
            success=True, position=position, order=order, slippage=slippage, protected=True
        )

    # ------------------------------------------------------------------ #
    async def _attach_protection(self, position: Position) -> bool:
        """Place the protective stop. Returns False if it could not be placed."""
        try:
            stop_order = await self.gateway.place_protective_order(
                symbol=position.symbol,
                order_type=OrderType(self.config.execution.stop_order_type),
                stop_price=position.stop_loss,
                quantity=position.quantity,
                direction_sign=position.direction.sign,
                client_order_id=f"tb_sl_{position.position_id[:12]}",
            )
        except ExchangeError as exc:
            log.error("protective_stop_failed", symbol=position.symbol, error=str(exc))
            await self._emit_risk_event(
                RiskEventType.MISSING_STOP_LOSS,
                "CRITICAL",
                f"{position.symbol}: could not place a protective stop ({exc})",
                position.symbol,
            )
            return False

        position.stop_order_id = stop_order.client_order_id
        self._track_resting(stop_order, position, "STOP")

        # A take-profit is desirable but not mandatory: the position monitor can
        # close on target. A missing STOP is fatal; a missing target is not.
        with contextlib.suppress(ExchangeError):
            target_order = await self.gateway.place_protective_order(
                symbol=position.symbol,
                order_type=OrderType(self.config.execution.take_profit_order_type),
                stop_price=position.take_profit,
                quantity=position.quantity,
                direction_sign=position.direction.sign,
                client_order_id=f"tb_tp_{position.position_id[:12]}",
            )
            position.take_profit_order_id = target_order.client_order_id
            self._track_resting(target_order, position, "TARGET")
        return True

    async def _emergency_close(
        self,
        symbol: str,
        order: Order,
        price: float,
        intent: OrderIntent,
        reason: ExitReason,
        detail: str,
    ) -> None:
        """Flatten immediately. Used when a position must not be kept."""
        position = self.positions.get(symbol)
        if position is None:
            position = Position(
                position_id=new_id("p_"),
                symbol=symbol,
                direction=intent.direction,
                quantity=order.filled_quantity,
                entry_price=price,
                leverage=intent.leverage,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                strategy=intent.strategy,
                regime=intent.regime,
                opened_at=self.clock.now_ms(),
                entry_notional=order.filled_quantity * price,
                allocated_initial_margin=order.filled_quantity * price / max(1, intent.leverage),
                entry_fee=order.total_commission,
                initial_stop=intent.stop_loss,
            )
            self.positions[symbol] = position

        log.critical("emergency_close", symbol=symbol, reason=detail)
        # Called from inside open_position, which already holds the symbol lock.
        await self._close_locked(symbol, reason, detail)

    # ------------------------------------------------------------------ #
    # Exit — never blocked, by anything
    # ------------------------------------------------------------------ #
    async def close_position(
        self, symbol: str, reason: ExitReason, detail: str = ""
    ) -> Trade | None:
        """Close a position. This path is never gated by any risk condition."""
        if symbol not in self.positions:
            return None
        async with self.lock_for(symbol):
            return await self._close_locked(symbol, reason, detail)

    async def _close_locked(
        self, symbol: str, reason: ExitReason, detail: str = ""
    ) -> Trade | None:
        """Close a position, assuming the caller already holds the symbol lock.

        Separate from :meth:`close_position` because ``asyncio.Lock`` is not
        reentrant: an emergency close raised from inside ``open_position`` — a
        failed protective stop, say — would otherwise deadlock against the lock
        its own caller is holding, and the engine would hang with a live,
        unprotected position.
        """
        position = self.positions.get(symbol)
        if position is None:
            return None

        # Cancel resting protection first, so the close is not racing our own
        # stop order.
        for resting in self.tracker.open_for(symbol):
            if resting.purpose in {"STOP", "TARGET"}:
                self.tracker.transition(
                    resting.client_order_id, OrderState.CANCEL_REQUESTED, "execution", "closing"
                )
        with contextlib.suppress(ExchangeError):
            await self.gateway.cancel_all_orders(symbol)
        for resting in self.tracker.open_for(symbol):
            if resting.purpose in {"STOP", "TARGET"}:
                self.tracker.transition(
                    resting.client_order_id, OrderState.CANCELLED, "execution", "closing"
                )

        try:
            order = await self.gateway.close_position(
                symbol, position, f"tb_x_{position.position_id[:12]}"
            )
        except ExchangeError as exc:
            log.critical(
                "position_close_failed",
                symbol=symbol,
                error=str(exc),
                message="POSITION REMAINS OPEN AND MAY BE UNPROTECTED",
            )
            await self._emit_risk_event(
                RiskEventType.MISSING_STOP_LOSS,
                "CRITICAL",
                f"{symbol}: close FAILED ({exc}); the position is still open",
                symbol,
            )
            return None

        self.tracker.create(
            order.client_order_id, symbol, position.direction.name, position.quantity, "EXIT"
        )
        self.tracker.transition(order.client_order_id, OrderState.SUBMITTED, "execution")
        self._sync_tracker(order, "rest_response")

        exit_price = order.average_price or position.entry_price
        trade = self._build_trade(position, order, exit_price, reason)
        self.trades.append(trade)
        self.positions.pop(symbol, None)

        await self.events.emit(EventType.TRADE_COMPLETED, trade, source="execution")
        log.info(
            "position_closed",
            symbol=symbol,
            reason=reason.value,
            detail=detail,
            entry=position.entry_price,
            exit=exit_price,
            net_pnl=round(trade.net_pnl, 6),
            duration_sec=round(trade.duration_sec, 1),
        )
        return trade

    def _build_trade(
        self, position: Position, order: Order, exit_price: float, reason: ExitReason
    ) -> Trade:
        gross = (exit_price - position.entry_price) * position.quantity * position.direction.sign
        exit_fee = order.total_commission or (
            exit_price * position.quantity * self.config.edge.taker_fee
        )
        fees = position.entry_fee + exit_fee
        net = gross - fees - position.funding_paid

        return Trade(
            trade_id=new_id("t_"),
            symbol=position.symbol,
            strategy=position.strategy,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            stop_loss=position.initial_stop,
            take_profit=position.take_profit,
            opened_at=position.opened_at,
            closed_at=self.clock.now_ms(),
            gross_pnl=gross,
            fees=fees,
            funding=position.funding_paid,
            slippage_cost=position.entry_slippage,
            net_pnl=net,
            exit_reason=reason,
            regime=position.regime,
            opportunity_score=position.opportunity_score,
            expected_net_edge=position.expected_net_edge,
            entry_notional=position.entry_notional,
            initial_risk=position.initial_risk,
            metadata=dict(position.metadata),
        )

    async def close_all(self, reason: ExitReason, detail: str = "") -> list[Trade]:
        """Flatten everything. Used by kill switches and shutdown."""
        closed: list[Trade] = []
        for symbol in list(self.positions):
            trade = await self.close_position(symbol, reason, detail)
            if trade is not None:
                closed.append(trade)
        return closed

    # ------------------------------------------------------------------ #
    async def update_stop(self, symbol: str, new_stop: float) -> bool:
        """Move a stop (trailing). Cancel-then-place, never place-then-cancel.

        Between the cancel and the place the position is briefly unprotected.
        That window is unavoidable with a cancel/replace, but doing it the other
        way round risks two live stops, which on a fill leaves a reversed
        position — strictly worse.
        """
        position = self.positions.get(symbol)
        if position is None:
            return False

        async with self.lock_for(symbol):
            if position.stop_order_id:
                with contextlib.suppress(ExchangeError):
                    await self.gateway.cancel_order(symbol, position.stop_order_id)

            try:
                order = await self.gateway.place_protective_order(
                    symbol=symbol,
                    order_type=OrderType(self.config.execution.stop_order_type),
                    stop_price=new_stop,
                    quantity=position.quantity,
                    direction_sign=position.direction.sign,
                    client_order_id=f"tb_sl_{new_id()[:10]}",
                )
            except ExchangeError as exc:
                log.error(
                    "stop_update_failed",
                    symbol=symbol,
                    error=str(exc),
                    message="POSITION IS NOW UNPROTECTED; closing it",
                )
                await self._close_locked(
                    symbol, ExitReason.RISK_EVENT, f"stop replacement failed: {exc}"
                )
                return False

            position.stop_loss = new_stop
            position.stop_order_id = order.client_order_id
            position.trailing_active = True
            return True

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Order state machine
    # ------------------------------------------------------------------ #
    def _track_resting(self, order: Order, position: Position, purpose: str) -> None:
        """Register a resting protective order so its fill is not a surprise."""
        self.tracker.create(
            order.client_order_id,
            position.symbol,
            "SELL" if position.direction.sign > 0 else "BUY",
            position.quantity,
            purpose=purpose,
        )
        self.tracker.transition(order.client_order_id, OrderState.SUBMITTED, "execution")
        self._sync_tracker(order, "rest_response")

    def _sync_tracker(self, order: Order, source: str) -> None:
        """Push a REST order response into the state machine."""
        self.tracker.apply_exchange_update(
            {
                "client_order_id": order.client_order_id,
                "exchange_order_id": order.exchange_order_id or "",
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price,
                "commission": order.total_commission,
            },
            source,
        )

    async def on_order_update(self, update: dict[str, Any]) -> None:
        """Apply a live ``ORDER_TRADE_UPDATE`` from the user data stream.

        This is the difference between learning about a stop fill in
        milliseconds and learning about it on the next reconciliation sweep,
        with the position still counted as open in between — which is how a
        flat account ends up refusing new trades on "max concurrent positions".
        """
        client_order_id = str(update.get("client_order_id") or "")
        if not client_order_id or client_order_id not in self.tracker:
            # Not ours: a manual order, or one from before a restart.
            # Reconciliation owns that case; guessing here would be worse.
            return

        before = self.tracker.get(client_order_id)
        purpose = before.purpose if before else "ENTRY"
        self.tracker.apply_exchange_update(update, "user_stream")

        after = self.tracker.get(client_order_id)
        if after is None or after.state is not OrderState.FILLED:
            return

        symbol = str(update.get("symbol") or (before.symbol if before else ""))
        if purpose in {"STOP", "TARGET"} and symbol in self.positions:
            # A protective order filled: the exchange has already flattened us.
            # Record the trade from the fill rather than sending another order.
            await self._settle_protective_fill(symbol, purpose, after.average_price)

    async def _settle_protective_fill(
        self, symbol: str, purpose: str, exit_price: float
    ) -> Trade | None:
        """Book the trade for a stop or target that the exchange filled."""
        async with self.lock_for(symbol):
            position = self.positions.get(symbol)
            if position is None:
                return None

            reason = ExitReason.STOP_LOSS if purpose == "STOP" else ExitReason.TAKE_PROFIT
            price = exit_price or (
                position.stop_loss if purpose == "STOP" else position.take_profit
            )
            settled = Order(
                client_order_id=(
                    position.stop_order_id if purpose == "STOP" else position.take_profit_order_id
                )
                or "",
                symbol=symbol,
                side=OrderSide.SELL if position.direction.sign > 0 else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                status=OrderStatus.FILLED,
                filled_quantity=position.quantity,
                average_price=price,
                reduce_only=True,
            )
            trade = self._build_trade(position, settled, price, reason)
            self.trades.append(trade)
            self.positions.pop(symbol, None)

            # The other side of the bracket is now dead; cancel it so it cannot
            # open a position in the opposite direction.
            with contextlib.suppress(ExchangeError):
                await self.gateway.cancel_all_orders(symbol)

            await self.events.emit(EventType.TRADE_COMPLETED, trade, source="execution")
            log.info(
                "protective_order_filled",
                symbol=symbol,
                purpose=purpose,
                exit=price,
                net_pnl=round(trade.net_pnl, 6),
            )
            return trade

    # ------------------------------------------------------------------ #
    async def _emit_risk_event(
        self,
        event_type: RiskEventType,
        severity: str,
        message: str,
        symbol: str | None = None,
        **data: Any,
    ) -> None:
        await self.events.emit(
            EventType.RISK_EVENT,
            RiskEvent(
                event_type=event_type,
                severity=severity,
                message=message,
                timestamp=self.clock.now_ms(),
                symbol=symbol,
                data=data,
            ),
            source="execution",
        )

    def stats(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "filled": self.filled,
            "rejected": self.rejected,
            "fill_rate": safe_div(self.filled, self.submitted, 0.0),
            "open_positions": len(self.positions),
            "completed_trades": len(self.trades),
            "unprotected_closures": self.unprotected_closures,
            "entries_blocked": self.entries_blocked,
            "entries_blocked_reason": self.entries_blocked_reason,
            "in_flight": sorted(self._in_flight),
            "orders": self.tracker.stats(),
        }


@dataclass(slots=True)
class UnprotectedPosition:
    """A position found without a stop — the state that must never persist."""

    symbol: str
    position: Position
    detail: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


def positions_without_stops(
    positions: dict[str, Position], orders: list[Order]
) -> list[UnprotectedPosition]:
    """Find open positions with no live protective order.

    Used by the reconciler and the health monitor. A position whose stop was
    cancelled, filled without closing, or never placed looks exactly like a
    protected one from the position endpoint alone.
    """
    protective_symbols = {
        order.symbol
        for order in orders
        if order.status.is_open
        and order.order_type
        in {
            OrderType.STOP_MARKET,
            OrderType.STOP,
            OrderType.TRAILING_STOP_MARKET,
        }
    }
    return [
        UnprotectedPosition(
            symbol=symbol,
            position=position,
            detail="no resting stop order for an open position",
            fields={"quantity": position.quantity, "direction": position.direction.value},
        )
        for symbol, position in positions.items()
        if symbol not in protective_symbols
    ]
