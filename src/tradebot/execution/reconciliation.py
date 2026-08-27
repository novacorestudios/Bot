"""State reconciliation.

**The exchange is the source of truth. Local state is a cache.**

Every restart, reconnection and periodic check re-derives reality from Binance
and resolves any disagreement in the exchange's favour. Trading on a stale local
view is the single most dangerous thing this system can do — it can size against
capital that is already committed, place a second position believing the first
closed, or leave a real position with no stop because the local copy thinks one
exists.

Resolution rules:

| Situation | Action |
|---|---|
| Position on the exchange, not local | **Adopt it and protect it immediately.** It has no known thesis, so it is flagged for closure at the next acceptable opportunity. |
| Position local, not on the exchange | Mark closed and reconcile the PnL from `userTrades`. |
| Quantity mismatch | Trust the exchange, log a mismatch, re-place protection at the true size. |
| Orphan protective order with no position | Cancel it. |
| Open position with no protective order | Place one at once; if that fails, close the position. |

Entries stay blocked for the whole of reconciliation. Exits never are.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.errors import ExchangeError
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import (
    Direction,
    ExitReason,
    Order,
    OrderSide,
    OrderType,
    Position,
    RiskEventType,
)

log = get_logger(__name__)

#: Relative quantity difference below which a mismatch is treated as rounding.
QUANTITY_TOLERANCE = 1e-6


@dataclass(slots=True)
class ReconciliationReport:
    """What was found and what was done about it."""

    exchange_positions: int = 0
    local_positions: int = 0
    adopted: list[str] = field(default_factory=list)
    closed_locally: list[str] = field(default_factory=list)
    quantity_mismatches: list[str] = field(default_factory=list)
    orphan_orders_cancelled: list[str] = field(default_factory=list)
    unprotected_fixed: list[str] = field(default_factory=list)
    unprotected_closed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def clean(self) -> bool:
        """True when local state already matched the exchange."""
        return not (
            self.adopted
            or self.closed_locally
            or self.quantity_mismatches
            or self.orphan_orders_cancelled
            or self.unprotected_fixed
            or self.unprotected_closed
            or self.errors
        )

    @property
    def mismatch_count(self) -> int:
        return len(self.adopted) + len(self.closed_locally) + len(self.quantity_mismatches)

    def summary(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "exchange_positions": self.exchange_positions,
            "local_positions": self.local_positions,
            "adopted": self.adopted,
            "closed_locally": self.closed_locally,
            "quantity_mismatches": self.quantity_mismatches,
            "orphan_orders_cancelled": self.orphan_orders_cancelled,
            "unprotected_fixed": self.unprotected_fixed,
            "unprotected_closed": self.unprotected_closed,
            "errors": self.errors,
            "duration_sec": round(self.duration_sec, 3),
        }


class Reconciler:
    """Re-derives local state from the exchange and resolves disagreements."""

    def __init__(
        self,
        gateway: Any,
        execution: Any,
        config: Any,
        clock: Clock | None = None,
        risk_engine: Any = None,
    ) -> None:
        self.gateway = gateway
        self.execution = execution
        self.config = config
        self.clock = clock or SystemClock()
        self.risk = risk_engine

        self.runs = 0
        self.consecutive_mismatches = 0
        self.last_report: ReconciliationReport | None = None

    # ------------------------------------------------------------------ #
    async def reconcile(self, block_entries: bool = True) -> ReconciliationReport:
        """Full reconciliation. Entries are blocked for its duration."""
        import time as _time

        started = _time.time()
        report = ReconciliationReport()
        self.runs += 1

        if block_entries:
            self.execution.block_entries("reconciliation in progress")

        try:
            exchange_positions = await self.gateway.get_positions()
            open_orders = await self.gateway.get_open_orders()
        except ExchangeError as exc:
            report.errors.append(f"could not read exchange state: {exc}")
            log.error(
                "reconciliation_failed",
                error=str(exc),
                message="entries remain blocked; local state is unverified",
            )
            report.duration_sec = _time.time() - started
            self.last_report = report
            return report

        local = self.execution.positions
        report.exchange_positions = len(exchange_positions)
        report.local_positions = len(local)

        await self._adopt_unknown(exchange_positions, local, report)
        await self._close_phantom(exchange_positions, local, report)
        await self._fix_quantities(exchange_positions, local, report)
        await self._cancel_orphans(exchange_positions, open_orders, report)
        await self._ensure_protection(open_orders, report)

        report.duration_sec = _time.time() - started
        self.last_report = report

        if report.mismatch_count:
            self.consecutive_mismatches += 1
            if self.risk is not None:
                self.risk.record_reconciliation_mismatch()
            log.warning("reconciliation_mismatch", **report.summary())
        else:
            self.consecutive_mismatches = 0
            if self.risk is not None:
                self.risk.kill_switches.clear_reconciliation_mismatches()

        if block_entries and not report.errors:
            self.execution.unblock_entries()

        log.info("reconciliation_complete", **report.summary())
        return report

    # ------------------------------------------------------------------ #
    async def _adopt_unknown(
        self,
        exchange: dict[str, Position],
        local: dict[str, Position],
        report: ReconciliationReport,
    ) -> None:
        """A position the exchange has and we do not.

        This is the dangerous case: it is real, it is losing or winning money
        right now, and nothing is protecting it. Adopt it, attach a stop
        immediately, and mark it for closure — we have no thesis for a position
        we did not knowingly open, so there is no basis for managing it.
        """
        for symbol, position in exchange.items():
            if symbol in local:
                continue

            log.critical(
                "unexpected_position_found",
                symbol=symbol,
                direction=position.direction.value,
                quantity=position.quantity,
                entry=position.entry_price,
            )

            position.adopted = True
            position.strategy = "adopted"
            position.metadata["adopted_at"] = self.clock.now_ms()
            position.metadata["reason"] = "found on the exchange, not in local state"

            # Give it a protective stop derived from a conservative fixed
            # distance — we have no ATR context for a position we did not plan.
            if position.stop_loss <= 0:
                distance = position.entry_price * self.config.stops.max_stop_pct
                position.stop_loss = (
                    position.entry_price - distance
                    if position.direction is Direction.LONG
                    else position.entry_price + distance
                )
                position.initial_stop = position.stop_loss
                position.initial_risk = distance * position.quantity

            local[symbol] = position
            report.adopted.append(symbol)

            protected = await self.execution._attach_protection(position)
            if not protected:
                report.errors.append(f"{symbol}: adopted but could not be protected")
                await self.execution.close_position(
                    symbol,
                    ExitReason.ADOPTED_POSITION,
                    "adopted position could not be protected",
                )
                continue

            if self.risk is not None:
                await self.execution._emit_risk_event(
                    RiskEventType.POSITION_ADOPTED,
                    "CRITICAL",
                    f"{symbol}: adopted an unexpected {position.direction.value} "
                    f"position of {position.quantity}; a protective stop was "
                    f"attached and it will be closed at the next opportunity",
                    symbol,
                )

    async def _close_phantom(
        self,
        exchange: dict[str, Position],
        local: dict[str, Position],
        report: ReconciliationReport,
    ) -> None:
        """A position we think we have and the exchange does not.

        Usually a stop filled while we were disconnected. The PnL is recovered
        from the actual fills rather than guessed.
        """
        for symbol in list(local):
            if symbol in exchange:
                continue

            position = local[symbol]
            log.warning(
                "phantom_position_removed",
                symbol=symbol,
                message="local state had a position the exchange does not",
            )

            exit_price = position.entry_price
            try:
                fills = await self.gateway.get_user_trades(
                    symbol, limit=20, start_ms=position.opened_at
                )
                closing = [
                    f
                    for f in fills
                    if str(f.get("side", "")).upper()
                    == ("SELL" if position.direction is Direction.LONG else "BUY")
                ]
                if closing:
                    total_qty = sum(float(f.get("qty", 0) or 0) for f in closing)
                    if total_qty > 0:
                        exit_price = (
                            sum(
                                float(f.get("price", 0) or 0) * float(f.get("qty", 0) or 0)
                                for f in closing
                            )
                            / total_qty
                        )
            except (ExchangeError, AttributeError, ValueError, ZeroDivisionError) as exc:
                report.errors.append(
                    f"{symbol}: closed locally but PnL could not be reconciled "
                    f"from fills ({exc}); entry price was used instead"
                )

            order = Order(
                client_order_id=f"recon_{symbol}",
                symbol=symbol,
                side=OrderSide.for_exit(position.direction),
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                filled_quantity=position.quantity,
                average_price=exit_price,
            )
            trade = self.execution._build_trade(position, order, exit_price, ExitReason.STOP_LOSS)
            self.execution.trades.append(trade)
            local.pop(symbol, None)
            report.closed_locally.append(symbol)

    async def _fix_quantities(
        self,
        exchange: dict[str, Position],
        local: dict[str, Position],
        report: ReconciliationReport,
    ) -> None:
        """Trust the exchange's quantity and re-protect at the true size."""
        for symbol, remote in exchange.items():
            position = local.get(symbol)
            if position is None:
                continue
            difference = safe_div(
                abs(position.quantity - remote.quantity),
                max(remote.quantity, 1e-12),
                0.0,
            )
            if difference <= QUANTITY_TOLERANCE:
                continue

            log.warning(
                "quantity_mismatch",
                symbol=symbol,
                local=position.quantity,
                exchange=remote.quantity,
            )
            position.quantity = remote.quantity
            position.entry_price = remote.entry_price or position.entry_price
            position.initial_risk = (
                abs(position.entry_price - position.initial_stop) * remote.quantity
            )
            report.quantity_mismatches.append(symbol)

            with contextlib.suppress(ExchangeError):
                await self.gateway.cancel_all_orders(symbol)
            if not await self.execution._attach_protection(position):
                report.errors.append(f"{symbol}: could not re-protect after a quantity mismatch")

    async def _cancel_orphans(
        self, exchange: dict[str, Position], orders: list[Order], report: ReconciliationReport
    ) -> None:
        """Cancel protective orders with no position behind them."""
        for order in orders:
            if not order.status.is_open:
                continue
            if order.symbol in exchange:
                continue
            if order.order_type not in {
                OrderType.STOP_MARKET,
                OrderType.STOP,
                OrderType.TAKE_PROFIT_MARKET,
                OrderType.TAKE_PROFIT,
                OrderType.TRAILING_STOP_MARKET,
            }:
                continue
            try:
                await self.gateway.cancel_order(order.symbol, order.client_order_id)
                report.orphan_orders_cancelled.append(order.client_order_id)
                log.info(
                    "orphan_order_cancelled",
                    symbol=order.symbol,
                    client_order_id=order.client_order_id,
                )
            except ExchangeError as exc:
                report.errors.append(f"{order.symbol}: could not cancel orphan order ({exc})")

    async def _ensure_protection(self, orders: list[Order], report: ReconciliationReport) -> None:
        """Every open position must have a live stop, or be closed."""
        from tradebot.execution.engine import positions_without_stops

        for unprotected in positions_without_stops(self.execution.positions, orders):
            symbol = unprotected.symbol
            log.critical("position_without_stop", symbol=symbol, detail=unprotected.detail)

            if await self.execution._attach_protection(unprotected.position):
                report.unprotected_fixed.append(symbol)
                continue

            # Could not protect it. An unprotected position is not something to
            # leave running while we work out why.
            await self.execution.close_position(
                symbol,
                ExitReason.RISK_EVENT,
                "position had no stop and one could not be placed",
            )
            report.unprotected_closed.append(symbol)

    # ------------------------------------------------------------------ #
    async def startup(self) -> ReconciliationReport:
        """Reconciliation on process start, before any entry is permitted."""
        log.info(
            "startup_reconciliation_beginning",
            message="entries are blocked until local state matches Binance",
        )
        self.execution.block_entries("startup reconciliation")
        report = await self.reconcile(block_entries=False)

        if report.errors:
            log.critical(
                "startup_reconciliation_incomplete",
                errors=report.errors,
                message="ENTRIES REMAIN BLOCKED",
            )
            return report

        self.execution.unblock_entries()
        log.info("startup_reconciliation_complete", **report.summary())
        return report
