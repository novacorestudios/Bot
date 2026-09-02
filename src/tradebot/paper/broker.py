"""Simulated broker for paper trading.

Implements the same gateway interface as the real Binance client, against a
**live** market feed. The engine above it cannot tell the difference, which is
the point: a paper run must exercise the identical code path a live run would.

Fills are modelled **pessimistically on purpose**. A paper broker that fills at
the mid price, instantly, in full, produces results that are strictly better
than live — and the gap only shows up after real money is committed. So:

* **Latency** — a configurable delay before the fill is decided, during which
  the price may move away.
* **Adverse slippage** — slippage is biased against us
  (``adverse_slippage_probability``), because the fills we get are the ones the
  market was willing to give.
* **Partial fills** — a configurable share of orders fill partially, exercising
  the engine's handling of a position smaller than requested.
* **Rejections** — a small rate, so the rejection path is exercised routinely
  rather than discovered in production.
* **Funding** — charged on positions that cross a funding timestamp.

Being harsher than reality is the safe direction to be wrong in.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.config import PaperConfig
from tradebot.core.errors import ExchangeError, FilterViolationError
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import from_bps, round_quantity, safe_div
from tradebot.core.types import (
    AccountState,
    BookTicker,
    Direction,
    Fill,
    MarkPriceInfo,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SymbolInfo,
    Ticker24h,
    new_id,
)
from tradebot.exchange.binance.filters import validate_order

log = get_logger(__name__)


@dataclass(slots=True)
class SimulatedProtectiveOrder:
    """A resting stop or take-profit the simulator must watch."""

    client_order_id: str
    symbol: str
    order_type: OrderType
    stop_price: float
    direction_sign: int
    created_at: int


@dataclass(slots=True)
class PaperAccount:
    balance: float
    equity: float
    margin_used: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)


class PaperBroker:
    """Simulated exchange gateway driven by a live market feed."""

    def __init__(
        self,
        config: PaperConfig,
        market_source: Any,
        initial_balance: float = 75.0,
        taker_fee: float = 0.0004,
        maker_fee: float = 0.0002,
        clock: Clock | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.market = market_source  # provides symbols/prices/books
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.clock = clock or SystemClock()
        # Simulation, not cryptography — and seedable on purpose, so a paper
        # run can be reproduced exactly when diagnosing a fill.
        self._rng = random.Random(seed)  # noqa: S311  # nosec B311

        self.account = PaperAccount(balance=initial_balance, equity=initial_balance)
        self.orders: dict[str, Order] = {}
        self.protective: dict[str, list[SimulatedProtectiveOrder]] = {}
        self.leverage: dict[str, int] = {}
        self.fills: list[Fill] = []

        self.rejected_count = 0
        self.partial_count = 0
        self.total_slippage = 0.0
        self.filled_count = 0

    # ------------------------------------------------------------------ #
    # Gateway interface — reference and market data delegate to the feed
    # ------------------------------------------------------------------ #
    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_symbols(self) -> dict[str, SymbolInfo]:
        return await self.market.load_symbols()

    @property
    def symbols(self) -> dict[str, SymbolInfo]:
        return self.market.symbols

    def symbol_info(self, symbol: str) -> SymbolInfo | None:
        return self.market.symbol_info(symbol)

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list:
        return await self.market.get_klines(symbol, interval, limit, start_ms, end_ms)

    async def get_book_ticker(self, symbol: str | None = None) -> dict[str, BookTicker]:
        return await self.market.get_book_ticker(symbol)

    async def get_ticker_24h(self, symbol: str | None = None) -> dict[str, Ticker24h]:
        return await self.market.get_ticker_24h(symbol)

    async def get_mark_price(self, symbol: str | None = None) -> dict[str, MarkPriceInfo]:
        return await self.market.get_mark_price(symbol)

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    async def get_account(self) -> AccountState:
        unrealized = self._unrealized()
        self.account.equity = self.account.balance + unrealized
        return AccountState(
            total_balance=self.account.balance,
            available_balance=max(0.0, self.account.balance - self.account.margin_used),
            equity=self.account.equity,
            unrealized_pnl=unrealized,
            margin_used=self.account.margin_used,
            timestamp=self.clock.now_ms(),
            positions=dict(self.account.positions),
        )

    async def get_positions(self) -> dict[str, Position]:
        return dict(self.account.positions)

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        orders = [o for o in self.orders.values() if o.status.is_open]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def set_leverage(self, symbol: str, leverage: int) -> int:
        info = self.symbol_info(symbol)
        applied = min(leverage, info.max_leverage) if info else leverage
        self.leverage[symbol] = applied
        return applied

    # ------------------------------------------------------------------ #
    # Order placement
    # ------------------------------------------------------------------ #
    async def place_order(self, intent: OrderIntent) -> Order:
        """Simulate an entry order, with latency, slippage, partials and rejections."""
        info = self.symbol_info(intent.symbol)
        if info is None:
            raise ExchangeError("unknown symbol", symbol=intent.symbol)

        reference = (
            intent.price or intent.metadata.get("reference_price") or self._price(intent.symbol)
        )
        validation = validate_order(
            info, intent.quantity, intent.price, intent.order_type, reference_price=reference
        )
        if not validation.ok:
            raise FilterViolationError(
                f"order fails filter validation: {validation.reason}",
                detail=validation.detail,
                symbol=intent.symbol,
            )

        order = Order(
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=validation.quantity,
            price=validation.price,
            status=OrderStatus.PENDING,
            intent_id=intent.intent_id,
        )
        self.orders[order.client_order_id] = order

        await self._simulate_latency()

        if self._rng.random() < self.config.reject_probability:
            order.status = OrderStatus.REJECTED
            order.error = "simulated exchange rejection"
            self.rejected_count += 1
            log.info(
                "paper_order_rejected", symbol=intent.symbol, client_order_id=order.client_order_id
            )
            return order

        market_price = self._price(intent.symbol)
        if market_price <= 0:
            order.status = OrderStatus.REJECTED
            order.error = "no market price available"
            return order

        fill_price = self._fill_price(intent.symbol, intent.direction, market_price)
        quantity = validation.quantity

        if self._rng.random() < self.config.partial_fill_probability:
            # Fill between 30% and 90%, step-aligned. The engine must cope with
            # holding less than it asked for.
            fraction = self._rng.uniform(0.3, 0.9)
            partial = round_quantity(quantity * fraction, info.step_size)
            if partial >= info.min_qty and partial * fill_price >= info.min_notional:
                quantity = partial
                self.partial_count += 1
                order.status = OrderStatus.PARTIALLY_FILLED

        notional = quantity * fill_price
        fee = notional * self.taker_fee
        leverage = max(1, self.leverage.get(intent.symbol, intent.leverage))
        margin = notional / leverage

        if margin + fee > self.account.balance - self.account.margin_used:
            order.status = OrderStatus.REJECTED
            order.error = "insufficient simulated margin"
            self.rejected_count += 1
            return order

        slippage = abs(fill_price - market_price)
        self.total_slippage += safe_div(slippage, market_price, 0.0)
        self.filled_count += 1

        self.account.balance -= fee
        self.account.fees_paid += fee
        self.account.margin_used += margin

        order.filled_quantity = quantity
        order.average_price = fill_price
        order.exchange_order_id = f"paper-{len(self.orders)}"
        order.updated_at = self.clock.now_ms()
        if order.status is not OrderStatus.PARTIALLY_FILLED:
            order.status = OrderStatus.FILLED

        fill = Fill(
            fill_id=new_id("f_"),
            order_id=order.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            price=fill_price,
            quantity=quantity,
            commission=fee,
            commission_asset="USDT",
            timestamp=self.clock.now_ms(),
            is_maker=False,
        )
        order.fills.append(fill)
        self.fills.append(fill)

        self.account.positions[intent.symbol] = Position(
            position_id=new_id("pp_"),
            symbol=intent.symbol,
            direction=intent.direction,
            quantity=quantity,
            entry_price=fill_price,
            leverage=leverage,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            strategy=intent.strategy,
            regime=intent.regime,
            opened_at=self.clock.now_ms(),
            entry_notional=notional,
            allocated_initial_margin=margin,
            entry_fee=fee,
            entry_slippage=slippage * quantity,
            initial_stop=intent.stop_loss,
            initial_risk=abs(fill_price - intent.stop_loss) * quantity,
            highest_price=fill_price,
            lowest_price=fill_price,
            entry_order_id=order.client_order_id,
            opportunity_score=intent.opportunity_score,
            expected_net_edge=intent.expected_net_edge,
            metadata={"margin": margin, "simulated": True, **intent.metadata},
        )

        log.info(
            "paper_order_filled",
            symbol=intent.symbol,
            direction=intent.direction.value,
            quantity=quantity,
            price=fill_price,
            slippage_bps=round(safe_div(slippage, market_price, 0.0) * 10_000, 2),
            partial=order.status is OrderStatus.PARTIALLY_FILLED,
        )
        return order

    async def place_protective_order(
        self,
        symbol: str,
        order_type: OrderType,
        stop_price: float,
        quantity: float,
        direction_sign: int,
        client_order_id: str,
    ) -> Order:
        """Register a resting stop/take-profit that :meth:`poll` will watch."""
        self.protective.setdefault(symbol, []).append(
            SimulatedProtectiveOrder(
                client_order_id=client_order_id,
                symbol=symbol,
                order_type=order_type,
                stop_price=stop_price,
                direction_sign=direction_sign,
                created_at=self.clock.now_ms(),
            )
        )
        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.SELL if direction_sign > 0 else OrderSide.BUY,
            order_type=order_type,
            quantity=quantity,
            stop_price=stop_price,
            status=OrderStatus.NEW,
            reduce_only=True,
            close_position=True,
            exchange_order_id=f"paper-prot-{len(self.orders)}",
        )
        self.orders[client_order_id] = order
        return order

    async def query_order(self, symbol: str, client_order_id: str) -> Order | None:
        order = self.orders.get(client_order_id)
        return order if order and order.symbol == symbol else None

    async def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        order = self.orders.get(client_order_id)
        if order and order.status.is_open:
            order.status = OrderStatus.CANCELED
        self.protective[symbol] = [
            p for p in self.protective.get(symbol, []) if p.client_order_id != client_order_id
        ]
        return True

    async def cancel_all_orders(self, symbol: str) -> bool:
        for order in self.orders.values():
            if order.symbol == symbol and order.status.is_open:
                order.status = OrderStatus.CANCELED
        self.protective.pop(symbol, None)
        return True

    async def close_position(self, symbol: str, position: Position, client_order_id: str) -> Order:
        """Flatten a simulated position at the current price, with slippage."""
        await self._simulate_latency()
        market_price = self._price(symbol)
        if market_price <= 0:
            raise ExchangeError("no market price to close against", symbol=symbol)

        # Exiting is marketable in the opposite direction, so slippage is adverse
        # to the exit as well.
        exit_price = self._fill_price(symbol, position.direction.opposite(), market_price)
        self._settle(position, exit_price)

        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.for_exit(position.direction),
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            status=OrderStatus.FILLED,
            filled_quantity=position.quantity,
            average_price=exit_price,
            reduce_only=True,
            exchange_order_id=f"paper-close-{len(self.orders)}",
        )
        self.orders[client_order_id] = order
        self.protective.pop(symbol, None)
        return order

    # ------------------------------------------------------------------ #
    # Simulation loop
    # ------------------------------------------------------------------ #
    async def poll(self) -> list[tuple[str, str, float]]:
        """Check resting protective orders against current prices.

        Returns ``(symbol, client_order_id, fill_price)`` for each triggered
        order, so the execution engine can process the exit exactly as it would
        an ``ORDER_TRADE_UPDATE`` from the real user stream.
        """
        triggered: list[tuple[str, str, float]] = []

        for symbol in list(self.protective):
            position = self.account.positions.get(symbol)
            if position is None:
                self.protective.pop(symbol, None)
                continue

            price = self._price(symbol)
            if price <= 0:
                continue

            for resting in list(self.protective.get(symbol, [])):
                hit = (
                    (
                        price <= resting.stop_price
                        if resting.direction_sign > 0
                        else price >= resting.stop_price
                    )
                    if resting.order_type in {OrderType.STOP_MARKET, OrderType.STOP}
                    else (
                        price >= resting.stop_price
                        if resting.direction_sign > 0
                        else price <= resting.stop_price
                    )
                )
                if not hit:
                    continue

                # A stop fills at its trigger or worse; a take-profit at its level.
                if resting.order_type in {OrderType.STOP_MARKET, OrderType.STOP}:
                    fill_price = (
                        min(resting.stop_price, price)
                        if resting.direction_sign > 0
                        else max(resting.stop_price, price)
                    )
                    fill_price = self._apply_slippage(fill_price, -resting.direction_sign)
                else:
                    fill_price = resting.stop_price

                self._settle(position, fill_price)
                order = self.orders.get(resting.client_order_id)
                if order is not None:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_price = fill_price
                triggered.append((symbol, resting.client_order_id, fill_price))
                self.protective.pop(symbol, None)
                break

        return triggered

    def apply_funding(self, symbol: str, rate: float) -> float:
        """Charge funding on an open position. Returns the amount paid."""
        if not self.config.apply_funding:
            return 0.0
        position = self.account.positions.get(symbol)
        if position is None or rate == 0.0:
            return 0.0
        payment = position.quantity * position.entry_price * rate * position.direction.sign
        position.funding_paid += payment
        self.account.balance -= payment
        self.account.funding_paid += payment
        return payment

    # ------------------------------------------------------------------ #
    def _settle(self, position: Position, exit_price: float) -> None:
        """Realise a position's PnL into the simulated balance."""
        gross = (exit_price - position.entry_price) * position.quantity * position.direction.sign
        exit_fee = exit_price * position.quantity * self.taker_fee

        self.account.balance += gross - exit_fee
        self.account.fees_paid += exit_fee
        self.account.realized_pnl += gross - exit_fee - position.entry_fee
        self.account.margin_used = max(
            0.0,
            self.account.margin_used
            - (
                position.allocated_initial_margin
                if position.allocated_initial_margin > 0
                else position.metadata.get("margin", 0.0)
            ),
        )
        self.account.positions.pop(position.symbol, None)

    def _unrealized(self) -> float:
        total = 0.0
        for symbol, position in self.account.positions.items():
            price = self._price(symbol)
            if price > 0:
                total += position.unrealized_pnl(price)
        return total

    def _price(self, symbol: str) -> float:
        return float(self.market.price(symbol))

    def _fill_price(self, symbol: str, direction: Direction, market_price: float) -> float:
        """Apply spread and slippage in the direction that costs us."""
        book = getattr(self.market, "book", lambda _s: None)(symbol)
        if book is not None and book.bid_price > 0 and book.ask_price > 0:
            base = book.ask_price if direction is Direction.LONG else book.bid_price
        else:
            base = market_price
        return self._apply_slippage(base, direction.sign)

    def _apply_slippage(self, price: float, sign: int) -> float:
        """Slippage biased against us, per ``adverse_slippage_probability``."""
        magnitude = from_bps(self.config.slippage_bps) * self._rng.uniform(0.4, 1.6)
        adverse = self._rng.random() < self.config.adverse_slippage_probability
        direction = sign if adverse else -sign
        return price * (1 + magnitude * direction)

    async def _simulate_latency(self) -> None:
        if self.config.latency_ms <= 0 and self.config.latency_jitter_ms <= 0:
            return
        delay = self.config.latency_ms + self._rng.uniform(0.0, self.config.latency_jitter_ms)
        await asyncio.sleep(delay / 1000.0)

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        return {
            "balance": round(self.account.balance, 6),
            "equity": round(self.account.balance + self._unrealized(), 6),
            "realized_pnl": round(self.account.realized_pnl, 6),
            "fees_paid": round(self.account.fees_paid, 6),
            "funding_paid": round(self.account.funding_paid, 6),
            "open_positions": len(self.account.positions),
            "orders": len(self.orders),
            "filled": self.filled_count,
            "rejected": self.rejected_count,
            "partial_fills": self.partial_count,
            "mean_slippage_bps": round(
                safe_div(self.total_slippage, self.filled_count, 0.0) * 10_000, 3
            ),
        }
