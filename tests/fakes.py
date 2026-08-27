"""Test doubles.

``FakeGateway`` implements the exchange interface over scripted data, so the
scanner, strategies, risk engine and execution engine can all be driven end to
end without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradebot.core.types import (
    AccountState,
    BookTicker,
    Candle,
    Direction,
    MarkPriceInfo,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SymbolInfo,
    Ticker24h,
)


def make_symbol_info(
    symbol: str,
    *,
    tick: float = 0.01,
    step: float = 0.001,
    min_qty: float = 0.001,
    min_notional: float = 5.0,
    quote: str = "USDT",
    status: str = "TRADING",
    contract_type: str = "PERPETUAL",
    max_leverage: int = 20,
) -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol,
        base_asset=symbol.replace(quote, ""),
        quote_asset=quote,
        status=status,
        contract_type=contract_type,
        price_precision=2,
        quantity_precision=3,
        tick_size=tick,
        step_size=step,
        min_qty=min_qty,
        max_qty=1_000_000.0,
        min_notional=min_notional,
        market_min_qty=min_qty,
        market_max_qty=1_000_000.0,
        max_leverage=max_leverage,
    )


@dataclass
class FakeGateway:
    """Scripted exchange. Records everything it is asked to do."""

    symbols: dict[str, SymbolInfo] = field(default_factory=dict)
    klines: dict[tuple[str, str], list[Candle]] = field(default_factory=dict)
    books: dict[str, BookTicker] = field(default_factory=dict)
    tickers: dict[str, Ticker24h] = field(default_factory=dict)
    marks: dict[str, MarkPriceInfo] = field(default_factory=dict)
    account: AccountState | None = None
    positions: dict[str, Position] = field(default_factory=dict)
    open_orders: list[Order] = field(default_factory=list)

    placed: list[OrderIntent] = field(default_factory=list)
    protective: list[dict[str, Any]] = field(default_factory=list)
    cancelled: list[tuple[str, str]] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    leverage_set: dict[str, int] = field(default_factory=dict)

    kline_calls: int = 0
    fail_symbols: set[str] = field(default_factory=set)
    fill_price_override: dict[str, float] = field(default_factory=dict)
    reject_next_order: bool = False

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_symbols(self) -> dict[str, SymbolInfo]:
        return self.symbols

    def symbol_info(self, symbol: str) -> SymbolInfo | None:
        return self.symbols.get(symbol)

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Candle]:
        self.kline_calls += 1
        if symbol in self.fail_symbols:
            raise ConnectionError(f"simulated failure for {symbol}")
        return self.klines.get((symbol, interval), [])[-limit:]

    async def get_book_ticker(self, symbol: str | None = None) -> dict[str, BookTicker]:
        if symbol:
            found = self.books.get(symbol)
            return {symbol: found} if found else {}
        return dict(self.books)

    async def get_ticker_24h(self, symbol: str | None = None) -> dict[str, Ticker24h]:
        if symbol:
            found = self.tickers.get(symbol)
            return {symbol: found} if found else {}
        return dict(self.tickers)

    async def get_mark_price(self, symbol: str | None = None) -> dict[str, MarkPriceInfo]:
        if symbol:
            found = self.marks.get(symbol)
            return {symbol: found} if found else {}
        return dict(self.marks)

    async def get_account(self) -> AccountState:
        return self.account or AccountState(
            total_balance=75.0,
            available_balance=75.0,
            equity=75.0,
            unrealized_pnl=0.0,
            margin_used=0.0,
            timestamp=0,
        )

    async def get_positions(self) -> dict[str, Position]:
        return dict(self.positions)

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol:
            return [o for o in self.open_orders if o.symbol == symbol]
        return list(self.open_orders)

    async def set_leverage(self, symbol: str, leverage: int) -> int:
        info = self.symbols.get(symbol)
        applied = min(leverage, info.max_leverage) if info else leverage
        self.leverage_set[symbol] = applied
        return applied

    async def place_order(self, intent: OrderIntent) -> Order:
        self.placed.append(intent)
        if self.reject_next_order:
            self.reject_next_order = False
            return Order(
                client_order_id=intent.client_order_id,
                symbol=intent.symbol,
                side=intent.side,
                order_type=intent.order_type,
                quantity=intent.quantity,
                status=OrderStatus.REJECTED,
                intent_id=intent.intent_id,
                error="simulated rejection",
            )
        price = self.fill_price_override.get(
            intent.symbol,
            intent.price or intent.metadata.get("reference_price", 100.0),
        )
        return Order(
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=intent.quantity,
            price=intent.price,
            status=OrderStatus.FILLED,
            exchange_order_id=f"ex{len(self.placed)}",
            filled_quantity=intent.quantity,
            average_price=price,
            intent_id=intent.intent_id,
        )

    async def place_protective_order(
        self,
        symbol: str,
        order_type: OrderType,
        stop_price: float,
        quantity: float,
        direction_sign: int,
        client_order_id: str,
    ) -> Order:
        self.protective.append(
            {
                "symbol": symbol,
                "type": order_type,
                "stop_price": stop_price,
                "quantity": quantity,
                "sign": direction_sign,
                "client_order_id": client_order_id,
            }
        )
        return Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.SELL if direction_sign > 0 else OrderSide.BUY,
            order_type=order_type,
            quantity=quantity,
            stop_price=stop_price,
            status=OrderStatus.NEW,
            exchange_order_id=f"prot{len(self.protective)}",
        )

    async def query_order(self, symbol: str, client_order_id: str) -> Order | None:
        return next(
            (
                o
                for o in self.open_orders
                if o.client_order_id == client_order_id and o.symbol == symbol
            ),
            None,
        )

    async def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        self.cancelled.append((symbol, client_order_id))
        self.open_orders = [o for o in self.open_orders if o.client_order_id != client_order_id]
        return True

    async def cancel_all_orders(self, symbol: str) -> bool:
        self.cancelled.append((symbol, "*"))
        self.open_orders = [o for o in self.open_orders if o.symbol != symbol]
        return True

    async def close_position(self, symbol: str, position: Position, client_order_id: str) -> Order:
        self.closed.append(symbol)
        price = self.fill_price_override.get(symbol, position.entry_price)
        self.positions.pop(symbol, None)
        return Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.for_exit(position.direction),
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            status=OrderStatus.FILLED,
            filled_quantity=position.quantity,
            average_price=price,
            reduce_only=True,
        )


def book_for(symbol: str, mid: float, spread_bps: float = 1.0, qty: float = 1000.0) -> BookTicker:
    half = mid * (spread_bps / 10_000.0) / 2.0
    return BookTicker(symbol, mid - half, qty, mid + half, qty, 0)


def ticker_for(symbol: str, price: float, quote_volume: float, change: float = 0.01) -> Ticker24h:
    return Ticker24h(
        symbol,
        price,
        change,
        price * 1.05,
        price * 0.95,
        quote_volume / max(price, 1e-9),
        quote_volume,
        10_000,
        0,
    )


def mark_for(
    symbol: str, price: float, funding: float = 0.0001, next_funding_ms: int = 0
) -> MarkPriceInfo:
    return MarkPriceInfo(symbol, price, price, funding, next_funding_ms, 0)


def position_for(
    symbol: str,
    direction: Direction = Direction.LONG,
    quantity: float = 1.0,
    entry: float = 100.0,
    stop: float = 98.0,
    target: float = 104.0,
    leverage: int = 3,
    strategy: str = "momentum",
    opened_at: int = 0,
) -> Position:
    from tradebot.core.types import MarketRegime

    return Position(
        position_id=f"p:{symbol}",
        symbol=symbol,
        direction=direction,
        quantity=quantity,
        entry_price=entry,
        leverage=leverage,
        stop_loss=stop,
        take_profit=target,
        strategy=strategy,
        regime=MarketRegime.STRONG_TREND,
        opened_at=opened_at,
        entry_notional=quantity * entry,
        initial_stop=stop,
        initial_risk=abs(entry - stop) * quantity,
        highest_price=entry,
        lowest_price=entry,
    )
