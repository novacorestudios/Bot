"""The exchange gateway interface.

Everything above this line is written once and runs identically in BACKTEST,
PAPER and LIVE. Three implementations satisfy it:

* ``exchange/binance/rest.py``  — real Binance USDⓈ-M Futures
* ``paper/broker.py``           — simulated fills against a live feed
* ``backtesting/engine.py``     — simulated fills against historical bars

Keeping this interface narrow is what makes it possible to test the trading
logic without a network, and to be confident that a paper run exercises the same
code path as a live one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tradebot.core.types import (
    AccountState,
    BookTicker,
    Candle,
    MarkPriceInfo,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    Position,
    SymbolInfo,
    Ticker24h,
)


@runtime_checkable
class ExchangeGateway(Protocol):
    """Order and account operations."""

    async def connect(self) -> None: ...
    async def close(self) -> None: ...

    # -- reference data ----------------------------------------------------- #
    async def load_symbols(self) -> dict[str, SymbolInfo]: ...
    def symbol_info(self, symbol: str) -> SymbolInfo | None: ...

    # -- market data -------------------------------------------------------- #
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Candle]: ...
    async def get_book_ticker(self, symbol: str | None = None) -> dict[str, BookTicker]: ...
    async def get_ticker_24h(self, symbol: str | None = None) -> dict[str, Ticker24h]: ...
    async def get_mark_price(self, symbol: str | None = None) -> dict[str, MarkPriceInfo]: ...

    # -- account ------------------------------------------------------------ #
    async def get_account(self) -> AccountState: ...
    async def get_positions(self) -> dict[str, Position]: ...
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]: ...

    # -- trading ------------------------------------------------------------ #
    async def set_leverage(self, symbol: str, leverage: int) -> int: ...
    async def place_order(self, intent: OrderIntent) -> Order: ...
    async def place_protective_order(
        self,
        symbol: str,
        order_type: OrderType,
        stop_price: float,
        quantity: float,
        direction_sign: int,
        client_order_id: str,
    ) -> Order: ...
    async def query_order(self, symbol: str, client_order_id: str) -> Order | None: ...
    async def cancel_order(self, symbol: str, client_order_id: str) -> bool: ...
    async def cancel_all_orders(self, symbol: str) -> bool: ...
    async def close_position(
        self, symbol: str, position: Position, client_order_id: str
    ) -> Order: ...


class GatewayCapabilities:
    """What a given gateway can actually do — checked at startup, not at 3am."""

    def __init__(
        self, places_real_orders: bool, supports_websocket: bool, supports_funding: bool, name: str
    ) -> None:
        self.places_real_orders = places_real_orders
        self.supports_websocket = supports_websocket
        self.supports_funding = supports_funding
        self.name = name

    def __repr__(self) -> str:
        return (
            f"GatewayCapabilities(name={self.name!r}, places_real_orders={self.places_real_orders})"
        )


def order_is_settled(status: OrderStatus) -> bool:
    """True when no further updates are expected for this order."""
    return status.is_terminal
