"""Live market state.

One object owning everything the engine knows about the market right now:
candles, book tickers, mark prices, and — the part that matters most —
**per-symbol freshness**.

Before this existed the engine polled REST on a 15-second loop and had no way to
say "how old is my view of SOLUSDT?" (AUDIT_REPORT.md C-1). Freshness is not a
detail: a scalper acting on 15-second-old prices is trading a market that has
already moved, and the resulting losses look like strategy failure rather than
what they are.

The write side accepts data from either transport:

    WebSocket ──┐
                ├──> MarketState ──> Scanner ──> Strategies
    REST     ───┘    (freshness)

The read side does not know or care which one supplied it, but it can always ask
how old the data is, and `is_tradable()` refuses a symbol whose view has gone
stale. REST remains the fallback: when a symbol's stream goes quiet the feed
back-fills it, and the state records that the data came from a slower path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.logging import get_logger
from tradebot.core.types import BookTicker, Candle, MarkPriceInfo, Ticker24h
from tradebot.market.candles import CandleStore

log = get_logger(__name__)


class DataSource(StrEnum):
    WEBSOCKET = "WEBSOCKET"
    REST = "REST"
    NONE = "NONE"


class Freshness(StrEnum):
    LIVE = "LIVE"  # streaming, recent
    LAGGING = "LAGGING"  # older than expected but usable
    STALE = "STALE"  # too old to act on — entries refused


@dataclass(slots=True)
class SymbolState:
    """What we know about one symbol, and how old it is."""

    symbol: str
    last_candle_ms: float = 0.0
    last_book_ms: float = 0.0
    last_mark_ms: float = 0.0
    candle_source: DataSource = DataSource.NONE
    book: BookTicker | None = None
    mark: MarkPriceInfo | None = None
    ticker: Ticker24h | None = None
    candle_updates: int = 0
    book_updates: int = 0
    mark_updates: int = 0

    def age_sec(self, now: float) -> float:
        """Seconds since ANY update. Infinite when nothing has arrived."""
        newest = max(self.last_candle_ms, self.last_book_ms, self.last_mark_ms)
        return float("inf") if newest <= 0 else max(0.0, now - newest)

    def price_age_sec(self, now: float) -> float:
        """Seconds since a PRICE update.

        Deliberately excludes mark price: the mark-price array stream ticks
        every second for every symbol, so including it would make a symbol with
        a dead kline stream look perfectly fresh.
        """
        newest = max(self.last_candle_ms, self.last_book_ms)
        return float("inf") if newest <= 0 else max(0.0, now - newest)


class MarketState:
    """The single source of truth for live market data."""

    def __init__(
        self,
        candles: CandleStore,
        stale_after_sec: float = 30.0,
        lagging_after_sec: float = 10.0,
        clock: Clock | None = None,
    ) -> None:
        self.candles = candles
        self.stale_after_sec = stale_after_sec
        self.lagging_after_sec = lagging_after_sec
        self.clock = clock or SystemClock()

        self._symbols: dict[str, SymbolState] = {}
        self._subscribed: set[str] = set()

        # Global stream health, distinct from per-symbol freshness: a symbol can
        # be quiet because nothing is trading it, which is not the same as the
        # connection being down.
        self.last_stream_message: float = 0.0
        self.stream_connected = False
        self.rest_fallbacks = 0

    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        return self.clock.now()

    def state_for(self, symbol: str) -> SymbolState:
        found = self._symbols.get(symbol)
        if found is None:
            found = SymbolState(symbol=symbol)
            self._symbols[symbol] = found
        return found

    @property
    def subscribed(self) -> tuple[str, ...]:
        return tuple(sorted(self._subscribed))

    def set_subscribed(self, symbols: set[str]) -> None:
        self._subscribed = set(symbols)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def apply_candle(
        self, symbol: str, timeframe: str, candle: Candle, source: DataSource = DataSource.WEBSOCKET
    ) -> bool:
        """Record a candle. Returns True when a bar CLOSED.

        The closed flag is what downstream code waits on: strategies read closed
        bars only, so a forming-bar update refreshes the price without
        triggering a re-evaluation.
        """
        closed = self.candles.append(symbol, timeframe, candle)
        state = self.state_for(symbol)
        state.last_candle_ms = self._now()
        state.candle_source = source
        state.candle_updates += 1
        if source is DataSource.WEBSOCKET:
            self.last_stream_message = state.last_candle_ms
        return closed

    def apply_book(self, book: BookTicker, source: DataSource = DataSource.WEBSOCKET) -> None:
        state = self.state_for(book.symbol)
        state.book = book
        state.last_book_ms = self._now()
        state.book_updates += 1
        if source is DataSource.WEBSOCKET:
            self.last_stream_message = state.last_book_ms

    def apply_mark(self, mark: MarkPriceInfo, source: DataSource = DataSource.WEBSOCKET) -> None:
        state = self.state_for(mark.symbol)
        state.mark = mark
        state.last_mark_ms = self._now()
        state.mark_updates += 1
        if source is DataSource.WEBSOCKET:
            self.last_stream_message = state.last_mark_ms

    def apply_ticker(self, ticker: Ticker24h) -> None:
        self.state_for(ticker.symbol).ticker = ticker

    def record_rest_fallback(self, symbol: str) -> None:
        """Note that a symbol had to be back-filled over REST."""
        self.rest_fallbacks += 1
        self.state_for(symbol).candle_source = DataSource.REST

    def set_stream_connected(self, connected: bool) -> None:
        if connected != self.stream_connected:
            log.info("market_stream_connectivity", connected=connected)
        self.stream_connected = connected
        if connected:
            self.last_stream_message = self._now()

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def price(self, symbol: str) -> float:
        """Best current price: the book mid if we have one, else the last close.

        The book mid is preferred because it is what an order would actually
        transact against; a last close can be seconds old and on the wrong side
        of the spread.
        """
        state = self._symbols.get(symbol)
        if state is not None and state.book is not None:
            mid = state.book.mid
            if mid > 0:
                return mid
        return self.candles.price(symbol)

    def book(self, symbol: str) -> BookTicker | None:
        state = self._symbols.get(symbol)
        return state.book if state else None

    def mark(self, symbol: str) -> MarkPriceInfo | None:
        state = self._symbols.get(symbol)
        return state.mark if state else None

    def funding_rate(self, symbol: str) -> float:
        mark = self.mark(symbol)
        return mark.funding_rate if mark else 0.0

    def seconds_to_funding(self, symbol: str) -> float:
        mark = self.mark(symbol)
        if mark is None or mark.next_funding_time <= 0:
            return float("inf")
        return max(0.0, (mark.next_funding_time - self.clock.now_ms()) / 1000.0)

    # ------------------------------------------------------------------ #
    def freshness(self, symbol: str) -> Freshness:
        """How current our view of this symbol is."""
        state = self._symbols.get(symbol)
        if state is None:
            return Freshness.STALE
        age = state.price_age_sec(self._now())
        if age > self.stale_after_sec:
            return Freshness.STALE
        if age > self.lagging_after_sec:
            return Freshness.LAGGING
        return Freshness.LIVE

    def age_sec(self, symbol: str) -> float:
        state = self._symbols.get(symbol)
        return state.price_age_sec(self._now()) if state else float("inf")

    def is_tradable(self, symbol: str) -> bool:
        """May a NEW position be opened on this symbol?

        Stale data means no. This is the check that stops the engine acting on a
        price that is no longer true — and it is deliberately about entries
        only: an exit on stale data is still far better than no exit.
        """
        return self.freshness(symbol) is not Freshness.STALE

    def stale_symbols(self) -> list[str]:
        """Subscribed symbols whose data has gone stale."""
        now = self._now()
        return sorted(
            symbol
            for symbol in self._subscribed
            if self.state_for(symbol).price_age_sec(now) > self.stale_after_sec
        )

    def stream_age_sec(self) -> float:
        """Seconds since ANY stream message, across all symbols."""
        if self.last_stream_message <= 0:
            return float("inf")
        return max(0.0, self._now() - self.last_stream_message)

    def stream_is_stale(self) -> bool:
        """True when the connection itself looks dead, not just one symbol."""
        return self.stream_age_sec() > self.stale_after_sec

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        now = self._now()
        live = lagging = stale = 0
        for symbol in self._subscribed:
            age = self.state_for(symbol).price_age_sec(now)
            if age > self.stale_after_sec:
                stale += 1
            elif age > self.lagging_after_sec:
                lagging += 1
            else:
                live += 1

        return {
            "stream_connected": self.stream_connected,
            "stream_age_sec": round(self.stream_age_sec(), 1) if self.last_stream_message else None,
            "subscribed": len(self._subscribed),
            "live": live,
            "lagging": lagging,
            "stale": stale,
            "rest_fallbacks": self.rest_fallbacks,
            "updates": {
                "candles": sum(s.candle_updates for s in self._symbols.values()),
                "books": sum(s.book_updates for s in self._symbols.values()),
                "marks": sum(s.mark_updates for s in self._symbols.values()),
            },
        }

    def symbol_report(self, limit: int = 30) -> list[dict[str, Any]]:
        """Per-symbol freshness, for the dashboard."""
        now = self._now()
        rows = []
        for symbol in sorted(self._subscribed):
            state = self.state_for(symbol)
            age = state.price_age_sec(now)
            rows.append(
                {
                    "symbol": symbol,
                    "freshness": self.freshness(symbol).value,
                    "age_sec": round(age, 1) if age != float("inf") else None,
                    "source": state.candle_source.value,
                    "price": self.price(symbol),
                    "spread_bps": round(state.book.spread_bps, 2) if state.book else None,
                }
            )
        rows.sort(key=lambda r: (r["age_sec"] is None, r["age_sec"] or 0), reverse=True)
        return rows[:limit]
