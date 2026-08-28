"""Live market state and freshness.

The property under test is not "does it store a price" but "does it know how
old that price is, and does it refuse to trade on it once it is too old".
"""

from __future__ import annotations

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.types import BookTicker, Candle, MarkPriceInfo
from tradebot.market.candles import CandleStore
from tradebot.market.state import DataSource, Freshness, MarketState


@pytest.fixture
def state(clock: VirtualClock) -> MarketState:
    return MarketState(
        CandleStore(500),
        stale_after_sec=30.0,
        lagging_after_sec=10.0,
        clock=clock,
    )


def make_candle(clock: VirtualClock, close: float = 100.0, closed: bool = True) -> Candle:
    now = clock.now_ms()
    return Candle(
        open_time=now - 60_000,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=10.0,
        close_time=now,
        closed=closed,
    )


def make_book(symbol: str = "BTCUSDT", bid: float = 99.9, ask: float = 100.1) -> BookTicker:
    return BookTicker(
        symbol=symbol, bid_price=bid, bid_qty=5.0, ask_price=ask, ask_qty=5.0, timestamp=0
    )


def make_mark(symbol: str = "BTCUSDT", rate: float = 0.0001) -> MarkPriceInfo:
    return MarkPriceInfo(
        symbol=symbol,
        mark_price=100.0,
        index_price=100.0,
        funding_rate=rate,
        next_funding_time=1_700_000_600_000,
        timestamp=0,
    )


class TestFreshness:
    def test_unknown_symbol_is_stale_not_live(self, state: MarketState) -> None:
        """The default must be the conservative one: never traded, never seen."""
        assert state.freshness("NEWUSDT") is Freshness.STALE
        assert state.is_tradable("NEWUSDT") is False
        assert state.age_sec("NEWUSDT") == float("inf")

    def test_transitions_live_to_lagging_to_stale(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        state.apply_candle("BTCUSDT", "1m", make_candle(clock))
        assert state.freshness("BTCUSDT") is Freshness.LIVE
        assert state.is_tradable("BTCUSDT") is True

        clock.advance(11)
        assert state.freshness("BTCUSDT") is Freshness.LAGGING
        assert state.is_tradable("BTCUSDT") is True  # usable, just flagged

        clock.advance(20)  # 31s total
        assert state.freshness("BTCUSDT") is Freshness.STALE
        assert state.is_tradable("BTCUSDT") is False

    def test_a_new_update_restores_freshness(self, state: MarketState, clock: VirtualClock) -> None:
        state.apply_candle("BTCUSDT", "1m", make_candle(clock))
        clock.advance(60)
        assert state.is_tradable("BTCUSDT") is False
        state.apply_book(make_book())
        assert state.freshness("BTCUSDT") is Freshness.LIVE

    def test_mark_price_alone_does_not_count_as_fresh(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        """The whole point of `price_age_sec`.

        `!markPrice@arr@1s` ticks every second for every symbol on the exchange.
        If it counted, a symbol whose kline and book streams were both dead
        would look perfectly healthy and keep taking entries.
        """
        state.apply_candle("BTCUSDT", "1m", make_candle(clock))
        clock.advance(60)
        state.apply_mark(make_mark())
        assert state.freshness("BTCUSDT") is Freshness.STALE
        assert state.is_tradable("BTCUSDT") is False
        # ...but the mark price itself is still readable, for funding.
        assert state.funding_rate("BTCUSDT") == 0.0001


class TestPrices:
    def test_book_mid_is_preferred_over_last_close(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        state.apply_candle("BTCUSDT", "1m", make_candle(clock, close=100.0))
        assert state.price("BTCUSDT") == 100.0
        state.apply_book(make_book(bid=101.0, ask=101.2))
        assert state.price("BTCUSDT") == pytest.approx(101.1)

    def test_falls_back_to_candle_close_without_a_book(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        state.apply_candle("ETHUSDT", "1m", make_candle(clock, close=3000.0))
        assert state.price("ETHUSDT") == 3000.0

    def test_unknown_symbol_price_is_zero(self, state: MarketState) -> None:
        assert state.price("NOPEUSDT") == 0.0

    def test_seconds_to_funding(self, state: MarketState, clock: VirtualClock) -> None:
        state.apply_mark(make_mark())
        expected = (1_700_000_600_000 - clock.now_ms()) / 1000.0
        assert state.seconds_to_funding("BTCUSDT") == pytest.approx(expected)

    def test_seconds_to_funding_unknown_is_infinite(self, state: MarketState) -> None:
        """Infinity means "no funding constraint", which is the safe default."""
        assert state.seconds_to_funding("NOPEUSDT") == float("inf")


class TestClosedBarSignal:
    def test_apply_candle_reports_bar_close(self, state: MarketState, clock: VirtualClock) -> None:
        forming = state.apply_candle("BTCUSDT", "1m", make_candle(clock, closed=False))
        assert forming is False
        clock.advance(60)
        closed = state.apply_candle("BTCUSDT", "1m", make_candle(clock, closed=True))
        assert closed is True


class TestStreamHealth:
    def test_stale_symbols_only_covers_the_subscribed_set(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        state.set_subscribed({"BTCUSDT", "ETHUSDT"})
        state.apply_candle("BTCUSDT", "1m", make_candle(clock))
        state.apply_candle("SOLUSDT", "1m", make_candle(clock))  # not subscribed
        clock.advance(45)
        state.apply_book(make_book("BTCUSDT"))

        # ETHUSDT was subscribed and never delivered; SOLUSDT is not our problem.
        assert state.stale_symbols() == ["ETHUSDT"]

    def test_stream_staleness_is_global_not_per_symbol(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        state.set_stream_connected(True)
        state.apply_candle("BTCUSDT", "1m", make_candle(clock))
        assert state.stream_is_stale() is False
        clock.advance(45)
        assert state.stream_is_stale() is True

    def test_rest_writes_do_not_mask_a_dead_stream(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        """A REST backfill keeps the symbol tradable but must not claim the
        stream is alive — otherwise the kill switch never fires."""
        state.apply_candle("BTCUSDT", "1m", make_candle(clock), DataSource.WEBSOCKET)
        clock.advance(45)
        state.apply_book(make_book(), DataSource.REST)
        assert state.is_tradable("BTCUSDT") is True
        assert state.stream_is_stale() is True

    def test_rest_fallback_is_counted(self, state: MarketState) -> None:
        state.record_rest_fallback("BTCUSDT")
        assert state.stats()["rest_fallbacks"] == 1
        assert state.state_for("BTCUSDT").candle_source is DataSource.REST


class TestReporting:
    def test_stats_counts_each_freshness_band(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        state.set_subscribed({"A", "B", "C"})
        state.apply_candle("A", "1m", make_candle(clock))
        clock.advance(12)
        state.apply_candle("B", "1m", make_candle(clock))
        clock.advance(25)  # A: 37s (stale), B: 25s (lagging), C: never seen

        stats = state.stats()
        assert stats["stale"] == 2  # A and the never-seen C
        assert stats["lagging"] == 1
        assert stats["live"] == 0
        assert stats["subscribed"] == 3

    def test_symbol_report_puts_the_worst_first(
        self, state: MarketState, clock: VirtualClock
    ) -> None:
        state.set_subscribed({"OLD", "NEW"})
        state.apply_candle("OLD", "1m", make_candle(clock))
        clock.advance(20)
        state.apply_candle("NEW", "1m", make_candle(clock))

        rows = state.symbol_report()
        assert rows[0]["symbol"] == "OLD"
        assert rows[0]["freshness"] == "STALE" if rows[0]["age_sec"] > 30 else "LAGGING"
        assert rows[-1]["symbol"] == "NEW"
