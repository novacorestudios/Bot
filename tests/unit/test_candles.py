"""Candle storage.

The critical property under test: a forming (unclosed) bar must never appear in
the arrays strategies read. If it does, every backtest silently disagrees with
live trading.
"""

from __future__ import annotations

import numpy as np
import pytest

from tradebot.core.types import Candle
from tradebot.market.candles import CandleSeries, CandleStore

from ..conftest import make_candles


def bar(open_time: int, close: float, closed: bool = True, volume: float = 100.0) -> Candle:
    return Candle(
        open_time=open_time,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=volume,
        close_time=open_time + 59_999,
        quote_volume=volume * close,
        closed=closed,
    )


class TestFormingBarIsolation:
    def test_forming_bar_is_excluded_from_arrays(self):
        series = CandleSeries("X", "1m", 100)
        series.append(bar(0, 100.0))
        series.append(bar(60_000, 101.0, closed=False))
        assert len(series) == 1
        assert series.closes.tolist() == [100.0]
        assert series.forming is not None

    def test_last_price_uses_the_forming_bar(self):
        """Current price should be live even though indicators must not be."""
        series = CandleSeries("X", "1m", 100)
        series.append(bar(0, 100.0))
        series.append(bar(60_000, 101.5, closed=False))
        assert series.last_price == 101.5
        assert series.last.close == 100.0

    def test_closing_the_forming_bar_appends_it_once(self):
        series = CandleSeries("X", "1m", 100)
        series.append(bar(0, 100.0))
        series.append(bar(60_000, 101.0, closed=False))
        series.append(bar(60_000, 101.0, closed=False))
        assert series.append(bar(60_000, 101.3, closed=True)) is True
        assert len(series) == 2
        assert series.forming is None
        assert series.closes.tolist() == [100.0, 101.3]

    def test_append_signals_only_on_close(self):
        series = CandleSeries("X", "1m", 100)
        assert series.append(bar(0, 100.0, closed=False)) is False
        assert series.append(bar(0, 100.0, closed=True)) is True


class TestOrderingAndCorrections:
    def test_out_of_order_bar_is_dropped(self):
        series = CandleSeries("X", "1m", 100)
        series.append(bar(60_000, 101.0))
        assert series.append(bar(0, 100.0)) is False
        assert len(series) == 1
        assert series.last.close == 101.0

    def test_duplicate_close_replaces_in_place(self):
        series = CandleSeries("X", "1m", 100)
        series.append(bar(0, 100.0))
        series.append(bar(0, 100.5))
        assert len(series) == 1
        assert series.last.close == 100.5

    def test_stale_forming_update_for_a_closed_bar_is_ignored(self):
        series = CandleSeries("X", "1m", 100)
        series.append(bar(60_000, 101.0, closed=True))
        series.append(bar(0, 99.0, closed=False))
        assert series.forming is None

    def test_max_bars_is_enforced(self):
        series = CandleSeries("X", "1m", max_bars=10)
        series.extend(make_candles([100.0 + i for i in range(50)]))
        assert len(series) == 10
        assert series.last.close == pytest.approx(149.0)


class TestGapsAndStaleness:
    def test_contiguous_series_has_no_gaps(self):
        series = CandleSeries("X", "1m", 100)
        series.extend(make_candles([100.0, 101.0, 102.0], interval_ms=60_000))
        assert series.gaps() == []

    def test_missing_bars_are_reported(self):
        series = CandleSeries("X", "1m", 100)
        series.append(bar(0, 100.0))
        series.append(bar(180_000, 103.0))  # two bars missing
        assert len(series.gaps()) == 1

    def test_stale_when_newest_bar_is_old(self):
        series = CandleSeries("X", "1m", 100)
        series.append(bar(0, 100.0))
        assert series.is_stale(now_ms=600_000)
        assert not series.is_stale(now_ms=100_000)

    def test_empty_series_counts_as_stale(self):
        assert CandleSeries("X", "1m", 100).is_stale(now_ms=0)


class TestArrays:
    def test_arrays_reflect_appends(self):
        series = CandleSeries("X", "1m", 100)
        series.extend(make_candles([100.0, 101.0]))
        first = series.closes.tolist()
        series.append(bar(series.last.open_time + 60_000, 102.0))
        assert series.closes.tolist() == [*first, 102.0]

    def test_arrays_are_float64_for_numpy_indicators(self):
        series = CandleSeries("X", "1m", 100)
        series.extend(make_candles([100.0, 101.0]))
        assert series.closes.dtype == np.float64

    def test_ready_requires_enough_closed_bars(self):
        series = CandleSeries("X", "1m", 100)
        series.extend(make_candles([100.0] * 5))
        assert series.ready(5)
        assert not series.ready(6)

    def test_session_resets_mark_day_boundaries(self):
        # Two bars either side of a UTC midnight.
        series = CandleSeries("X", "1h", 100)
        midnight = 1_700_000_000_000 // 86_400_000 * 86_400_000
        series.append(bar(midnight - 3_600_000, 100.0))
        series.append(bar(midnight, 101.0))
        resets = series.session_resets(hour_utc=0)
        assert bool(resets[0])
        assert bool(resets[1])


class TestStore:
    def test_series_are_isolated_per_symbol_and_timeframe(self):
        store = CandleStore(100)
        store.append("A", "1m", bar(0, 1.0))
        store.append("A", "5m", bar(0, 2.0))
        store.append("B", "1m", bar(0, 3.0))
        assert store.series("A", "1m").last.close == 1.0
        assert store.series("A", "5m").last.close == 2.0
        assert store.series("B", "1m").last.close == 3.0

    def test_price_falls_back_to_the_freshest_timeframe(self):
        store = CandleStore(100)
        store.append("A", "5m", bar(0, 10.0))
        store.append("A", "1m", bar(300_000, 11.0))
        assert store.price("A") == 11.0

    def test_price_of_an_unknown_symbol_is_zero_not_an_error(self):
        assert CandleStore(100).price("NOPE") == 0.0

    def test_retain_drops_symbols_that_left_the_ranking(self):
        store = CandleStore(100)
        for symbol in ("A", "B", "C"):
            store.append(symbol, "1m", bar(0, 1.0))
        assert store.retain({"A"}) == 2
        assert store.symbols() == {"A"}

    def test_retain_protects_symbols_with_open_positions(self):
        """A symbol that fell out of the top-N still has a live position to manage."""
        store = CandleStore(100)
        for symbol in ("A", "B", "C"):
            store.append(symbol, "1m", bar(0, 1.0))
        store.retain({"A"}, keep={"C"})
        assert store.symbols() == {"A", "C"}
