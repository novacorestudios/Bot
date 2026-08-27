"""Rolling OHLCV storage.

The one rule that matters here: **strategies see closed bars only.**

A live kline stream sends repeated updates for the bar that is still forming.
If a strategy reads that forming bar it is effectively reading a partial future,
and every backtest built on closed bars will disagree with live behaviour. So
``CandleSeries`` keeps the forming bar separate from the closed history, and the
array accessors that strategies use return closed bars exclusively.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np

from tradebot.core.types import Candle, Timeframe


class CandleSeries:
    """A bounded, append-only series of closed candles for one symbol/timeframe."""

    __slots__ = (
        "symbol",
        "timeframe",
        "max_bars",
        "_candles",
        "_forming",
        "_cache",
        "_cache_version",
        "_version",
    )

    def __init__(self, symbol: str, timeframe: str, max_bars: int = 500) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_bars = max_bars
        self._candles: deque[Candle] = deque(maxlen=max_bars)
        self._forming: Candle | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_version = -1
        self._version = 0

    # -- population -------------------------------------------------------- #
    def append(self, candle: Candle) -> bool:
        """Add or update a candle. Returns True when a bar has just CLOSED.

        Handles the three cases a live stream produces: a new forming bar, an
        update to the current forming bar, and the final update that closes it.
        Out-of-order and duplicate bars are ignored rather than corrupting the
        series.
        """
        if not candle.closed:
            if self._candles and candle.open_time <= self._candles[-1].open_time:
                return False  # stale update for an already-closed bar
            self._forming = candle
            return False

        if self._candles:
            last = self._candles[-1]
            if candle.open_time < last.open_time:
                return False  # out of order — drop it
            if candle.open_time == last.open_time:
                # A correction to the most recent closed bar. Replace in place.
                self._candles[-1] = candle
                self._invalidate()
                return False

        self._candles.append(candle)
        if self._forming is not None and self._forming.open_time <= candle.open_time:
            self._forming = None
        self._invalidate()
        return True

    def extend(self, candles: Iterable[Candle]) -> None:
        """Bulk-load history (from REST). Assumes ascending order."""
        for candle in candles:
            self.append(candle)

    def replace_all(self, candles: Iterable[Candle]) -> None:
        """Discard everything and reload — used after a gap is detected."""
        self._candles.clear()
        self._forming = None
        for candle in candles:
            if candle.closed:
                self._candles.append(candle)
        self._invalidate()

    def _invalidate(self) -> None:
        self._version += 1

    # -- access ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._candles)

    def __iter__(self) -> Iterator[Candle]:
        return iter(self._candles)

    def __getitem__(self, index: int) -> Candle:
        return list(self._candles)[index]

    @property
    def is_empty(self) -> bool:
        return not self._candles

    @property
    def last(self) -> Candle | None:
        """Most recent CLOSED candle."""
        return self._candles[-1] if self._candles else None

    @property
    def forming(self) -> Candle | None:
        """The bar currently being built. Strategies must not use this."""
        return self._forming

    @property
    def last_price(self) -> float:
        """Best available current price: the forming bar's close, else the last close."""
        if self._forming is not None:
            return self._forming.close
        return self._candles[-1].close if self._candles else 0.0

    @property
    def last_close_time(self) -> int:
        return self._candles[-1].close_time if self._candles else 0

    def ready(self, minimum: int) -> bool:
        """True when enough closed bars exist for an indicator of this length."""
        return len(self._candles) >= minimum

    def is_stale(self, now_ms: int, tolerance_multiple: float = 2.5) -> bool:
        """True when the newest bar is older than the timeframe allows.

        Used by the health monitor: a symbol whose stream silently died looks
        exactly like a symbol that simply isn't moving, unless we check the clock.
        """
        if not self._candles:
            return True
        try:
            interval_ms = Timeframe(self.timeframe).milliseconds
        except ValueError:
            interval_ms = 60_000
        age = now_ms - self._candles[-1].close_time
        return age > interval_ms * tolerance_multiple

    def gaps(self) -> list[tuple[int, int]]:
        """Missing-bar ranges as (after_open_time, before_open_time) pairs.

        A gap means the stream dropped bars; indicators computed across it are
        wrong, so the feed reloads history when this is non-empty.
        """
        if len(self._candles) < 2:
            return []
        try:
            interval = Timeframe(self.timeframe).milliseconds
        except ValueError:
            return []
        out: list[tuple[int, int]] = []
        candles = list(self._candles)
        for prev, curr in zip(candles, candles[1:], strict=False):
            if curr.open_time - prev.open_time > interval:
                out.append((prev.open_time, curr.open_time))
        return out

    # -- numpy views (cached per version) ----------------------------------- #
    def _arrays(self) -> dict[str, np.ndarray]:
        if self._cache_version != self._version:
            candles = list(self._candles)
            self._cache = {
                "open": np.fromiter(
                    (c.open for c in candles), dtype=np.float64, count=len(candles)
                ),
                "high": np.fromiter(
                    (c.high for c in candles), dtype=np.float64, count=len(candles)
                ),
                "low": np.fromiter((c.low for c in candles), dtype=np.float64, count=len(candles)),
                "close": np.fromiter(
                    (c.close for c in candles), dtype=np.float64, count=len(candles)
                ),
                "volume": np.fromiter(
                    (c.volume for c in candles), dtype=np.float64, count=len(candles)
                ),
                "quote_volume": np.fromiter(
                    (c.quote_volume for c in candles), dtype=np.float64, count=len(candles)
                ),
                "open_time": np.fromiter(
                    (c.open_time for c in candles), dtype=np.int64, count=len(candles)
                ),
                "taker_buy_volume": np.fromiter(
                    (c.taker_buy_volume for c in candles), dtype=np.float64, count=len(candles)
                ),
            }
            self._cache_version = self._version
        return self._cache

    @property
    def opens(self) -> np.ndarray:
        return self._arrays()["open"]

    @property
    def highs(self) -> np.ndarray:
        return self._arrays()["high"]

    @property
    def lows(self) -> np.ndarray:
        return self._arrays()["low"]

    @property
    def closes(self) -> np.ndarray:
        return self._arrays()["close"]

    @property
    def volumes(self) -> np.ndarray:
        return self._arrays()["volume"]

    @property
    def quote_volumes(self) -> np.ndarray:
        return self._arrays()["quote_volume"]

    @property
    def open_times(self) -> np.ndarray:
        return self._arrays()["open_time"]

    @property
    def taker_buy_volumes(self) -> np.ndarray:
        return self._arrays()["taker_buy_volume"]

    def session_resets(self, hour_utc: int = 0) -> np.ndarray:
        """Boolean mask marking the first bar of each UTC session — for VWAP."""
        times = self.open_times
        if times.size == 0:
            return np.zeros(0, dtype=bool)
        seconds_into_day = (times // 1000) % 86_400
        session_start = hour_utc * 3600
        day_index = (times // 1000 - session_start) // 86_400
        resets = np.zeros(times.size, dtype=bool)
        if times.size:
            resets[0] = True
            resets[1:] = day_index[1:] != day_index[:-1]
        _ = seconds_into_day
        return resets


class CandleStore:
    """All series for all symbols and timeframes."""

    def __init__(self, max_bars: int = 500) -> None:
        self.max_bars = max_bars
        self._series: dict[tuple[str, str], CandleSeries] = {}

    def series(self, symbol: str, timeframe: str) -> CandleSeries:
        """Get or create the series for a symbol/timeframe."""
        key = (symbol, timeframe)
        found = self._series.get(key)
        if found is None:
            found = CandleSeries(symbol, timeframe, self.max_bars)
            self._series[key] = found
        return found

    def get(self, symbol: str, timeframe: str) -> CandleSeries | None:
        return self._series.get((symbol, timeframe))

    def append(self, symbol: str, timeframe: str, candle: Candle) -> bool:
        return self.series(symbol, timeframe).append(candle)

    def load(self, symbol: str, timeframe: str, candles: Iterable[Candle]) -> None:
        self.series(symbol, timeframe).replace_all(candles)

    def has(self, symbol: str, timeframe: str, minimum: int = 1) -> bool:
        found = self._series.get((symbol, timeframe))
        return found is not None and found.ready(minimum)

    def price(self, symbol: str, timeframe: str | None = None) -> float:
        """Latest known price for a symbol, from any timeframe if unspecified."""
        if timeframe is not None:
            found = self._series.get((symbol, timeframe))
            return found.last_price if found else 0.0
        best_time, best_price = -1, 0.0
        for (sym, _tf), series in self._series.items():
            if sym != symbol or series.is_empty:
                continue
            if series.last_close_time > best_time:
                best_time, best_price = series.last_close_time, series.last_price
        return best_price

    def symbols(self) -> set[str]:
        return {symbol for symbol, _ in self._series}

    def drop_symbol(self, symbol: str) -> None:
        """Release memory for a symbol that left the candidate set."""
        for key in [k for k in self._series if k[0] == symbol]:
            del self._series[key]

    def retain(self, symbols: set[str], keep: set[str] | None = None) -> int:
        """Drop every symbol not in `symbols` or `keep`. Returns the count dropped.

        `keep` protects symbols with open positions, which must never lose their
        data just because they fell out of the top-N ranking.
        """
        protected = symbols | (keep or set())
        doomed = {sym for sym, _ in self._series} - protected
        for symbol in doomed:
            self.drop_symbol(symbol)
        return len(doomed)

    def stats(self) -> dict[str, int]:
        return {
            "series": len(self._series),
            "symbols": len(self.symbols()),
            "candles": sum(len(s) for s in self._series.values()),
        }
