"""Strategy interface.

A strategy is a **pure function of market data**. It receives a read-only view
and returns a :class:`Signal`. It cannot place an order, size a position, read
the account, or reach the exchange — there is no reference to any of those in
this module, by design. That is what makes the guarantee "no strategy can
bypass the risk engine" structural rather than aspirational.

Stop and target derivation lives here, shared by all eight strategies, because:

* every stop must respect ATR, configured bounds *and* market structure — a stop
  placed just inside an obvious swing low is a stop that will be hit;
* a strategy that invents its own stop geometry produces R-multiples that are
  not comparable with any other strategy's, which breaks performance tracking
  and risk budgeting alike.

A strategy still chooses its ATR multiple and reward:risk through configuration;
it just does not choose the *method*.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tradebot.core.config import StopsConfig, TargetsConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp, safe_div
from tradebot.core.types import Direction, MarketRegime, Signal
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries, CandleStore

log = get_logger(__name__)


@dataclass(slots=True)
class MarketView:
    """Read-only market context handed to a strategy.

    Deliberately contains no account state, no position information and no
    gateway. A strategy physically cannot act on the account through this.
    """

    symbol: str
    candles: CandleStore
    regime: MarketRegime
    regime_confidence: float = 0.0
    regime_direction: Direction = Direction.WAIT
    book_imbalance: float = 0.0
    spread_bps: float = 0.0
    funding_rate: float = 0.0
    now_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: dict[str, Any] = field(default_factory=dict)

    def series(self, timeframe: str) -> CandleSeries | None:
        """Closed candles for a timeframe, or None when unavailable."""
        found = self.candles.get(self.symbol, timeframe)
        return found if found is not None and not found.is_empty else None

    def price(self) -> float:
        return self.candles.price(self.symbol)


class Strategy(ABC):
    """Base class for every strategy.

    Subclasses implement :meth:`evaluate`, which returns a direction, a
    confidence and reason codes. The base class turns that into a complete,
    validated :class:`Signal` with structure-aware levels.
    """

    #: Unique key. Must match the entry in config/strategies.yaml and the names
    #: used in regime.strategy_weights.
    name: str = "base"

    #: Bars required on the primary timeframe before this strategy may speak.
    min_bars: int = 60

    def __init__(self, params: dict[str, Any], stops: StopsConfig, targets: TargetsConfig) -> None:
        self.params = params
        self.stops = stops
        self.targets = targets
        self.enabled = bool(params.get("enabled", True))
        self.timeframe: str = params.get("timeframe", "5m")
        self.confirm_timeframe: str | None = params.get("confirm_timeframe")
        self.min_confidence: float = float(params.get("min_confidence", 55.0))
        self.atr_stop_multiple: float = float(params.get("atr_stop_multiple", stops.atr_multiple))
        self.base_rr: float = float(params.get("base_rr", targets.base_rr))

        # Performance counters, updated by the tracker. Used for the strategy
        # kill switch and risk allocation, never by the strategy itself.
        self.signals_emitted = 0
        self.errors = 0

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        """Return this strategy's opinion. Must not raise for ordinary data."""

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def generate(self, view: MarketView) -> Signal:
        """Evaluate and package a validated signal.

        Errors inside a strategy are contained: a broken strategy returns WAIT
        rather than taking down the engine or, worse, emitting a malformed
        signal that the risk engine has to catch.
        """
        if not self.enabled:
            return self._wait(view, ("STRATEGY_DISABLED",))

        series = view.series(self.timeframe)
        if series is None or not series.ready(self.min_bars):
            return self._wait(view, ("INSUFFICIENT_DATA",))

        try:
            opinion = self.evaluate(view, series)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            self.errors += 1
            log.warning(
                "strategy_evaluation_failed",
                strategy=self.name,
                symbol=view.symbol,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return self._wait(view, ("STRATEGY_ERROR",))

        if opinion.direction is Direction.WAIT:
            return self._wait(view, opinion.reasons)

        if opinion.confidence < self.min_confidence:
            return self._wait(
                view, (*opinion.reasons, f"CONFIDENCE_BELOW_{self.min_confidence:.0f}")
            )

        signal = self._build_signal(view, series, opinion)
        problems = signal.validate()
        if problems:
            # A malformed signal is a bug in the strategy. Refuse it here rather
            # than letting downstream code work around it.
            log.error(
                "strategy_produced_invalid_signal",
                strategy=self.name,
                symbol=view.symbol,
                problems=problems,
            )
            return self._wait(view, ("INVALID_SIGNAL_GEOMETRY",))

        self.signals_emitted += 1
        return signal

    # ------------------------------------------------------------------ #
    # Level derivation — shared so R-multiples are comparable across strategies
    # ------------------------------------------------------------------ #
    def _build_signal(
        self, view: MarketView, series: CandleSeries, opinion: StrategyOpinion
    ) -> Signal:
        entry = opinion.entry_price or series.last_price or series.closes[-1]
        atr_value = self.atr(series)

        stop = opinion.stop_loss or self.derive_stop(series, opinion.direction, entry, atr_value)
        target = opinion.take_profit or self.derive_target(
            opinion.direction, entry, stop, atr_value, opinion.reward_risk
        )

        volatility = safe_div(atr_value, entry, 0.0)

        return Signal(
            symbol=view.symbol,
            strategy=self.name,
            direction=opinion.direction,
            confidence=clamp(opinion.confidence, 0.0, 100.0),
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            timeframe=self.timeframe,
            signal_timestamp=view.now_ms,
            expected_duration_sec=opinion.expected_duration_sec or self.expected_duration(),
            volatility=volatility,
            risk_score=opinion.risk_score,
            reason_codes=opinion.reasons,
            metadata={**opinion.metadata, "atr": atr_value, "regime": view.regime.value},
        )

    def atr(self, series: CandleSeries) -> float:
        value = ind.last_valid(
            ind.atr(series.highs, series.lows, series.closes, self.stops.atr_period),
            default=0.0,
        )
        if value > 0:
            return value
        # Fall back to the recent average range rather than returning zero, which
        # would produce a zero-distance stop.
        recent = series.highs[-20:] - series.lows[-20:]
        return float(np.mean(recent)) if recent.size else 0.0

    def derive_stop(
        self, series: CandleSeries, direction: Direction, entry: float, atr_value: float
    ) -> float:
        """ATR-based stop, clamped to configured bounds and pushed beyond structure.

        The structure step is what stops this being a naive percentage stop: if a
        swing low sits between the entry and the ATR stop, the stop is moved just
        beyond that low, because price reaching the low will very likely take out
        anything above it.
        """
        cfg = self.stops
        distance = atr_value * self.atr_stop_multiple

        # Clamp to configured percentage bounds so a volatility spike cannot
        # produce an absurd stop in either direction.
        min_distance = entry * cfg.min_stop_pct
        max_distance = entry * cfg.max_stop_pct
        distance = clamp(distance, min_distance, max_distance)

        raw_stop = entry - distance if direction is Direction.LONG else entry + distance

        structural = self._structural_stop(series, direction, entry, atr_value)
        if structural is not None:
            if direction is Direction.LONG:
                raw_stop = min(raw_stop, structural)
            else:
                raw_stop = max(raw_stop, structural)
            # Re-clamp: structure must not blow past the maximum stop distance.
            widest = entry - max_distance if direction is Direction.LONG else entry + max_distance
            raw_stop = (
                max(raw_stop, widest) if direction is Direction.LONG else min(raw_stop, widest)
            )

        return raw_stop

    def _structural_stop(
        self, series: CandleSeries, direction: Direction, entry: float, atr_value: float
    ) -> float | None:
        """Just beyond the nearest relevant swing point, or None if there is none."""
        cfg = self.stops
        lookback = min(cfg.structure_lookback, len(series))
        if lookback < 5 or atr_value <= 0:
            return None
        buffer = atr_value * cfg.structure_buffer_atr

        if direction is Direction.LONG:
            recent_low = float(np.min(series.lows[-lookback:]))
            if recent_low >= entry:
                return None
            return recent_low - buffer

        recent_high = float(np.max(series.highs[-lookback:]))
        if recent_high <= entry:
            return None
        return recent_high + buffer

    def derive_target(
        self,
        direction: Direction,
        entry: float,
        stop: float,
        atr_value: float,
        reward_risk: float | None = None,
    ) -> float:
        """Target from reward:risk, capped by a plausible ATR-based distance.

        The cap matters: a wide stop multiplied by an ambitious R gives a target
        price the market will not reach inside a 60-minute window, which turns
        winners into time-based exits.
        """
        cfg = self.targets
        rr = clamp(reward_risk or self.base_rr, cfg.min_rr, cfg.max_rr)
        risk_distance = abs(entry - stop)
        if risk_distance <= 0:
            risk_distance = entry * self.stops.min_stop_pct

        distance = risk_distance * rr
        if atr_value > 0:
            distance = min(distance, atr_value * cfg.atr_multiple_cap)
        # But never below the minimum acceptable R, or the trade cannot pay.
        distance = max(distance, risk_distance * cfg.min_rr)

        return entry + distance if direction is Direction.LONG else entry - distance

    def expected_duration(self) -> int:
        """Rough expected holding time in seconds, used by the funding estimate."""
        from tradebot.core.types import Timeframe

        try:
            bar_sec = Timeframe(self.timeframe).seconds
        except ValueError:
            bar_sec = 300
        return bar_sec * 8

    # ------------------------------------------------------------------ #
    def _wait(self, view: MarketView, reasons: tuple[str, ...]) -> Signal:
        return Signal(
            symbol=view.symbol,
            strategy=self.name,
            direction=Direction.WAIT,
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            timeframe=self.timeframe,
            signal_timestamp=view.now_ms,
            reason_codes=reasons,
        )

    # -- helpers available to subclasses ---------------------------------- #
    def confirm_series(self, view: MarketView) -> CandleSeries | None:
        """The higher-timeframe series, when the strategy configures one."""
        if not self.confirm_timeframe:
            return None
        return view.series(self.confirm_timeframe)

    @staticmethod
    def scale_confidence(base: float, *bonuses: tuple[bool, float]) -> float:
        """Accumulate confidence from a set of confirmations, clamped to 100."""
        total = base
        for condition, amount in bonuses:
            if condition:
                total += amount
        return clamp(total, 0.0, 100.0)

    def param(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)


@dataclass(slots=True)
class StrategyOpinion:
    """What a strategy concluded, before levels and validation are applied."""

    direction: Direction
    confidence: float = 0.0
    reasons: tuple[str, ...] = ()
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    reward_risk: float | None = None
    expected_duration_sec: int = 0
    risk_score: float = 50.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def wait(cls, *reasons: str) -> StrategyOpinion:
        return cls(direction=Direction.WAIT, confidence=0.0, reasons=reasons)

    @classmethod
    def long(cls, confidence: float, *reasons: str, **kwargs: Any) -> StrategyOpinion:
        return cls(direction=Direction.LONG, confidence=confidence, reasons=reasons, **kwargs)

    @classmethod
    def short(cls, confidence: float, *reasons: str, **kwargs: Any) -> StrategyOpinion:
        return cls(direction=Direction.SHORT, confidence=confidence, reasons=reasons, **kwargs)
