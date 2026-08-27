"""Volatility-expansion strategy.

Thesis: volatility is persistent. When realised range expands sharply relative to
its own recent norm, the expanded state tends to last several bars, and moves
initiated during it travel further.

This is not a breakout strategy. It does not require a level to be broken; it
requires the *character* of the market to have changed. The direction comes from
where price sits within the expanding range, which is why the
``min_directional_fraction`` filter matters — an expansion with price in the
middle of its range is two-sided volatility with no directional edge, and
trading it means taking the cost of both sides.

The stop is deliberately wider than other strategies' (higher ATR multiple),
because entering during an expansion with a normal stop is a request to be
stopped out by the very volatility that formed the thesis.
"""

from __future__ import annotations

import numpy as np

from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class VolatilityExpansionStrategy(Strategy):
    name = "volatility_expansion"
    min_bars = 80

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        closes, highs, lows = series.closes, series.highs, series.lows

        atr_period = int(self.param("atr_period", 14))
        atr_lookback = int(self.param("atr_lookback", 60))
        atr_series = ind.atr(highs, lows, closes, atr_period)
        finite = atr_series[np.isfinite(atr_series)]
        if finite.size < atr_lookback // 2:
            return StrategyOpinion.wait("INSUFFICIENT_ATR_HISTORY")

        current_atr = float(finite[-1])
        window = finite[-atr_lookback:-1] if finite.size > atr_lookback else finite[:-1]
        if window.size < 10:
            return StrategyOpinion.wait("INSUFFICIENT_ATR_HISTORY")
        baseline = float(np.median(window))
        if baseline <= 0:
            return StrategyOpinion.wait("ATR_BASELINE_ZERO")

        expansion = current_atr / baseline
        required = float(self.param("expansion_multiple", 1.6))
        if expansion < required:
            return StrategyOpinion.wait(f"NO_EXPANSION_{expansion:.2f}X_BELOW_{required}")

        # Direction: where does price sit inside the recent range?
        recent = min(10, len(series))
        window_high = float(np.max(highs[-recent:]))
        window_low = float(np.min(lows[-recent:]))
        span = window_high - window_low
        if span <= 0:
            return StrategyOpinion.wait("NO_RANGE")

        position = (closes[-1] - window_low) / span  # 0 = low, 1 = high
        min_fraction = float(self.param("min_directional_fraction", 0.6))

        if position >= min_fraction:
            direction = Direction.LONG
        elif position <= 1.0 - min_fraction:
            direction = Direction.SHORT
        else:
            # Two-sided volatility: expanding, but with no directional edge.
            return StrategyOpinion.wait(f"EXPANSION_WITHOUT_DIRECTION_POSITION_{position:.2f}")

        slope = ind.last_valid(ind.linear_slope(closes, 10), default=0.0)
        slope_agrees = (slope > 0) if direction is Direction.LONG else (slope < 0)
        if not slope_agrees:
            return StrategyOpinion.wait("SLOPE_CONTRADICTS_RANGE_POSITION")

        vol_ratio = ind.last_valid(ind.volume_ratio(series.volumes, 20), default=1.0)

        confidence = self.scale_confidence(
            50.0 + min((expansion - required) / required, 1.0) * 18.0,
            (vol_ratio > 1.3, 8.0),
            (abs(position - 0.5) > 0.35, 8.0),
            (view.regime.value in {"HIGH_VOLATILITY", "BREAKOUT"}, 8.0),
        )

        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=(
                f"ATR_EXPANSION_{expansion:.2f}X",
                f"RANGE_POSITION_{position:.2f}",
                f"VOLUME_{vol_ratio:.1f}X",
            ),
            reward_risk=self.base_rr,
            # Expanding volatility is genuinely riskier: the same stop distance
            # is crossed more easily.
            risk_score=55.0 + min((expansion - 1.0) * 20.0, 25.0),
            metadata={
                "expansion": expansion,
                "range_position": position,
                "atr": current_atr,
                "atr_baseline": baseline,
            },
        )
