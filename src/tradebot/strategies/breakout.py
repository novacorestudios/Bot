"""Breakout strategy.

Thesis: a range that has been compressed, then breaks with volume, tends to
continue in the breakout direction.

Three refusals define this strategy:

* **No prior compression, no trade.** A "breakout" from an already-wide range is
  just continuation, and the stop has nowhere sensible to sit.
* **No volume, no trade.** Breaks on thin volume are the classic false break.
* **Already extended, no trade.** If price has run well past the level before we
  see it, the risk:reward has already been consumed by the move — chasing it
  means a wide stop for a target that has mostly happened.

Donchian levels exclude the current bar (see ``indicators.donchian``); using the
current bar would make every bar a breakout of itself.
"""

from __future__ import annotations

import numpy as np

from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class BreakoutStrategy(Strategy):
    name = "breakout"
    min_bars = 60

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        closes, highs, lows, volumes = (series.closes, series.highs, series.lows, series.volumes)
        lookback = int(self.param("lookback", 20))

        upper_band, lower_band = ind.donchian(highs, lows, lookback)
        upper = ind.last_valid(upper_band, default=float("inf"))
        lower = ind.last_valid(lower_band, default=0.0)
        if not np.isfinite(upper) or lower <= 0:
            return StrategyOpinion.wait("RANGE_UNAVAILABLE")

        atr_value = self.atr(series)
        if atr_value <= 0:
            return StrategyOpinion.wait("ATR_UNAVAILABLE")

        buffer = atr_value * float(self.param("breakout_atr_buffer", 0.15))
        price = closes[-1]

        if price > upper + buffer:
            direction, level = Direction.LONG, upper
        elif price < lower - buffer:
            direction, level = Direction.SHORT, lower
        else:
            return StrategyOpinion.wait("NO_BREAKOUT")

        # Prior compression is mandatory.
        bandwidth = ind.bollinger_bandwidth(closes, 20, 2.0)
        prior = bandwidth[:-1]
        prior = prior[np.isfinite(prior)]
        if prior.size < lookback:
            return StrategyOpinion.wait("INSUFFICIENT_BANDWIDTH_HISTORY")
        prior_min = float(np.min(prior[-lookback:]))
        max_prior = float(self.param("max_prior_bandwidth", 0.030))
        if prior_min > max_prior:
            return StrategyOpinion.wait(f"NO_PRIOR_COMPRESSION_{prior_min:.4f}_ABOVE_{max_prior}")

        # Volume confirmation.
        required = float(self.param("volume_multiple", 1.6))
        vol_ratio = ind.last_valid(ind.volume_ratio(volumes, 20), default=1.0)
        if vol_ratio < required:
            return StrategyOpinion.wait(f"VOLUME_{vol_ratio:.2f}X_BELOW_{required}")

        # Do not chase a break that has already run.
        extension_atr = abs(price - level) / atr_value
        max_extension = float(self.param("max_extension_atr", 1.2))
        if extension_atr > max_extension:
            return StrategyOpinion.wait(f"ALREADY_EXTENDED_{extension_atr:.1f}_ATR_PAST_LEVEL")

        # The breakout bar should be decisive, not a long wick.
        last_bar = series.last
        body_fraction = last_bar.body_fraction if last_bar else 0.0
        bar_agrees = (
            (last_bar.is_bullish if direction is Direction.LONG else not last_bar.is_bullish)
            if last_bar
            else False
        )

        confidence = self.scale_confidence(
            55.0 + min(vol_ratio / (required * 2), 1.0) * 15.0,
            (body_fraction > 0.5 and bar_agrees, 10.0),
            (extension_atr < max_extension * 0.5, 8.0),
            (prior_min < max_prior * 0.6, 8.0),
            (view.book_imbalance * direction.sign > 0.1, 4.0),
        )

        # The stop belongs just inside the broken level: if price returns there,
        # the breakout has failed by definition.
        stop = level - atr_value * 0.3 if direction is Direction.LONG else level + atr_value * 0.3

        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=(
                f"BREAK_{'UP' if direction is Direction.LONG else 'DOWN'}",
                f"PRIOR_SQUEEZE_{prior_min:.4f}",
                f"VOLUME_{vol_ratio:.1f}X",
                f"EXTENSION_{extension_atr:.2f}_ATR",
            ),
            stop_loss=stop,
            reward_risk=self.base_rr,
            risk_score=45.0 + extension_atr * 15.0,
            metadata={
                "level": level,
                "extension_atr": extension_atr,
                "prior_bandwidth": prior_min,
                "volume_ratio": vol_ratio,
            },
        )
