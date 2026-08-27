"""Momentum strategy.

Thesis: a decisive, volume-backed move over a short window tends to continue for
a few more bars.

The part that matters is what it *refuses*. Naive momentum buys the top: by the
time a move is obvious it is often exhausted. So this requires:

* rate of change past a threshold, **and**
* EMA alignment confirming the direction is structural rather than a single bar,
* RSI inside a band — strong enough to confirm, but **not** so extended that we
  are buying from someone taking profit (the `rsi_long_max` ceiling),
* above-average volume, because a move on no volume is noise,
* higher-timeframe agreement when configured.
"""

from __future__ import annotations

from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class MomentumStrategy(Strategy):
    name = "momentum"
    min_bars = 60

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        closes, volumes = series.closes, series.volumes

        roc_period = int(self.param("roc_period", 10))
        threshold = float(self.param("roc_threshold", 0.0035))
        rate = ind.last_valid(ind.roc(closes, roc_period), default=0.0)

        if abs(rate) < threshold:
            return StrategyOpinion.wait(f"ROC_{rate:.4f}_BELOW_{threshold}")

        direction = Direction.LONG if rate > 0 else Direction.SHORT

        # EMA alignment: the move must have structure, not just one big bar.
        fast = ind.last_valid(ind.ema(closes, int(self.param("ema_fast", 9))))
        slow = ind.last_valid(ind.ema(closes, int(self.param("ema_slow", 21))))
        if fast <= 0 or slow <= 0:
            return StrategyOpinion.wait("EMA_UNAVAILABLE")
        aligned = (fast > slow) if direction is Direction.LONG else (fast < slow)
        if not aligned:
            return StrategyOpinion.wait("EMA_NOT_ALIGNED")

        # RSI band: confirmation without exhaustion.
        rsi_value = ind.last_valid(ind.rsi(closes, int(self.param("rsi_period", 14))), default=50.0)
        if direction is Direction.LONG:
            low = float(self.param("rsi_long_min", 52.0))
            high = float(self.param("rsi_long_max", 78.0))
            if rsi_value < low:
                return StrategyOpinion.wait(f"RSI_{rsi_value:.0f}_BELOW_{low:.0f}")
            if rsi_value > high:
                # Not a WAIT for lack of signal — a refusal to chase.
                return StrategyOpinion.wait(f"RSI_{rsi_value:.0f}_EXHAUSTED")
        else:
            low = float(self.param("rsi_short_min", 22.0))
            high = float(self.param("rsi_short_max", 48.0))
            if rsi_value > high:
                return StrategyOpinion.wait(f"RSI_{rsi_value:.0f}_ABOVE_{high:.0f}")
            if rsi_value < low:
                return StrategyOpinion.wait(f"RSI_{rsi_value:.0f}_EXHAUSTED")

        # Volume backing.
        required_volume = float(self.param("volume_multiple", 1.3))
        vol_ratio = ind.last_valid(ind.volume_ratio(volumes, 20), default=1.0)
        if vol_ratio < required_volume:
            return StrategyOpinion.wait(f"VOLUME_{vol_ratio:.2f}X_BELOW_{required_volume}")

        # Higher-timeframe agreement, when configured.
        htf_agrees = True
        confirm = self.confirm_series(view)
        if confirm is not None and confirm.ready(30):
            htf_slope = ind.last_valid(ind.linear_slope(confirm.closes, 20), default=0.0)
            htf_agrees = (htf_slope > 0) if direction is Direction.LONG else (htf_slope < 0)
            if not htf_agrees:
                return StrategyOpinion.wait("HIGHER_TIMEFRAME_DISAGREES")

        strength = min(abs(rate) / (threshold * 3), 1.0)
        confidence = self.scale_confidence(
            50.0 + strength * 25.0,
            (vol_ratio > required_volume * 1.5, 8.0),
            (abs(fast - slow) / max(slow, 1e-9) > 0.002, 6.0),
            (htf_agrees and confirm is not None, 8.0),
            (view.regime_direction is direction, 5.0),
        )

        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=(
                f"ROC_{rate * 100:.2f}PCT",
                f"RSI_{rsi_value:.0f}",
                f"VOLUME_{vol_ratio:.1f}X",
                "EMA_ALIGNED",
            ),
            reward_risk=self.base_rr,
            risk_score=40.0 + strength * 20.0,
            metadata={"roc": rate, "rsi": rsi_value, "volume_ratio": vol_ratio},
        )
