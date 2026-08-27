"""Trend-following strategy.

Thesis: in a confirmed trend, a pullback toward the fast moving average offers a
better entry than the extreme, because the stop can sit behind structure instead
of miles away.

Entering on a pullback rather than a breakout is what distinguishes this from
the momentum and breakout strategies, and it is why the three can coexist
without simply duplicating each other's trades: they enter the same trend at
different moments, and the aggregator's consensus is meaningful precisely
because they disagree about timing.
"""

from __future__ import annotations

from tradebot.core.mathutil import safe_div
from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    min_bars = 80

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        closes, highs, lows = series.closes, series.highs, series.lows

        adx_period = int(self.param("adx_period", 14))
        adx_series, plus_di, minus_di = ind.adx(highs, lows, closes, adx_period)
        adx_value = ind.last_valid(adx_series, default=0.0)
        adx_min = float(self.param("adx_min", 22.0))
        if adx_value < adx_min:
            return StrategyOpinion.wait(f"ADX_{adx_value:.1f}_BELOW_{adx_min}")

        fast_period = int(self.param("ema_fast", 20))
        slow_period = int(self.param("ema_slow", 50))
        fast = ind.last_valid(ind.ema(closes, fast_period))
        slow = ind.last_valid(ind.ema(closes, slow_period))
        if fast <= 0 or slow <= 0:
            return StrategyOpinion.wait("EMA_UNAVAILABLE")

        pdi = ind.last_valid(plus_di, default=0.0)
        mdi = ind.last_valid(minus_di, default=0.0)

        if fast > slow and pdi > mdi:
            direction = Direction.LONG
        elif fast < slow and mdi > pdi:
            direction = Direction.SHORT
        else:
            return StrategyOpinion.wait("TREND_DIRECTION_UNCLEAR")

        # Higher-timeframe agreement: trading a 5m trend against the 1h trend is
        # how a pullback entry becomes a reversal entry.
        confirm = self.confirm_series(view)
        htf_confirms = False
        if confirm is not None and confirm.ready(60):
            htf_fast = ind.last_valid(ind.ema(confirm.closes, 20))
            htf_slow = ind.last_valid(ind.ema(confirm.closes, 50))
            if htf_fast > 0 and htf_slow > 0:
                htf_up = htf_fast > htf_slow
                if (direction is Direction.LONG) != htf_up:
                    return StrategyOpinion.wait("HIGHER_TIMEFRAME_TREND_OPPOSES")
                htf_confirms = True

        # Pullback: price should be near the fast EMA, not extended from it.
        atr_value = self.atr(series)
        if atr_value <= 0:
            return StrategyOpinion.wait("ATR_UNAVAILABLE")
        distance_atr = abs(closes[-1] - fast) / atr_value
        max_pullback = float(self.param("pullback_atr", 0.8))

        if distance_atr > max_pullback * 3:
            return StrategyOpinion.wait(f"EXTENDED_{distance_atr:.1f}_ATR_FROM_EMA")

        # The pullback must not have broken the trend: price still on the right
        # side of the slow EMA.
        if direction is Direction.LONG and closes[-1] < slow:
            return StrategyOpinion.wait("PRICE_BELOW_SLOW_EMA")
        if direction is Direction.SHORT and closes[-1] > slow:
            return StrategyOpinion.wait("PRICE_ABOVE_SLOW_EMA")

        in_pullback = distance_atr <= max_pullback
        separation = safe_div(abs(fast - slow), slow, 0.0)

        confidence = self.scale_confidence(
            50.0 + min((adx_value - adx_min) / 20.0, 1.0) * 20.0,
            (in_pullback, 12.0),
            (htf_confirms, 10.0),
            (separation > 0.004, 6.0),
            (view.regime_direction is direction, 5.0),
        )

        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=(
                f"ADX_{adx_value:.1f}",
                "EMA_STACK_UP" if direction is Direction.LONG else "EMA_STACK_DOWN",
                "PULLBACK_ENTRY" if in_pullback else "TREND_CONTINUATION",
            ),
            reward_risk=self.base_rr,
            risk_score=35.0 + distance_atr * 10.0,
            metadata={"adx": adx_value, "distance_atr": distance_atr, "ema_separation": separation},
        )
