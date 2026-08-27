"""Support/resistance strategy.

Thesis: price levels that have been tested repeatedly attract reactions, and a
rejection from such a level offers a well-defined entry — the stop goes just
beyond the level, which is both tight and logically placed.

The reason this strategy earns its place alongside mean reversion (which also
buys weakness) is the *stop*: mean reversion stops at an ATR distance from a
statistical band, while this one stops just past a price level that many
participants can see. When the level fails, the thesis is falsified immediately
and cheaply.

Levels are built by clustering confirmed swing pivots within an ATR-scaled
tolerance, requiring a minimum number of touches. A "level" with one touch is
just a previous price.
"""

from __future__ import annotations

from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class SupportResistanceStrategy(Strategy):
    name = "support_resistance"
    min_bars = 80

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        closes, highs, lows = series.closes, series.highs, series.lows
        atr_value = self.atr(series)
        if atr_value <= 0:
            return StrategyOpinion.wait("ATR_UNAVAILABLE")

        levels = ind.price_levels(
            highs,
            lows,
            atr_value,
            lookback=int(self.param("lookback", 60)),
            cluster_atr=float(self.param("cluster_atr", 0.4)),
            min_touches=int(self.param("min_touches", 2)),
        )
        if not levels:
            return StrategyOpinion.wait("NO_ESTABLISHED_LEVELS")

        price = closes[-1]
        entry_distance = float(self.param("entry_distance_atr", 0.5)) * atr_value

        # The nearest level we are currently interacting with.
        nearby = [
            (abs(price - level), level, touches)
            for level, touches in levels
            if abs(price - level) <= entry_distance
        ]
        if not nearby:
            return StrategyOpinion.wait("NOT_AT_A_LEVEL")
        _, level, touches = min(nearby)

        # Which side of the level are we on, and did price reject from it?
        last_bar = series.last
        if last_bar is None:
            return StrategyOpinion.wait("NO_BAR")

        approached_from_above = price >= level
        if approached_from_above:
            # Support: price came down to the level and bounced.
            direction = Direction.LONG
            wick = last_bar.low <= level <= last_bar.close
            rejected = wick and last_bar.close > last_bar.open
        else:
            direction = Direction.SHORT
            wick = last_bar.high >= level >= last_bar.close
            rejected = wick and last_bar.close < last_bar.open

        if not rejected:
            return StrategyOpinion.wait("NO_REJECTION_FROM_LEVEL")

        # Do not buy support inside a strong downtrend, or sell resistance inside
        # a strong uptrend: levels break in trends, and this stop is tight.
        adx_value = ind.last_valid(ind.adx(highs, lows, closes, 14)[0], default=0.0)
        if adx_value > 30 and view.regime_direction not in (Direction.WAIT, direction):
            return StrategyOpinion.wait("STRONG_TREND_AGAINST_LEVEL")

        # Stop just beyond the level: if it fails, the thesis is dead.
        buffer = atr_value * 0.3
        stop = level - buffer if direction is Direction.LONG else level + buffer

        # The next established level in the trade's direction is the natural
        # target: that is where the reaction is most likely to stall.
        margin = atr_value * 0.5
        if direction is Direction.LONG:
            forward = [lvl for lvl, _ in levels if lvl > price + margin]
            target = min(forward) if forward else None
        else:
            forward = [lvl for lvl, _ in levels if lvl < price - margin]
            target = max(forward) if forward else None

        wick_size = (
            (last_bar.close - last_bar.low)
            if direction is Direction.LONG
            else (last_bar.high - last_bar.close)
        )
        strong_rejection = wick_size > atr_value * 0.4

        confidence = self.scale_confidence(
            50.0 + min(touches / 5.0, 1.0) * 15.0,
            (strong_rejection, 10.0),
            (touches >= 3, 8.0),
            (adx_value < 20, 6.0),
            (target is not None, 5.0),
        )

        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=(
                f"{'SUPPORT' if direction is Direction.LONG else 'RESISTANCE'}_LEVEL",
                f"TOUCHES_{touches}",
                "REJECTION_CONFIRMED",
            ),
            stop_loss=stop,
            take_profit=target,
            reward_risk=self.base_rr,
            risk_score=45.0,
            metadata={
                "level": level,
                "touches": touches,
                "levels_found": len(levels),
                "adx": adx_value,
                "next_level": target,
            },
        )
