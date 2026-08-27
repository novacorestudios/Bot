"""Mean-reversion strategy.

Thesis: in a range-bound market, price stretched far from its mean tends to
snap back.

This is the most dangerous strategy in the set, because a strong trend looks
exactly like a permanently stretched market — and fading a trend is unbounded
loss. Every guard here exists for that reason:

* **ADX ceiling.** Above it, the market is trending and this strategy stands
  down entirely.
* **Higher-timeframe check.** Fading a move that the higher timeframe agrees
  with is a bet against both.
* **Stretch in ATR, not percent.** A 2 % move is enormous for one symbol and
  routine for another.
* **Tighter stop, lower reward:risk.** Reversion targets the mean, which is
  nearer than a trend target; pretending otherwise inflates R and produces
  targets price never reaches.
"""

from __future__ import annotations

from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    min_bars = 60

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        closes, highs, lows = series.closes, series.highs, series.lows

        # Guard 1: never fade a trending market.
        adx_period = int(self.param("adx_period", 14))
        adx_value = ind.last_valid(ind.adx(highs, lows, closes, adx_period)[0], default=0.0)
        max_adx = float(self.param("max_adx", 22.0))
        if adx_value > max_adx:
            return StrategyOpinion.wait(f"ADX_{adx_value:.1f}_TRENDING_WILL_NOT_FADE")

        bb_period = int(self.param("bb_period", 20))
        bb_std = float(self.param("bb_std", 2.2))
        upper_band, middle_band, lower_band = ind.bollinger(closes, bb_period, bb_std)
        upper = ind.last_valid(upper_band, default=0.0)
        middle = ind.last_valid(middle_band, default=0.0)
        lower = ind.last_valid(lower_band, default=0.0)
        if middle <= 0 or upper <= 0 or lower <= 0:
            return StrategyOpinion.wait("BANDS_UNAVAILABLE")

        price = closes[-1]
        rsi_value = ind.last_valid(ind.rsi(closes, int(self.param("rsi_period", 14))), default=50.0)
        oversold = float(self.param("rsi_oversold", 26.0))
        overbought = float(self.param("rsi_overbought", 74.0))

        if price <= lower and rsi_value <= oversold:
            direction = Direction.LONG
        elif price >= upper and rsi_value >= overbought:
            direction = Direction.SHORT
        else:
            return StrategyOpinion.wait("NOT_STRETCHED")

        # Guard 2: measure the stretch in ATR so it is comparable across symbols.
        atr_value = self.atr(series)
        if atr_value <= 0:
            return StrategyOpinion.wait("ATR_UNAVAILABLE")
        stretch_atr = abs(price - middle) / atr_value
        min_stretch = float(self.param("min_stretch_atr", 1.5))
        if stretch_atr < min_stretch:
            return StrategyOpinion.wait(f"STRETCH_{stretch_atr:.2f}_ATR_BELOW_{min_stretch}")

        # Guard 3: do not fade a move the higher timeframe endorses.
        confirm = self.confirm_series(view)
        if confirm is not None and confirm.ready(40):
            htf_adx = ind.last_valid(
                ind.adx(confirm.highs, confirm.lows, confirm.closes, 14)[0], default=0.0
            )
            htf_slope = ind.last_valid(ind.linear_slope(confirm.closes, 20), default=0.0)
            if htf_adx > 25:
                against_htf = (direction is Direction.LONG and htf_slope < 0) or (
                    direction is Direction.SHORT and htf_slope > 0
                )
                if against_htf:
                    return StrategyOpinion.wait("HIGHER_TIMEFRAME_TRENDS_AGAINST_FADE")

        # The target is the mean, not an arbitrary R multiple.
        target = middle
        extremity = (
            min(oversold - rsi_value, 15.0)
            if direction is Direction.LONG
            else min(rsi_value - overbought, 15.0)
        )

        confidence = self.scale_confidence(
            50.0 + min(stretch_atr / (min_stretch * 2), 1.0) * 18.0,
            (extremity > 5, 8.0),
            (adx_value < max_adx * 0.6, 8.0),
            (view.regime.value in {"SIDEWAYS", "LOW_VOLATILITY"}, 8.0),
        )

        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=(
                f"RSI_{rsi_value:.0f}",
                f"STRETCH_{stretch_atr:.2f}_ATR",
                f"ADX_{adx_value:.1f}_RANGING",
                "TARGET_IS_MEAN",
            ),
            take_profit=target,
            reward_risk=self.base_rr,
            # Reversion is inherently riskier than continuation: it is a bet
            # against the most recent evidence.
            risk_score=55.0 + max(0.0, adx_value - 10.0),
            metadata={
                "rsi": rsi_value,
                "stretch_atr": stretch_atr,
                "adx": adx_value,
                "mean": middle,
            },
        )
