"""VWAP strategy.

Thesis: session VWAP is where the day's volume actually traded, so it acts as a
magnet in balanced markets and as support/resistance in directional ones.

Two modes, selected by configuration:

* ``fade`` (default) — price stretched beyond a VWAP band in a non-trending
  market reverts toward VWAP. The target is VWAP itself.
* ``ride`` — price holding above/below VWAP in a trending market continues.

Fade mode carries the same danger as mean reversion and the same guard: an ADX
ceiling, above which it stands down. The bands are volume-weighted standard
deviations, not fixed percentages, so they adapt to the session's own character.
VWAP resets each UTC session; without the reset it slowly becomes a meaningless
long-run average.
"""

from __future__ import annotations

from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class VWAPStrategy(Strategy):
    name = "vwap"
    min_bars = 60

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        closes, highs, lows, volumes = (series.closes, series.highs, series.lows, series.volumes)
        if float(volumes.sum()) <= 0:
            return StrategyOpinion.wait("NO_VOLUME_DATA")

        reset_hour = int(self.param("session_reset_hour_utc", 0))
        resets = series.session_resets(reset_hour)
        band_std = float(self.param("band_std", 1.8))

        upper_band, centre_band, lower_band = ind.vwap_bands(
            highs, lows, closes, volumes, band_std, resets
        )
        vwap_value = ind.last_valid(centre_band, default=0.0)
        upper = ind.last_valid(upper_band, default=0.0)
        lower = ind.last_valid(lower_band, default=0.0)
        if vwap_value <= 0 or upper <= 0 or lower <= 0:
            return StrategyOpinion.wait("VWAP_UNAVAILABLE")

        price = closes[-1]
        mode = str(self.param("mode", "fade")).lower()
        adx_value = ind.last_valid(
            ind.adx(highs, lows, closes, int(self.param("adx_period", 14)))[0],
            default=0.0,
        )
        max_adx = float(self.param("max_adx", 24.0))
        atr_value = self.atr(series)
        if atr_value <= 0:
            return StrategyOpinion.wait("ATR_UNAVAILABLE")

        if mode == "fade":
            if adx_value > max_adx:
                return StrategyOpinion.wait(f"ADX_{adx_value:.1f}_TRENDING_NO_FADE")

            if price >= upper:
                direction = Direction.SHORT
            elif price <= lower:
                direction = Direction.LONG
            else:
                return StrategyOpinion.wait("INSIDE_VWAP_BANDS")

            distance_atr = abs(price - vwap_value) / atr_value
            confidence = self.scale_confidence(
                50.0 + min(distance_atr / 3.0, 1.0) * 18.0,
                (adx_value < max_adx * 0.6, 10.0),
                (view.regime.value in {"SIDEWAYS", "LOW_VOLATILITY"}, 8.0),
                (view.book_imbalance * direction.sign > 0.1, 4.0),
            )
            return StrategyOpinion(
                direction=direction,
                confidence=confidence,
                reasons=(
                    "VWAP_FADE",
                    f"BAND_{band_std}_SIGMA",
                    f"DISTANCE_{distance_atr:.2f}_ATR",
                    f"ADX_{adx_value:.1f}",
                ),
                take_profit=vwap_value,  # the magnet, not an arbitrary R
                reward_risk=self.base_rr,
                risk_score=52.0 + adx_value,
                metadata={
                    "vwap": vwap_value,
                    "distance_atr": distance_atr,
                    "adx": adx_value,
                    "mode": "fade",
                },
            )

        # ride mode: continuation from the VWAP side.
        if adx_value < max_adx:
            return StrategyOpinion.wait(f"ADX_{adx_value:.1f}_TOO_WEAK_TO_RIDE")

        if price > vwap_value and price < upper:
            direction = Direction.LONG
        elif price < vwap_value and price > lower:
            direction = Direction.SHORT
        else:
            return StrategyOpinion.wait("NOT_IN_RIDE_ZONE")

        confidence = self.scale_confidence(
            50.0 + min((adx_value - max_adx) / 20.0, 1.0) * 18.0,
            (view.regime_direction is direction, 10.0),
            (abs(price - vwap_value) / atr_value < 1.0, 8.0),
        )
        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=("VWAP_RIDE", f"ADX_{adx_value:.1f}"),
            reward_risk=self.base_rr,
            risk_score=45.0,
            metadata={"vwap": vwap_value, "adx": adx_value, "mode": "ride"},
        )
