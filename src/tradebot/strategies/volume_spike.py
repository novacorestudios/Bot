"""Volume-spike strategy.

Thesis: an abrupt volume surge marks the arrival of an informed or forced
participant, and price tends to continue briefly in the direction of the surge.

The two filters that make this workable rather than random:

* **Body fraction.** A spike bar that is mostly wick means the move was rejected
  — that is a reversal signal, not a continuation one, so it is refused rather
  than traded backwards.
* **Confirmation bar.** Requiring the following bar to hold the move filters out
  the single-print liquidations that spike volume and instantly revert. It costs
  one bar of entry price, which is the price of not being the exit liquidity.
"""

from __future__ import annotations

from tradebot.core.types import Direction
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.strategies.base import MarketView, Strategy, StrategyOpinion


class VolumeSpikeStrategy(Strategy):
    name = "volume_spike"
    min_bars = 50

    def evaluate(self, view: MarketView, series: CandleSeries) -> StrategyOpinion:
        lookback = int(self.param("volume_lookback", 30))
        required = float(self.param("volume_multiple", 3.0))
        confirm_bars = int(self.param("confirm_bars", 1))

        ratios = ind.volume_ratio(series.volumes, lookback)
        if ratios.size <= confirm_bars:
            return StrategyOpinion.wait("INSUFFICIENT_DATA")

        # The spike bar sits `confirm_bars` back; the bars after it confirm.
        spike_index = len(series) - 1 - confirm_bars
        if spike_index < 0:
            return StrategyOpinion.wait("INSUFFICIENT_DATA")

        spike_ratio = float(ratios[spike_index])
        if not spike_ratio or spike_ratio < required:
            return StrategyOpinion.wait(f"NO_SPIKE_{spike_ratio:.2f}X_BELOW_{required}")

        spike_bar = series[spike_index]
        min_body = float(self.param("min_body_fraction", 0.55))
        if spike_bar.body_fraction < min_body:
            # Mostly wick: the move was rejected. Refuse rather than fade — that
            # is a different thesis with a different stop.
            return StrategyOpinion.wait(f"SPIKE_BAR_REJECTED_BODY_{spike_bar.body_fraction:.2f}")

        direction = Direction.LONG if spike_bar.is_bullish else Direction.SHORT

        # Confirmation: the bars since the spike must not have given it all back.
        confirm_close = series.closes[-1]
        if direction is Direction.LONG and confirm_close < spike_bar.open:
            return StrategyOpinion.wait("SPIKE_FADED")
        if direction is Direction.SHORT and confirm_close > spike_bar.open:
            return StrategyOpinion.wait("SPIKE_FADED")

        held = (
            (confirm_close >= spike_bar.close)
            if direction is Direction.LONG
            else (confirm_close <= spike_bar.close)
        )

        # Taker-flow agreement, when the data is present.
        taker_agrees = False
        taker = series.taker_buy_volumes
        if taker.size > spike_index and series.volumes[spike_index] > 0:
            buy_fraction = taker[spike_index] / series.volumes[spike_index]
            taker_agrees = (
                (buy_fraction > 0.6) if direction is Direction.LONG else (buy_fraction < 0.4)
            )

        confidence = self.scale_confidence(
            50.0 + min(spike_ratio / (required * 2), 1.0) * 18.0,
            (held, 10.0),
            (spike_bar.body_fraction > 0.75, 8.0),
            (taker_agrees, 8.0),
            (view.book_imbalance * direction.sign > 0.15, 4.0),
        )

        return StrategyOpinion(
            direction=direction,
            confidence=confidence,
            reasons=(
                f"VOLUME_SPIKE_{spike_ratio:.1f}X",
                f"BODY_{spike_bar.body_fraction:.2f}",
                "CONFIRMED_HOLD" if held else "CONFIRMED_NO_FADE",
            ),
            reward_risk=self.base_rr,
            risk_score=50.0 + min(spike_ratio * 3, 25.0),
            metadata={"spike_ratio": spike_ratio, "body_fraction": spike_bar.body_fraction},
        )
