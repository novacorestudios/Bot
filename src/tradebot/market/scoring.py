"""Market scoring.

Turns raw market data into a comparable 0-100 score per symbol so the scanner
can rank a whole universe. Every component is itself 0-100 and is combined by
configurable weights, then reduced by cost and risk penalties.

Two principles:

* **No symbol is privileged.** There is no list of "good" coins anywhere in this
  file. BTC ranks where its current data puts it, which on a quiet day is
  nowhere near the top.
* **Volatility is scored as a band, not a maximum.** For a sub-hour strategy,
  too little movement means nothing to capture and too much means stops are hit
  by noise. The preferred band is configuration, and it is one of the first
  things worth re-fitting from real results.

A high market score means "this market is *worth watching*", not "enter now".
Entry requires a strategy signal, a consensus, an opportunity score and a
positive expected net edge — four further gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tradebot.core.config import ScannerConfig
from tradebot.core.mathutil import band_score, clamp, normalise_score, safe_div
from tradebot.core.types import MarketScore
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries
from tradebot.market.microstructure import CostModel, LiquiditySnapshot


@dataclass(slots=True)
class ScoringInputs:
    """Everything needed to score one symbol."""

    symbol: str
    series: CandleSeries  # primary timeframe, closed bars only
    liquidity: LiquiditySnapshot
    funding_rate: float = 0.0
    quote_volume_24h: float = 0.0
    price_change_24h: float = 0.0
    correlation_penalty: float = 0.0  # 0..1, set by the correlation engine
    timestamp: int = 0


class MarketScorer:
    """Computes the composite market score for a symbol."""

    #: A component that cannot be computed scores neutral rather than zero, so a
    #: missing order book does not silently push a symbol to the bottom.
    NEUTRAL = 50.0

    def __init__(self, config: ScannerConfig, cost_model: CostModel) -> None:
        self.config = config
        self.cost_model = cost_model
        self._weights = config.weights.normalised()

    # ------------------------------------------------------------------ #
    # Components — each returns 0..100
    # ------------------------------------------------------------------ #
    def liquidity_score(self, inputs: ScoringInputs) -> float:
        """Depth near the touch, on a log scale.

        Log scale because the difference between 10k and 100k of resting depth
        matters enormously for a small account, while 10M vs 100M does not.
        """
        depth = inputs.liquidity.depth_notional
        if depth <= 0:
            return 0.0
        # 1k -> 0, 1M -> 100
        return normalise_score(float(np.log10(max(depth, 1.0))), 3.0, 6.0)

    def volume_score(self, inputs: ScoringInputs) -> float:
        """24h quote volume, log scaled. 10M -> 0, 10B -> 100."""
        volume = inputs.quote_volume_24h
        if volume <= 0:
            return 0.0
        return normalise_score(float(np.log10(max(volume, 1.0))), 7.0, 10.0)

    def recent_volume_score(self, inputs: ScoringInputs) -> float:
        """Is this market active *now*, versus its own recent norm?

        Distinct from 24h volume: a symbol can have huge daily volume and be
        completely dead in the current hour, which is useless for a scalper.
        """
        series = inputs.series
        if not series.ready(40):
            return self.NEUTRAL
        ratio = ind.last_valid(ind.volume_ratio(series.volumes, 30), default=1.0)
        # 0.5x -> 0, 1x -> 50, 2x+ -> 100
        return normalise_score(ratio, 0.5, 2.0)

    def spread_score(self, inputs: ScoringInputs) -> float:
        """Tighter is better. Scored against the configured maximum."""
        spread = inputs.liquidity.spread_bps
        if spread == float("inf"):
            return 0.0
        return normalise_score(spread, 0.0, self.config.max_spread_bps, invert=True)

    def volatility_score(self, inputs: ScoringInputs) -> float:
        """Band score: too calm and too wild are both penalised."""
        series = inputs.series
        if not series.ready(20):
            return self.NEUTRAL
        atr_pct = ind.last_valid(
            ind.atr_percent(series.highs, series.lows, series.closes, 14), default=0.0
        )
        if atr_pct <= 0:
            return 0.0
        return band_score(
            atr_pct, self.config.volatility_target_low, self.config.volatility_target_high
        )

    def momentum_score(self, inputs: ScoringInputs) -> float:
        """Absolute recent rate of change — direction-agnostic.

        The scanner ranks *opportunity*, not direction; a strong down-move is as
        tradable as a strong up-move because the system trades both sides.
        """
        series = inputs.series
        if not series.ready(20):
            return self.NEUTRAL
        rate = ind.last_valid(ind.roc(series.closes, 10), default=0.0)
        return normalise_score(abs(rate), 0.0, 0.01)

    def trend_score(self, inputs: ScoringInputs) -> float:
        """ADX-based trend strength, direction-agnostic."""
        series = inputs.series
        if not series.ready(40):
            return self.NEUTRAL
        adx_value = ind.last_valid(
            ind.adx(series.highs, series.lows, series.closes, 14)[0], default=0.0
        )
        return normalise_score(adx_value, 10.0, 45.0)

    def volume_anomaly_score(self, inputs: ScoringInputs) -> float:
        """How unusual current volume is, in standard deviations."""
        series = inputs.series
        if not series.ready(40):
            return self.NEUTRAL
        z = ind.last_valid(ind.volume_zscore(series.volumes, 30), default=0.0)
        return normalise_score(z, -1.0, 4.0)

    def breakout_potential_score(self, inputs: ScoringInputs) -> float:
        """A compressed range that is coiling — bandwidth low relative to itself."""
        series = inputs.series
        if not series.ready(60):
            return self.NEUTRAL
        bandwidth = ind.bollinger_bandwidth(series.closes, 20, 2.0)
        current = ind.last_valid(bandwidth, default=0.0)
        history = bandwidth[np.isfinite(bandwidth)]
        if current <= 0 or history.size < 20:
            return self.NEUTRAL
        # Percentile rank of the current bandwidth: low rank = tight squeeze.
        rank = float((history < current).mean())
        return (1.0 - rank) * 100.0

    def mean_reversion_potential_score(self, inputs: ScoringInputs) -> float:
        """How stretched price is from its mean, in a non-trending market.

        Stretch alone is not enough — a strong trend is stretched by definition
        and fading it is how accounts die. So this is damped by trend strength.
        """
        series = inputs.series
        if not series.ready(40):
            return self.NEUTRAL
        closes = series.closes
        _, middle, _ = ind.bollinger(closes, 20, 2.0)
        centre = ind.last_valid(middle, default=0.0)
        atr_value = ind.last_valid(ind.atr(series.highs, series.lows, closes, 14), default=0.0)
        if centre <= 0 or atr_value <= 0:
            return self.NEUTRAL
        stretch_atr = abs(closes[-1] - centre) / atr_value
        stretch = normalise_score(stretch_atr, 0.0, 3.0)

        adx_value = ind.last_valid(ind.adx(series.highs, series.lows, closes, 14)[0], default=20.0)
        trend_damping = normalise_score(adx_value, 15.0, 40.0, invert=True) / 100.0
        return stretch * trend_damping

    def funding_score(self, inputs: ScoringInputs) -> float:
        """Extreme funding is both a cost and a crowding signal.

        Near-zero funding is cheapest to trade in either direction, so it scores
        highest. Extreme funding does carry information about positioning, but
        the scanner treats it primarily as a cost.
        """
        rate = abs(inputs.funding_rate)
        # 0.01% per 8h is normal; 0.1% is extreme.
        return normalise_score(rate, 0.0, 0.001, invert=True)

    def structure_score(self, inputs: ScoringInputs) -> float:
        """Clean higher-highs/lower-lows structure, direction-agnostic."""
        series = inputs.series
        if not series.ready(50):
            return self.NEUTRAL
        return abs(ind.structure_score(series.closes, 50)) * 100.0

    def book_imbalance_score(self, inputs: ScoringInputs) -> float:
        """Magnitude of order-book skew. Neutral when depth is unknown.

        Scored by absolute value: an imbalance in either direction is a tradable
        signal, and the strategies decide which way to take it.
        """
        if inputs.liquidity.depth_notional <= 0:
            return self.NEUTRAL
        return normalise_score(abs(inputs.liquidity.book_imbalance), 0.0, 0.5)

    # ------------------------------------------------------------------ #
    # Penalties — subtracted, in points, after the weighted sum
    # ------------------------------------------------------------------ #
    def cost_penalty(self, inputs: ScoringInputs) -> float:
        """Scaled by how expensive a round trip is relative to typical volatility.

        A symbol whose costs eat most of its typical move is nearly untradable
        no matter how attractive it otherwise looks.
        """
        cost = self.cost_model.breakeven_move(inputs.liquidity)
        series = inputs.series
        atr_pct = (
            ind.last_valid(
                ind.atr_percent(series.highs, series.lows, series.closes, 14),
                default=0.0,
            )
            if series.ready(20)
            else 0.0
        )
        if atr_pct <= 0:
            return self.config.penalties.estimated_cost
        # Cost as a fraction of one ATR of movement.
        ratio = clamp(safe_div(cost, atr_pct, 1.0), 0.0, 1.0)
        return self.config.penalties.estimated_cost * ratio

    def risk_penalty(self, inputs: ScoringInputs) -> float:
        """Penalise instability that makes any estimate unreliable."""
        series = inputs.series
        penalty = 0.0
        max_penalty = self.config.penalties.risk

        if not series.ready(30):
            return max_penalty  # not enough history to judge anything

        # Volatility-of-volatility: an erratic ATR means stops sized from it
        # will be wrong.
        atr_series = ind.atr_percent(series.highs, series.lows, series.closes, 14)
        finite = atr_series[np.isfinite(atr_series)]
        if finite.size >= 20:
            recent = finite[-20:]
            instability = safe_div(float(np.std(recent)), float(np.mean(recent)), 0.0)
            penalty += max_penalty * 0.5 * clamp(instability / 0.5, 0.0, 1.0)

        # Correlation with what we already hold.
        penalty += max_penalty * 0.5 * clamp(inputs.correlation_penalty, 0.0, 1.0)
        return clamp(penalty, 0.0, max_penalty)

    # ------------------------------------------------------------------ #
    # Composite
    # ------------------------------------------------------------------ #
    def score(self, inputs: ScoringInputs) -> MarketScore:
        components = {
            "liquidity": self.liquidity_score(inputs),
            "volume": self.volume_score(inputs),
            "recent_volume": self.recent_volume_score(inputs),
            "spread": self.spread_score(inputs),
            "volatility": self.volatility_score(inputs),
            "momentum": self.momentum_score(inputs),
            "trend": self.trend_score(inputs),
            "volume_anomaly": self.volume_anomaly_score(inputs),
            "breakout_potential": self.breakout_potential_score(inputs),
            "mean_reversion_potential": self.mean_reversion_potential_score(inputs),
            "funding": self.funding_score(inputs),
            "structure": self.structure_score(inputs),
            "book_imbalance": self.book_imbalance_score(inputs),
        }

        weighted = sum(
            components[name] * weight
            for name, weight in self._weights.items()
            if name in components
        )

        penalties = {
            "estimated_cost": self.cost_penalty(inputs),
            "risk": self.risk_penalty(inputs),
        }

        total = clamp(weighted - sum(penalties.values()), 0.0, 100.0)

        series = inputs.series
        volatility = (
            ind.last_valid(
                ind.atr_percent(series.highs, series.lows, series.closes, 14),
                default=0.0,
            )
            if series.ready(20)
            else 0.0
        )

        return MarketScore(
            symbol=inputs.symbol,
            total=total,
            components=components,
            penalties=penalties,
            volatility=volatility,
            liquidity_usd=inputs.liquidity.depth_notional,
            spread_bps=inputs.liquidity.spread_bps,
            funding_rate=inputs.funding_rate,
            timestamp=inputs.timestamp,
        )
