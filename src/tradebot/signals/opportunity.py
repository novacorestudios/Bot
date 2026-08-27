"""Opportunity scoring.

The final analytical ranking before risk. It blends *market quality* (from the
scanner) with *signal quality* (from the aggregator) and *execution quality*
(from the order book), then subtracts penalties for cost and correlation.

Why a separate score when the scanner already ranked the market and the
aggregator already scored consensus: those two answer different questions. The
scanner says "this market is worth watching"; the aggregator says "the
strategies agree". Neither asks "is *this specific trade*, right now, at this
spread, with this reward:risk, and given what we already hold, better than doing
nothing?" That is what this score is for, and it is why a top-ranked market with
a strong consensus can still be correctly rejected.

The grading bands (exceptional / strong / moderate / reject) are configuration,
not truth. They are starting points to be re-fitted once real trade outcomes
exist to fit them against.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.core.config import OpportunityConfig, ScannerConfig
from tradebot.core.mathutil import band_score, clamp, normalise_score, safe_div
from tradebot.core.types import (
    AggregatedSignal,
    MarketScore,
    OpportunityScore,
)
from tradebot.market.microstructure import LiquiditySnapshot


@dataclass(slots=True)
class OpportunityInputs:
    """Everything the opportunity score consumes."""

    signal: AggregatedSignal
    market: MarketScore
    liquidity: LiquiditySnapshot
    expected_net_edge: float
    cost_total: float
    correlation: float = 0.0  # 0..1, portfolio correlation if taken
    notional: float = 0.0


class OpportunityScorer:
    """Produces the final 0-100 pre-risk score."""

    def __init__(self, config: OpportunityConfig, scanner_config: ScannerConfig) -> None:
        self.config = config
        self.scanner_config = scanner_config
        self._weights = config.weights.normalised()

    # -- components, each 0..100 ------------------------------------------ #
    def market_quality(self, inputs: OpportunityInputs) -> float:
        return clamp(inputs.market.total, 0.0, 100.0)

    def consensus(self, inputs: OpportunityInputs) -> float:
        return clamp(inputs.signal.consensus_score, 0.0, 100.0)

    def momentum(self, inputs: OpportunityInputs) -> float:
        return clamp(inputs.market.components.get("momentum", 50.0), 0.0, 100.0)

    def volume(self, inputs: OpportunityInputs) -> float:
        return clamp(inputs.market.components.get("recent_volume", 50.0), 0.0, 100.0)

    def trend(self, inputs: OpportunityInputs) -> float:
        return clamp(inputs.market.components.get("trend", 50.0), 0.0, 100.0)

    def liquidity(self, inputs: OpportunityInputs) -> float:
        return clamp(inputs.market.components.get("liquidity", 50.0), 0.0, 100.0)

    def volatility_fit(self, inputs: OpportunityInputs) -> float:
        """Is this symbol's volatility in the band this system trades well?"""
        return band_score(
            inputs.market.volatility,
            self.scanner_config.volatility_target_low,
            self.scanner_config.volatility_target_high,
        )

    def execution_quality(self, inputs: OpportunityInputs) -> float:
        """How cheaply and reliably can this size actually be executed?

        Combines the quoted spread with how much of the visible depth this order
        would consume. A tight spread on a book too thin to absorb the order is
        not good execution.
        """
        spread = normalise_score(
            inputs.liquidity.spread_bps,
            0.0,
            self.scanner_config.max_spread_bps,
            invert=True,
        )
        depth = inputs.liquidity.depth_for(inputs.signal.direction)
        if depth <= 0 or inputs.notional <= 0:
            return spread * 0.5  # unknown depth: assume the worse half
        participation = safe_div(inputs.notional, depth, 1.0)
        # Consuming under 1% of visible depth is ideal; 20%+ is poor.
        depth_score = normalise_score(participation, 0.01, 0.20, invert=True)
        return spread * 0.5 + depth_score * 0.5

    def risk_reward(self, inputs: OpportunityInputs) -> float:
        """Reward:risk relative to what this system considers acceptable."""
        rr = inputs.signal.risk_reward
        if rr <= 0:
            return 0.0
        return normalise_score(rr, 1.0, 3.0)

    # -- penalties, in points --------------------------------------------- #
    def cost_penalty(self, inputs: OpportunityInputs) -> float:
        """Scaled by how much of the expected gross move the costs consume."""
        signal = inputs.signal
        entry = signal.entry_price
        if entry <= 0:
            return self.config.penalties.cost
        gross = safe_div(abs(signal.take_profit - entry), entry, 0.0)
        if gross <= 0:
            return self.config.penalties.cost
        ratio = clamp(safe_div(inputs.cost_total, gross, 1.0), 0.0, 1.0)
        return self.config.penalties.cost * ratio

    def correlation_penalty(self, inputs: OpportunityInputs) -> float:
        """Scaled by correlation with existing exposure.

        Four correlated longs are one leveraged bet wearing four names. The
        correlation engine computes the number; this converts it into a score
        reduction so a crowded trade must be materially better than an
        uncorrelated one to be taken.
        """
        return self.config.penalties.correlation * clamp(inputs.correlation, 0.0, 1.0)

    # -- composite --------------------------------------------------------- #
    def score(self, inputs: OpportunityInputs) -> OpportunityScore:
        components = {
            "market_quality": self.market_quality(inputs),
            "consensus": self.consensus(inputs),
            "momentum": self.momentum(inputs),
            "volume": self.volume(inputs),
            "trend": self.trend(inputs),
            "liquidity": self.liquidity(inputs),
            "volatility_fit": self.volatility_fit(inputs),
            "execution_quality": self.execution_quality(inputs),
            "risk_reward": self.risk_reward(inputs),
        }
        weighted = sum(
            components[name] * weight
            for name, weight in self._weights.items()
            if name in components
        )

        penalties = {
            "cost": self.cost_penalty(inputs),
            "correlation": self.correlation_penalty(inputs),
        }

        total = clamp(weighted - sum(penalties.values()), 0.0, 100.0)
        return OpportunityScore(total=total, components=components, penalties=penalties)

    def accepts(self, score: OpportunityScore) -> bool:
        return score.total >= self.config.min_score

    def grade(self, score: OpportunityScore) -> str:
        cfg = self.config
        if score.total >= cfg.exceptional:
            return "EXCEPTIONAL"
        if score.total >= cfg.strong:
            return "STRONG"
        if score.total >= cfg.moderate:
            return "MODERATE"
        return "REJECT"
