"""Signal aggregation.

Several strategies look at the same symbol. This module decides whether their
combined opinion is worth acting on, and what the resulting entry, stop and
target should be.

The important behaviour is the **conflict rule**. When strategies disagree
strongly, the correct action is to stand aside, not to side with the louder
one. Two strategies saying LONG at 90 while two say SHORT at 85 is not a
"weak long"; it is a market that has not decided, and entering it means paying
the spread to hold a position with no thesis.

Weighting combines two independent factors:

* the **regime weight** — how appropriate that strategy is to current conditions
* the **strategy's own confidence** — how strong its signal is

so a strategy that is highly confident but poorly suited to the regime does not
dominate one that is well suited and moderately confident.

The consensus stop is the **most conservative** of the contributing stops, and
the target the **least ambitious**. That is deliberate: when strategies disagree
about levels, the trade should survive the pessimistic case, and a target that
only the most optimistic strategy believes in is one the market rarely reaches.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.core.config import AggregatorConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp, safe_div
from tradebot.core.types import (
    AggregatedSignal,
    Direction,
    MarketRegime,
    RejectionReason,
    Signal,
)

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Either an aggregated signal, or a recorded reason there is none."""

    signal: AggregatedSignal | None
    rejection: RejectionReason | None = None
    detail: str = ""
    long_weight: float = 0.0
    short_weight: float = 0.0
    considered: int = 0

    @property
    def accepted(self) -> bool:
        return self.signal is not None


class SignalAggregator:
    """Combines per-strategy signals into one opinion, or refuses."""

    def __init__(self, config: AggregatorConfig) -> None:
        self.config = config

    def aggregate(
        self,
        symbol: str,
        signals: list[Signal],
        weights: dict[str, float],
        regime: MarketRegime,
        timestamp: int = 0,
        strategy_allocation: dict[str, float] | None = None,
    ) -> AggregationResult:
        """Combine signals for one symbol.

        ``strategy_allocation`` is the risk-budget multiplier from the allocator,
        which lets a historically strong strategy carry more weight in consensus
        without changing its own logic.
        """
        cfg = self.config
        actionable = [s for s in signals if s.direction is not Direction.WAIT]

        if not actionable:
            return AggregationResult(
                None,
                RejectionReason.NO_SIGNAL,
                "no strategy produced a direction",
                considered=len(signals),
            )

        # Confidence below the floor is treated as no opinion at all.
        confident = [s for s in actionable if s.confidence >= cfg.min_signal_confidence]
        if not confident:
            best = max(actionable, key=lambda s: s.confidence)
            return AggregationResult(
                None,
                RejectionReason.LOW_CONFIDENCE,
                f"best confidence {best.confidence:.0f} below {cfg.min_signal_confidence:.0f}",
                considered=len(signals),
            )

        def weight_of(signal: Signal) -> float:
            regime_weight = weights.get(signal.strategy, 0.0)
            allocation = (strategy_allocation or {}).get(signal.strategy, 1.0)
            return regime_weight * allocation * (signal.confidence / 100.0)

        longs = [s for s in confident if s.direction is Direction.LONG]
        shorts = [s for s in confident if s.direction is Direction.SHORT]
        long_weight = sum(weight_of(s) for s in longs)
        short_weight = sum(weight_of(s) for s in shorts)

        if long_weight >= short_weight:
            direction, agreeing, opposing = Direction.LONG, longs, shorts
            agree_weight, oppose_weight = long_weight, short_weight
        else:
            direction, agreeing, opposing = Direction.SHORT, shorts, longs
            agree_weight, oppose_weight = short_weight, long_weight

        if agree_weight <= 0:
            return AggregationResult(
                None,
                RejectionReason.NO_SIGNAL,
                "all weights are zero",
                long_weight=long_weight,
                short_weight=short_weight,
                considered=len(signals),
            )

        # -- conflict: disagreement means stand aside ----------------------- #
        conflict_ratio = safe_div(oppose_weight, agree_weight, 0.0)
        if conflict_ratio > cfg.max_conflict_ratio:
            return AggregationResult(
                None,
                RejectionReason.CONFLICTING_SIGNALS,
                f"opposing weight {oppose_weight:.2f} is {conflict_ratio:.0%} of "
                f"agreeing weight {agree_weight:.2f}; the market has not decided",
                long_weight=long_weight,
                short_weight=short_weight,
                considered=len(signals),
            )

        # -- breadth: one strategy is an opinion, several are a consensus ---- #
        if len(agreeing) < cfg.min_agreeing_strategies:
            return AggregationResult(
                None,
                RejectionReason.INSUFFICIENT_CONSENSUS,
                f"only {len(agreeing)} strategy agrees; {cfg.min_agreeing_strategies} required",
                long_weight=long_weight,
                short_weight=short_weight,
                considered=len(signals),
            )

        consensus = self._consensus_score(agreeing, opposing, agree_weight, oppose_weight, weights)
        if consensus < cfg.min_consensus:
            return AggregationResult(
                None,
                RejectionReason.INSUFFICIENT_CONSENSUS,
                f"consensus {consensus:.1f} below {cfg.min_consensus:.1f}",
                long_weight=long_weight,
                short_weight=short_weight,
                considered=len(signals),
            )

        entry, stop, target = self._combine_levels(agreeing, direction, weight_of)
        weighted_confidence = safe_div(
            sum(s.confidence * weight_of(s) for s in agreeing), agree_weight, 0.0
        )

        signal = AggregatedSignal(
            symbol=symbol,
            direction=direction,
            consensus_score=consensus,
            confidence=clamp(weighted_confidence, 0.0, 100.0),
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            contributing=tuple(agreeing),
            opposing=tuple(opposing),
            conflict_ratio=conflict_ratio,
            regime=regime,
            timestamp=timestamp,
            reason_codes=tuple(code for s in agreeing for code in s.reason_codes)[:12],
            metadata={
                "agreeing_strategies": [s.strategy for s in agreeing],
                "opposing_strategies": [s.strategy for s in opposing],
                "agree_weight": agree_weight,
                "oppose_weight": oppose_weight,
            },
        )

        # A combined signal whose geometry is broken is a bug in this module.
        if signal.stop_distance <= 0:
            return AggregationResult(
                None,
                RejectionReason.INVALID_STOP,
                "combined stop distance is zero",
                long_weight=long_weight,
                short_weight=short_weight,
                considered=len(signals),
            )

        return AggregationResult(
            signal, long_weight=long_weight, short_weight=short_weight, considered=len(signals)
        )

    # ------------------------------------------------------------------ #
    def _consensus_score(
        self,
        agreeing: list[Signal],
        opposing: list[Signal],
        agree_weight: float,
        oppose_weight: float,
        weights: dict[str, float],
    ) -> float:
        """0..100 combining agreement share, breadth and confidence.

        All three matter independently: five weak strategies agreeing is not the
        same as two strong ones, and either can beat one very confident outlier.
        """
        total_weight = agree_weight + oppose_weight
        agreement_share = safe_div(agree_weight, total_weight, 0.0)

        # Breadth relative to how many strategies the regime permits at all.
        available = max(len(weights), 1)
        breadth = clamp(len(agreeing) / available, 0.0, 1.0)

        mean_confidence = safe_div(sum(s.confidence for s in agreeing), len(agreeing), 0.0) / 100.0

        score = (agreement_share * 45.0) + (breadth * 25.0) + (mean_confidence * 30.0)
        return clamp(score, 0.0, 100.0)

    def _combine_levels(
        self, agreeing: list[Signal], direction: Direction, weight_of
    ) -> tuple[float, float, float]:
        """Combine levels as FRACTIONAL DISTANCES, not absolute prices.

        Strategies run on different timeframes and therefore compute their
        levels from different reference closes. Averaging their absolute stop
        and target prices mixes those references and can invert the resulting
        risk:reward — combining a 3m strategy's tight stop with a 5m strategy's
        near target produced an R below 1 in testing, which the edge filter then
        rejected every time.

        So each signal's stop and target are converted to fractions of its own
        entry, combined in that space, and re-applied to the consensus entry:

        * **stop** — weighted mean distance. This is the consensus view of where
          the shared thesis is falsified. The mean rather than the tightest,
          because a 5m thesis given a 3m stop is stopped out by noise before it
          can be right.
        * **target** — the *smallest* distance among the agreeing strategies. A
          target only the most optimistic strategy believes in is one the market
          rarely reaches, which converts winners into time-based exits.
        """
        total = sum(weight_of(s) for s in agreeing)
        entry = (
            safe_div(sum(s.entry_price * weight_of(s) for s in agreeing), total, 0.0)
            or agreeing[0].entry_price
        )
        if entry <= 0:
            return 0.0, 0.0, 0.0

        stop_fractions: list[tuple[float, float]] = []  # (distance, weight)
        target_fractions: list[float] = []
        for candidate in agreeing:
            reference = candidate.entry_price
            if reference <= 0:
                continue
            if candidate.stop_loss > 0:
                stop_fractions.append(
                    (abs(reference - candidate.stop_loss) / reference, weight_of(candidate))
                )
            if candidate.take_profit > 0:
                target_fractions.append(abs(candidate.take_profit - reference) / reference)

        stop_weight = sum(weight for _, weight in stop_fractions)
        stop_distance = (
            safe_div(sum(d * w for d, w in stop_fractions), stop_weight, 0.0)
            if stop_fractions
            else 0.0
        )
        target_distance = min(target_fractions) if target_fractions else 0.0

        # Fall back to the configured floor rather than emitting a zero-distance
        # stop, which would divide by zero downstream.
        if stop_distance <= 0:
            stop_distance = 0.005
        if target_distance <= 0:
            target_distance = stop_distance

        sign = direction.sign
        stop = entry * (1 - sign * stop_distance)
        target = entry * (1 + sign * target_distance)
        return entry, stop, target
