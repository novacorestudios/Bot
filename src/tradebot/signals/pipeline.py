"""The analytical pipeline: signals → consensus → opportunity → edge.

Everything from a symbol's market data up to (but not including) the risk
engine. Kept as one object so the audit trail is produced in one place and every
rejection is recorded with the same structure, whether it happened at the
consensus stage or the edge stage.

The pipeline never returns "no" without saying why. Being able to answer "why
didn't it trade?" is as important as answering "why did it?" — a bot that
silently declines everything looks identical to a bot that is broken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tradebot.core.config import TunableConfig
from tradebot.core.logging import get_logger
from tradebot.core.types import (
    AggregatedSignal,
    Direction,
    MarketRegime,
    MarketScore,
    OpportunityScore,
    RejectionReason,
    Signal,
)
from tradebot.market.microstructure import CostModel, LiquiditySnapshot
from tradebot.signals.aggregator import AggregationResult, SignalAggregator
from tradebot.signals.edge import EdgeCalculator, EdgeDecision
from tradebot.signals.opportunity import OpportunityInputs, OpportunityScorer
from tradebot.strategies.base import MarketView

if TYPE_CHECKING:
    from tradebot.backtesting.diagnostics import CandidateRecorder
from tradebot.strategies.registry import StrategyRegistry

log = get_logger(__name__)


@dataclass(slots=True)
class Opportunity:
    """A candidate trade that has cleared every analytical gate."""

    symbol: str
    signal: AggregatedSignal
    opportunity_score: OpportunityScore
    edge: EdgeDecision
    market: MarketScore
    liquidity: LiquiditySnapshot
    regime: MarketRegime
    notional_estimate: float
    timestamp: int

    @property
    def direction(self) -> Direction:
        return self.signal.direction

    @property
    def strategy(self) -> str:
        """The strategy this trade belongs to.

        Delegates to `AggregatedSignal.primary_strategy` so that edge
        statistics, risk allocation and trade ownership cannot disagree.
        """
        return self.signal.primary_strategy

    @property
    def contributing_strategies(self) -> tuple[str, ...]:
        """Every strategy that agreed, primary first."""
        return self.signal.contributing_strategies

    @property
    def contribution_weights(self) -> dict[str, float]:
        """Each contributor's share of the agreeing confidence."""
        return self.signal.contribution_weights

    @property
    def expected_net_edge(self) -> float:
        return self.edge.expected_net


@dataclass(slots=True)
class PipelineResult:
    """Outcome for one symbol, accepted or not, always with a reason."""

    symbol: str
    opportunity: Opportunity | None
    rejection: RejectionReason | None
    detail: str
    stage: str
    signals: list[Signal] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.opportunity is not None


class SignalPipeline:
    """Runs strategies, aggregates, scores and applies the edge filter."""

    def __init__(
        self, config: TunableConfig, registry: StrategyRegistry, cost_model: CostModel
    ) -> None:
        self.config = config
        self.registry = registry
        self.cost_model = cost_model
        self.aggregator = SignalAggregator(config.aggregator)
        self.opportunity_scorer = OpportunityScorer(config.opportunity, config.scanner)
        self.edge_calculator = EdgeCalculator(config.edge, cost_model)

        self.evaluated = 0
        self.accepted = 0
        self.rejections: dict[str, int] = {}

        #: Observational only. `None` in every normal run; a diagnostic attaches
        #: a CandidateRecorder here to capture the scalars each gate thresholds.
        #: Nothing in this class reads it to decide anything.
        self.recorder: CandidateRecorder | None = None

    def evaluate(
        self,
        view: MarketView,
        market: MarketScore,
        liquidity: LiquiditySnapshot,
        notional_estimate: float,
        correlation: float = 0.0,
        strategy_allocation: dict[str, float] | None = None,
        seconds_to_funding: float = float("inf"),
        now: float = 0.0,
    ) -> PipelineResult:
        """Run the full analytical pipeline for one symbol."""
        self.evaluated += 1
        symbol = view.symbol
        timestamp = view.now_ms or int(time.time() * 1000)

        # -- 0. regime gate --------------------------------------------------
        if view.regime.blocks_entries:
            return self._reject(
                symbol,
                RejectionReason.REGIME_BLOCKED,
                f"regime {view.regime.value} permits no entries",
                "regime",
            )

        # -- 1. strategies ---------------------------------------------------
        signals, weights = self.registry.evaluate(view, now)
        if not weights:
            return self._reject(
                symbol,
                RejectionReason.REGIME_BLOCKED,
                f"no strategy is enabled for regime {view.regime.value}",
                "regime",
                signals,
            )

        # -- 2. consensus ----------------------------------------------------
        if self.config.trade.raw_signal_mode:
            aggregation = self._raw_signal(symbol, signals, weights, view.regime, timestamp)
        else:
            aggregation = self.aggregator.aggregate(
                symbol, signals, weights, view.regime, timestamp, strategy_allocation
            )
        if not aggregation.accepted:
            return self._reject(
                symbol,
                aggregation.rejection or RejectionReason.NO_SIGNAL,
                aggregation.detail,
                "aggregation",
                signals,
                long_weight=aggregation.long_weight,
                short_weight=aggregation.short_weight,
                agreeing=aggregation.agreeing,
                consensus=aggregation.consensus,
            )

        signal = aggregation.signal
        if signal is None:
            # Unreachable via AggregationResult's contract, but an assert would
            # be stripped under -O and turn this into an AttributeError several
            # frames away. A rejection with a reason is the safe failure.
            return self._reject(
                symbol,
                RejectionReason.NO_SIGNAL,
                "aggregation reported success without producing a signal",
                "aggregation",
                signals,
            )

        # -- 3. expected net edge -------------------------------------------
        # Computed BEFORE the opportunity score because the opportunity score
        # needs the cost total, and because a negative-edge trade should be
        # rejected on that ground specifically — it is the more informative
        # reason for the audit log.
        expected_duration = self._expected_duration(signal)
        edge = self.edge_calculator.evaluate(
            signal,
            liquidity,
            notional_estimate,
            funding_rate=view.funding_rate,
            seconds_to_funding=seconds_to_funding,
            expected_duration_sec=expected_duration,
        )

        # -- 4. opportunity score -------------------------------------------
        opportunity_score = self.opportunity_scorer.score(
            OpportunityInputs(
                signal=signal,
                market=market,
                liquidity=liquidity,
                expected_net_edge=edge.expected_net,
                cost_total=edge.estimate.costs.total,
                correlation=correlation,
                notional=notional_estimate,
            )
        )

        if not self.config.trade.raw_signal_mode and not self.opportunity_scorer.accepts(
            opportunity_score
        ):
            return self._reject(
                symbol,
                RejectionReason.LOW_OPPORTUNITY_SCORE,
                f"score {opportunity_score.total:.1f} below "
                f"{self.config.opportunity.min_score:.1f}",
                "opportunity",
                signals,
                opportunity_score=opportunity_score.total,
                components=opportunity_score.components,
                expected_net_edge=edge.expected_net,
                agreeing=len(signal.contributing),
                consensus=signal.consensus_score,
            )

        if not self.config.trade.raw_signal_mode and not edge.accepted:
            breakeven = self.edge_calculator.breakeven_win_rate(
                edge.estimate.gross_win,
                edge.estimate.gross_loss,
                edge.estimate.costs,
            )
            return self._reject(
                symbol,
                RejectionReason.NEGATIVE_EXPECTED_EDGE,
                f"{edge.detail}; needs a {breakeven:.1%} win rate to break even",
                "edge",
                signals,
                expected_net_edge=edge.expected_net,
                costs=edge.estimate.costs.as_dict(),
                win_probability=edge.estimate.win_probability,
                breakeven_win_rate=breakeven,
                opportunity_score=opportunity_score.total,
                agreeing=len(signal.contributing),
                consensus=signal.consensus_score,
            )

        self.accepted += 1
        if self.recorder is not None:
            self.recorder.record(
                symbol,
                "complete",
                None,
                agreeing=len(signal.contributing),
                consensus=signal.consensus_score,
                expected_net=edge.expected_net,
            )
        opportunity = Opportunity(
            symbol=symbol,
            signal=signal,
            opportunity_score=opportunity_score,
            edge=edge,
            market=market,
            liquidity=liquidity,
            regime=view.regime,
            notional_estimate=notional_estimate,
            timestamp=timestamp,
        )

        log.info(
            "opportunity_accepted",
            symbol=symbol,
            direction=signal.direction.value,
            strategy=opportunity.strategy,
            regime=view.regime.value,
            consensus=round(signal.consensus_score, 1),
            opportunity_score=round(opportunity_score.total, 1),
            grade=self.opportunity_scorer.grade(opportunity_score),
            expected_net_edge=round(edge.expected_net, 6),
            strategies=list(signal.strategies),
        )
        return PipelineResult(
            symbol=symbol,
            opportunity=opportunity,
            rejection=None,
            detail="accepted",
            stage="complete",
            signals=signals,
            audit=self._audit(signal, opportunity_score, edge, view),
        )

    # ------------------------------------------------------------------ #
    def _raw_signal(
        self,
        symbol: str,
        signals: list[Signal],
        weights: dict[str, float],
        regime: MarketRegime,
        timestamp: int,
    ) -> AggregationResult:
        """Return the strongest executable raw opinion for this symbol/cycle."""
        actionable = [
            signal
            for signal in signals
            if signal.direction is not Direction.WAIT
            and signal.entry_price > 0
            and weights.get(signal.strategy, 0.0) > 0
        ]
        if not actionable:
            return AggregationResult(
                None,
                RejectionReason.NO_SIGNAL,
                "no strategy produced an actionable raw signal",
                considered=len(signals),
                agreeing=0,
            )

        chosen = max(
            actionable,
            key=lambda signal: (
                weights.get(signal.strategy, 0.0) * signal.confidence,
                signal.strategy,
            ),
        )
        cfg = self.config.trade
        target_pct = max(
            cfg.raw_take_profit_min_pct,
            min(cfg.raw_take_profit_max_pct, chosen.volatility),
        )
        sign = 1.0 if chosen.direction is Direction.LONG else -1.0
        stop = chosen.entry_price * (1.0 - sign * cfg.raw_stop_pct)
        target = chosen.entry_price * (1.0 + sign * target_pct)
        opposing = tuple(s for s in actionable if s.direction is not chosen.direction)
        raw = AggregatedSignal(
            symbol=symbol,
            direction=chosen.direction,
            consensus_score=chosen.confidence,
            confidence=chosen.confidence,
            entry_price=chosen.entry_price,
            stop_loss=stop,
            take_profit=target,
            contributing=(chosen,),
            opposing=opposing,
            conflict_ratio=0.0,
            regime=regime,
            timestamp=timestamp,
            reason_codes=chosen.reason_codes,
            metadata={
                "raw_signal_mode": True,
                "raw_strategy": chosen.strategy,
                "raw_target_pct": target_pct,
                "raw_stop_pct": cfg.raw_stop_pct,
            },
        )
        return AggregationResult(
            raw,
            considered=len(signals),
            agreeing=1,
            consensus=chosen.confidence,
        )

    def _expected_duration(self, signal: AggregatedSignal) -> float:
        durations = [
            s.expected_duration_sec for s in signal.contributing if s.expected_duration_sec > 0
        ]
        if durations:
            return float(sum(durations) / len(durations))
        return float(self.config.trade.max_duration_sec) / 2.0

    def _reject(
        self,
        symbol: str,
        reason: RejectionReason,
        detail: str,
        stage: str,
        signals: list[Signal] | None = None,
        **audit: Any,
    ) -> PipelineResult:
        self.rejections[reason.value] = self.rejections.get(reason.value, 0) + 1
        if self.recorder is not None:
            # Purely additive: `audit` is already built for the log, and nothing
            # below reads what is recorded here.
            self.recorder.record(
                symbol,
                stage,
                reason.value,
                agreeing=audit.get("agreeing"),
                consensus=audit.get("consensus"),
                expected_net=audit.get("expected_net_edge"),
            )
        log.debug(
            "opportunity_rejected", symbol=symbol, reason=reason.value, stage=stage, detail=detail
        )
        return PipelineResult(
            symbol=symbol,
            opportunity=None,
            rejection=reason,
            detail=detail,
            stage=stage,
            signals=signals or [],
            audit=audit,
        )

    def _audit(
        self,
        signal: AggregatedSignal,
        score: OpportunityScore,
        edge: EdgeDecision,
        view: MarketView,
    ) -> dict[str, Any]:
        """The full 'why did it enter?' record required by the brief."""
        return {
            "regime": view.regime.value,
            "regime_confidence": view.regime_confidence,
            "direction": signal.direction.value,
            "consensus_score": signal.consensus_score,
            "confidence": signal.confidence,
            "conflict_ratio": signal.conflict_ratio,
            "agreeing_strategies": list(signal.strategies),
            "opposing_strategies": [s.strategy for s in signal.opposing],
            "reason_codes": list(signal.reason_codes),
            "opportunity_score": score.total,
            "opportunity_components": score.components,
            "opportunity_penalties": score.penalties,
            "expected_net_edge": edge.expected_net,
            "expected_gross_edge": edge.estimate.expected_gross,
            "win_probability": edge.estimate.win_probability,
            "costs": edge.estimate.costs.as_dict(),
            "entry": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_reward": signal.risk_reward,
            "spread_bps": view.spread_bps,
            "book_imbalance": view.book_imbalance,
            "funding_rate": view.funding_rate,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / max(1, self.evaluated),
            "rejections": dict(self.rejections),
            "edge_model": self.edge_calculator.summary(),
        }
