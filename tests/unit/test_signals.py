"""Signal aggregation, opportunity scoring and the expected-net-edge gate.

The edge gate is the most consequential arithmetic in the system: it decides
whether a technically valid signal is worth the cost of trading it. These tests
pin both directions of that — it must reject trades whose costs exceed their
expected gross, and it must NOT be so strict that a genuinely good trade cannot
pass.
"""

from __future__ import annotations

import pytest

from tradebot.core.config import (
    AggregatorConfig,
    EdgeConfig,
    OpportunityConfig,
    ScannerConfig,
)
from tradebot.core.types import (
    AggregatedSignal,
    Direction,
    MarketRegime,
    MarketScore,
    RejectionReason,
    Signal,
)
from tradebot.market.microstructure import CostModel, LiquiditySnapshot
from tradebot.signals.aggregator import SignalAggregator
from tradebot.signals.edge import EdgeCalculator, required_move_for_edge
from tradebot.signals.opportunity import OpportunityInputs, OpportunityScorer

LIQUID = LiquiditySnapshot(
    "X",
    spread_bps=1.0,
    bid_notional=500_000.0,
    ask_notional=500_000.0,
    book_imbalance=0.0,
    quote_volume_24h=1e9,
)
ILLIQUID = LiquiditySnapshot(
    "X",
    spread_bps=6.0,
    bid_notional=20_000.0,
    ask_notional=20_000.0,
    book_imbalance=0.0,
    quote_volume_24h=3e7,
)


def signal(
    strategy: str,
    direction: Direction,
    confidence: float,
    entry: float = 100.0,
    stop: float | None = None,
    target: float | None = None,
) -> Signal:
    if direction is Direction.LONG:
        stop = stop if stop is not None else entry * 0.995
        target = target if target is not None else entry * 1.010
    else:
        stop = stop if stop is not None else entry * 1.005
        target = target if target is not None else entry * 0.990
    return Signal(
        symbol="X",
        strategy=strategy,
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        timeframe="3m",
        signal_timestamp=0,
        expected_duration_sec=600,
    )


def aggregated(
    direction: Direction = Direction.LONG,
    entry: float = 100.0,
    stop_pct: float = 0.005,
    target_pct: float = 0.010,
    strategy: str = "momentum",
) -> AggregatedSignal:
    if direction is Direction.LONG:
        stop, target = entry * (1 - stop_pct), entry * (1 + target_pct)
    else:
        stop, target = entry * (1 + stop_pct), entry * (1 - target_pct)
    contributing = (signal(strategy, direction, 80.0, entry, stop, target),)
    return AggregatedSignal(
        symbol="X",
        direction=direction,
        consensus_score=75.0,
        confidence=80.0,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        contributing=contributing,
        opposing=(),
        conflict_ratio=0.0,
        regime=MarketRegime.STRONG_TREND,
        timestamp=0,
    )


WEIGHTS = {
    "momentum": 1.0,
    "trend_following": 1.0,
    "breakout": 0.8,
    "mean_reversion": 1.0,
    "vwap": 0.6,
}


class TestAggregation:
    def aggregator(self, **overrides) -> SignalAggregator:
        return SignalAggregator(AggregatorConfig(**overrides))

    def test_agreeing_strategies_produce_a_signal(self):
        signals = [
            signal("momentum", Direction.LONG, 88.0),
            signal("trend_following", Direction.LONG, 81.0),
            signal("breakout", Direction.LONG, 91.0),
        ]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        assert result.accepted
        assert result.signal.direction is Direction.LONG
        assert set(result.signal.strategies) == {"momentum", "trend_following", "breakout"}

    def test_strong_disagreement_stands_aside(self):
        """Two at 90 long against two at 85 short is not a weak long."""
        signals = [
            signal("momentum", Direction.LONG, 90.0),
            signal("trend_following", Direction.LONG, 88.0),
            signal("mean_reversion", Direction.SHORT, 85.0),
            signal("vwap", Direction.SHORT, 87.0),
        ]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        assert not result.accepted
        assert result.rejection is RejectionReason.CONFLICTING_SIGNALS

    def test_mild_disagreement_is_tolerated(self):
        """One dissenter should not veto a clear consensus."""
        signals = [
            signal("momentum", Direction.LONG, 88.0),
            signal("trend_following", Direction.LONG, 85.0),
            signal("breakout", Direction.LONG, 82.0),
            signal("mean_reversion", Direction.SHORT, 55.0),
        ]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        assert result.accepted
        assert result.signal.conflict_ratio > 0

    def test_single_strategy_is_not_a_consensus(self):
        result = self.aggregator(min_agreeing_strategies=2).aggregate(
            "X", [signal("momentum", Direction.LONG, 95.0)], WEIGHTS, MarketRegime.STRONG_TREND
        )
        assert not result.accepted
        assert result.rejection is RejectionReason.INSUFFICIENT_CONSENSUS

    def test_all_wait_produces_no_signal(self):
        signals = [Signal("X", "momentum", Direction.WAIT, 0.0, 0, 0, 0, "3m", 0)]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.SIDEWAYS)
        assert result.rejection is RejectionReason.NO_SIGNAL

    def test_low_confidence_signals_are_treated_as_no_opinion(self):
        signals = [
            signal("momentum", Direction.LONG, 20.0),
            signal("trend_following", Direction.LONG, 25.0),
        ]
        result = self.aggregator(min_signal_confidence=50.0).aggregate(
            "X", signals, WEIGHTS, MarketRegime.STRONG_TREND
        )
        assert result.rejection is RejectionReason.LOW_CONFIDENCE

    def test_regime_weight_and_confidence_both_matter(self):
        """A confident but regime-inappropriate strategy must not dominate."""
        signals = [
            signal("vwap", Direction.SHORT, 95.0),
            signal("momentum", Direction.LONG, 70.0),
            signal("trend_following", Direction.LONG, 70.0),
        ]
        weights = {"vwap": 0.2, "momentum": 1.0, "trend_following": 1.0}
        result = self.aggregator().aggregate("X", signals, weights, MarketRegime.STRONG_TREND)
        assert result.accepted
        assert result.signal.direction is Direction.LONG

    def test_strategy_allocation_influences_consensus(self):
        signals = [
            signal("momentum", Direction.LONG, 70.0),
            signal("trend_following", Direction.LONG, 70.0),
        ]
        plain = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        boosted = self.aggregator().aggregate(
            "X",
            signals,
            WEIGHTS,
            MarketRegime.STRONG_TREND,
            strategy_allocation={"momentum": 2.0, "trend_following": 2.0},
        )
        assert plain.accepted and boosted.accepted
        assert boosted.long_weight > plain.long_weight

    def test_combined_stop_is_the_weighted_mean_distance(self):
        """The consensus falsification point.

        Not the tightest: a 5m thesis given a 3m strategy's stop is stopped out
        by noise before it has a chance to be right.
        """
        signals = [
            signal("momentum", Direction.LONG, 80.0, 100.0, 99.0, 102.0),
            signal("trend_following", Direction.LONG, 80.0, 100.0, 98.0, 103.0),
        ]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        # distances 1% and 2%, equally weighted -> 1.5%
        assert result.signal.stop_loss == pytest.approx(98.5)

    def test_combined_target_is_the_least_ambitious(self):
        """A target only the most optimistic strategy believes in is a time-exit."""
        signals = [
            signal("momentum", Direction.LONG, 80.0, 100.0, 99.0, 102.0),
            signal("trend_following", Direction.LONG, 80.0, 100.0, 98.0, 103.0),
        ]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        assert result.signal.take_profit == pytest.approx(102.0)

    def test_levels_are_combined_as_distances_not_absolute_prices(self):
        """Strategies on different timeframes reference different closes.

        Combining their absolute levels mixes those references and can invert
        the risk:reward. Distances are reference-free, so they compose.
        """
        signals = [
            signal("momentum", Direction.LONG, 80.0, 100.0, 99.0, 102.0),
            # Same 1%/2% geometry, but referenced to a different close.
            signal("trend_following", Direction.LONG, 80.0, 105.0, 103.95, 107.1),
        ]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        assert result.accepted
        entry = result.signal.entry_price
        assert (entry - result.signal.stop_loss) / entry == pytest.approx(0.01, abs=1e-6)
        assert result.signal.risk_reward == pytest.approx(2.0, abs=0.01)

    def test_short_consensus_levels_are_inverted_correctly(self):
        signals = [
            signal("momentum", Direction.SHORT, 80.0, 100.0, 101.0, 98.0),
            signal("trend_following", Direction.SHORT, 80.0, 100.0, 102.0, 97.0),
        ]
        result = self.aggregator().aggregate("X", signals, WEIGHTS, MarketRegime.STRONG_TREND)
        assert result.accepted
        # Stop: mean of the 1% and 2% distances, applied above entry.
        # Target: the nearest of the 2% and 3% distances, applied below entry.
        assert result.signal.stop_loss == pytest.approx(101.5)
        assert result.signal.take_profit == pytest.approx(98.0)
        assert result.signal.stop_loss > result.signal.entry_price
        assert result.signal.take_profit < result.signal.entry_price

    def test_consensus_score_rises_with_agreement(self):
        two = self.aggregator().aggregate(
            "X",
            [
                signal("momentum", Direction.LONG, 80.0),
                signal("trend_following", Direction.LONG, 80.0),
            ],
            WEIGHTS,
            MarketRegime.STRONG_TREND,
        )
        contested = self.aggregator().aggregate(
            "X",
            [
                signal("momentum", Direction.LONG, 80.0),
                signal("trend_following", Direction.LONG, 80.0),
                signal("vwap", Direction.SHORT, 70.0),
            ],
            WEIGHTS,
            MarketRegime.STRONG_TREND,
        )
        assert two.signal.consensus_score > contested.signal.consensus_score

    def test_every_rejection_carries_a_reason_and_detail(self):
        result = self.aggregator().aggregate("X", [], WEIGHTS, MarketRegime.SIDEWAYS)
        assert result.rejection is not None
        assert result.detail


class TestEdgeCalculation:
    def calculator(self, **overrides) -> EdgeCalculator:
        config = EdgeConfig(**overrides)
        return EdgeCalculator(config, CostModel(config))

    def proven(
        self,
        calc: EdgeCalculator,
        name: str = "momentum",
        win_rate: float = 0.55,
        trades: int = 100,
    ) -> None:
        """Give a strategy a real track record so shrinkage does not dominate."""
        for i in range(trades):
            calc.record_result(name, i < int(trades * win_rate), 0.01, 0.001, 0.001)

    def test_costs_are_subtracted_from_gross_expectation(self):
        calc = self.calculator()
        estimate = calc.estimate(aggregated(), LIQUID, notional=600.0)
        assert estimate.expected_net < estimate.expected_gross
        assert estimate.costs.total > 0

    def test_a_tight_scalp_cannot_pay_for_itself(self):
        """0.15% target against a 0.11% round trip is not a trade."""
        calc = self.calculator()
        self.proven(calc)
        decision = calc.evaluate(
            aggregated(stop_pct=0.001, target_pct=0.0015),
            LIQUID,
            600.0,
            strategy="momentum",
        )
        assert not decision.accepted
        assert decision.expected_net < 0

    def test_a_genuinely_good_trade_passes(self):
        """The gate must not be so strict that nothing can ever clear it."""
        calc = self.calculator()
        self.proven(calc)
        decision = calc.evaluate(
            aggregated(stop_pct=0.005, target_pct=0.010),
            LIQUID,
            600.0,
            strategy="momentum",
        )
        assert decision.accepted
        assert decision.expected_net > calc.config.min_expected_edge

    def test_same_trade_is_rejected_on_an_illiquid_symbol(self):
        calc = self.calculator()
        self.proven(calc)
        good = aggregated(stop_pct=0.004, target_pct=0.008)
        liquid = calc.evaluate(good, LIQUID, 600.0, strategy="momentum")
        illiquid = calc.evaluate(good, ILLIQUID, 600.0, strategy="momentum")
        assert liquid.expected_net > illiquid.expected_net

    def test_win_probability_is_shrunk_toward_the_prior(self):
        """Six wins from eight trades is noise, not a 75% win rate."""
        calc = self.calculator()
        for i in range(8):
            calc.record_result("lucky", i < 6, 0.01, 0.001, 0.001)
        assert calc.stats_for("lucky").observed_win_rate == pytest.approx(0.75)
        assert calc.win_probability("lucky") < 0.55

    def test_win_probability_converges_with_evidence(self):
        calc = self.calculator()
        self.proven(calc, "seasoned", win_rate=0.60, trades=400)
        assert calc.win_probability("seasoned") == pytest.approx(0.60, abs=0.03)

    def test_unknown_strategy_uses_the_prior(self):
        calc = self.calculator()
        assert calc.win_probability("brand_new") == pytest.approx(
            calc.config.win_rate_prior, abs=0.01
        )

    def test_ambitious_reward_risk_lowers_the_prior_win_rate(self):
        """A 4:1 target is reached less often than a 1:1 one."""
        calc = self.calculator()
        assert calc.win_probability("new", reward_risk=4.0) < calc.win_probability(
            "new", reward_risk=1.0
        )

    def test_funding_is_charged_only_when_the_trade_straddles_it(self):
        calc = self.calculator()
        short_trade = calc.estimate(
            aggregated(),
            LIQUID,
            600.0,
            funding_rate=0.0005,
            expected_duration_sec=300,
            seconds_to_funding=7200,
        )
        straddles = calc.estimate(
            aggregated(),
            LIQUID,
            600.0,
            funding_rate=0.0005,
            expected_duration_sec=3600,
            seconds_to_funding=60,
        )
        assert short_trade.costs.funding == 0.0
        assert straddles.costs.funding > 0

    def test_a_short_receives_positive_funding(self):
        calc = self.calculator()
        estimate = calc.estimate(
            aggregated(Direction.SHORT),
            LIQUID,
            600.0,
            funding_rate=0.0005,
            expected_duration_sec=3600,
            seconds_to_funding=60,
        )
        assert estimate.costs.funding < 0
        assert (
            estimate.expected_net
            > estimate.expected_gross
            - (
                estimate.costs.entry_fee
                + estimate.costs.exit_fee
                + estimate.costs.spread_cost
                + estimate.costs.slippage
            )
            - 1e-12
        )

    def test_breakeven_win_rate_is_computed_correctly(self):
        calc = self.calculator()
        estimate = calc.estimate(aggregated(stop_pct=0.003, target_pct=0.006), LIQUID, 600.0)
        breakeven = calc.breakeven_win_rate(0.006, 0.003, estimate.costs)
        # (loss + costs) / (win + loss) = (0.003 + 0.0011) / 0.009
        assert breakeven == pytest.approx((0.003 + estimate.costs.total) / 0.009, abs=1e-6)

    def test_realised_versus_expected_gap_is_tracked(self):
        """A persistent gap means the cost model is wrong and must be corrected."""
        calc = self.calculator()
        for _ in range(20):
            calc.record_result("optimistic", True, 0.01, expected_edge=0.002, realised_edge=0.0005)
        report = calc.realised_vs_expected("optimistic")
        assert report["gap"] == pytest.approx(-0.0015, abs=1e-6)

    def test_target_before_stop_not_net_profit_trains_probability(self):
        calc = self.calculator()
        for _ in range(25):
            calc.record_result(
                "momentum",
                won=True,
                gross_return=0.001,
                expected_edge=0.001,
                realised_edge=0.0001,
                target_before_stop=False,
            )
        assert calc.stats_for("momentum").wins == 0

    def test_context_evidence_is_separate_and_conservative(self):
        calc = self.calculator(contextual_min_trades=5, confidence_lower_bound_z=1.0)
        signal = aggregated()
        key = calc.context_key(signal, LIQUID)
        for _ in range(10):
            calc.record_result(
                "momentum",
                True,
                0.01,
                0.001,
                0.001,
                target_before_stop=True,
                context_key=key,
            )
        contextual = calc.contextual_win_probability("momentum", key)
        assert contextual < calc.stats_for("momentum").observed_win_rate
        assert f"context::{key}" in calc.export_stats()

    def test_frozen_evaluation_cannot_learn_or_bootstrap(self):
        calc = self.calculator(bootstrap_enabled=True, bootstrap_min_trades=30)
        calc.disable_bootstrap()
        calc.freeze_learning()
        calc.record_result("momentum", True, 0.01, 0.001, 0.001)
        assert calc.stats_for("momentum").trades == 0
        assert not calc.uses_bootstrap("momentum")

    def test_threshold_is_configurable(self):
        strict = self.calculator(min_expected_edge=0.01)
        self.proven(strict)
        decision = strict.evaluate(
            aggregated(stop_pct=0.005, target_pct=0.010), LIQUID, 600.0, strategy="momentum"
        )
        assert not decision.accepted

    def test_zero_entry_price_yields_a_negative_edge_not_a_crash(self):
        calc = self.calculator()
        broken = aggregated()
        object.__setattr__(broken, "entry_price", 0.0)
        assert calc.estimate(broken, LIQUID, 600.0).expected_net < 0

    def test_required_move_reports_infinity_for_an_impossible_payoff(self):
        from tradebot.core.types import CostEstimate

        costs = CostEstimate(0.0004, 0.0004, 0.0001, 0.0002, 0.0)
        # A 30% win rate on 1:3 reward:risk cannot produce a positive edge.
        assert required_move_for_edge(costs, 0.30, 0.33, 0.0008) == float("inf")

    def test_required_move_is_finite_for_a_viable_payoff(self):
        from tradebot.core.types import CostEstimate

        costs = CostEstimate(0.0004, 0.0004, 0.0001, 0.0002, 0.0)
        move = required_move_for_edge(costs, 0.55, 2.0, 0.0008)
        assert 0 < move < 0.05


class TestOpportunityScoring:
    def scorer(self, **overrides) -> OpportunityScorer:
        return OpportunityScorer(OpportunityConfig(**overrides), ScannerConfig())

    def market(self, total: float = 80.0, volatility: float = 0.005, **components) -> MarketScore:
        base = {"momentum": 70.0, "recent_volume": 70.0, "trend": 70.0, "liquidity": 80.0}
        base.update(components)
        return MarketScore(
            symbol="X",
            total=total,
            components=base,
            penalties={},
            volatility=volatility,
            liquidity_usd=500_000.0,
            spread_bps=1.0,
            funding_rate=0.0001,
            timestamp=0,
        )

    def inputs(self, **overrides) -> OpportunityInputs:
        params = {
            "signal": aggregated(),
            "market": self.market(),
            "liquidity": LIQUID,
            "expected_net_edge": 0.0015,
            "cost_total": 0.0011,
            "correlation": 0.0,
            "notional": 600.0,
        }
        params.update(overrides)
        return OpportunityInputs(**params)

    def test_score_is_bounded(self):
        assert 0.0 <= self.scorer().score(self.inputs()).total <= 100.0

    def test_better_market_scores_higher(self):
        good = self.scorer().score(self.inputs(market=self.market(95.0)))
        poor = self.scorer().score(self.inputs(market=self.market(30.0)))
        assert good.total > poor.total

    def test_correlation_penalises_a_crowded_trade(self):
        """Four correlated longs are one leveraged bet wearing four names."""
        alone = self.scorer().score(self.inputs(correlation=0.0))
        crowded = self.scorer().score(self.inputs(correlation=1.0))
        assert crowded.total < alone.total
        assert crowded.penalties["correlation"] > 0

    def test_high_costs_relative_to_the_move_are_penalised(self):
        cheap = self.scorer().score(self.inputs(cost_total=0.0005))
        dear = self.scorer().score(self.inputs(cost_total=0.008))
        assert dear.total < cheap.total

    def test_wide_spread_lowers_execution_quality(self):
        tight = self.scorer().score(self.inputs(liquidity=LIQUID))
        wide = self.scorer().score(self.inputs(liquidity=ILLIQUID))
        assert wide.components["execution_quality"] < tight.components["execution_quality"]

    def test_consuming_most_of_the_book_lowers_execution_quality(self):
        small = self.scorer().score(self.inputs(notional=100.0))
        huge = self.scorer().score(self.inputs(notional=400_000.0))
        assert huge.components["execution_quality"] < small.components["execution_quality"]

    def test_better_reward_risk_scores_higher(self):
        modest = self.scorer().score(
            self.inputs(signal=aggregated(stop_pct=0.005, target_pct=0.006))
        )
        strong = self.scorer().score(
            self.inputs(signal=aggregated(stop_pct=0.005, target_pct=0.015))
        )
        assert strong.components["risk_reward"] > modest.components["risk_reward"]

    def test_volatility_outside_the_band_is_penalised(self):
        ideal = self.scorer().score(self.inputs(market=self.market(volatility=0.005)))
        wild = self.scorer().score(self.inputs(market=self.market(volatility=0.08)))
        assert wild.components["volatility_fit"] < ideal.components["volatility_fit"]

    def test_grades_follow_the_configured_bands(self):
        from tradebot.core.types import OpportunityScore

        scorer = self.scorer()
        assert scorer.grade(OpportunityScore(95.0, {}, {})) == "EXCEPTIONAL"
        assert scorer.grade(OpportunityScore(85.0, {}, {})) == "STRONG"
        assert scorer.grade(OpportunityScore(72.0, {}, {})) == "MODERATE"
        assert scorer.grade(OpportunityScore(50.0, {}, {})) == "REJECT"

    def test_threshold_is_enforced(self):
        from tradebot.core.types import OpportunityScore

        scorer = self.scorer(min_score=70.0)
        assert scorer.accepts(OpportunityScore(70.0, {}, {}))
        assert not scorer.accepts(OpportunityScore(69.9, {}, {}))
