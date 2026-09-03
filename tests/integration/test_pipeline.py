"""The analytical pipeline end to end: strategies → consensus → score → edge.

Two properties matter most here:

* **Every rejection is explained.** A bot that silently declines everything is
  indistinguishable from a broken one, so the pipeline must always say which
  stage refused and why.
* **The gates compose.** Passing the strategies does not mean passing consensus;
  passing consensus does not mean clearing the edge filter.
"""

from __future__ import annotations

import pytest

from tradebot.core.config import load_tunables
from tradebot.core.types import Direction, MarketRegime, MarketScore, RejectionReason
from tradebot.market.candles import CandleStore
from tradebot.market.microstructure import CostModel, LiquiditySnapshot
from tradebot.signals.pipeline import SignalPipeline
from tradebot.strategies.base import MarketView
from tradebot.strategies.registry import StrategyRegistry

from ..conftest import (
    REPO_ROOT,
    choppy_prices,
    flat_prices,
    impulse_prices,
    make_candles,
    trending_with_pullbacks,
)

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
)

LIQUID = LiquiditySnapshot("TESTUSDT", 1.0, 500_000.0, 500_000.0, 0.1, 1e9)
ILLIQUID = LiquiditySnapshot("TESTUSDT", 6.0, 15_000.0, 15_000.0, 0.0, 3e7)


def build_pipeline(config=CONFIG) -> SignalPipeline:
    standard = config.model_copy(
        update={"trade": config.trade.model_copy(update={"raw_signal_mode": False})}
    )
    return SignalPipeline(
        standard, StrategyRegistry.from_config(standard), CostModel(standard.edge)
    )


def market_score(total: float = 85.0, volatility: float = 0.005) -> MarketScore:
    return MarketScore(
        symbol="TESTUSDT",
        total=total,
        components={"momentum": 75.0, "recent_volume": 75.0, "trend": 80.0, "liquidity": 85.0},
        penalties={},
        volatility=volatility,
        liquidity_usd=500_000.0,
        spread_bps=1.0,
        funding_rate=0.0001,
        timestamp=0,
    )


def build_view(
    prices_by_timeframe: dict[str, list[float]],
    regime: MarketRegime,
    volumes_by_timeframe=None,
    symbol: str = "TESTUSDT",
) -> MarketView:
    store = CandleStore(500)
    for timeframe, prices in prices_by_timeframe.items():
        volumes = (volumes_by_timeframe or {}).get(timeframe)
        store.series(symbol, timeframe).extend(make_candles(prices, volumes=volumes))
    return MarketView(
        symbol=symbol,
        candles=store,
        regime=regime,
        regime_confidence=85.0,
        spread_bps=1.0,
        book_imbalance=0.1,
        funding_rate=0.0001,
        now_ms=1_700_000_000_000,
    )


def trending_view() -> MarketView:
    """A market several trend strategies should agree on."""
    fast = impulse_prices(220, direction=1, seed=3)
    slow = trending_with_pullbacks(320, drift=0.0006, seed=3)
    volumes = {"3m": [1000.0] * 220 + [2400.0] * (len(fast) - 220)}
    return build_view(
        {"1m": fast, "3m": fast, "5m": slow, "15m": slow, "1h": slow},
        MarketRegime.STRONG_TREND,
        volumes,
    )


class TestPipelineGates:
    def test_raw_mode_uses_one_signal_and_requested_exit_band(self):
        config = CONFIG.model_copy(
            update={"trade": CONFIG.trade.model_copy(update={"raw_signal_mode": True})}
        )
        pipeline = SignalPipeline(
            config, StrategyRegistry.from_config(config), CostModel(config.edge)
        )
        result = pipeline.evaluate(trending_view(), market_score(95.0), LIQUID, 25.0)
        assert result.accepted
        signal = result.opportunity.signal
        assert signal.metadata["raw_signal_mode"] is True
        assert signal.stop_distance / signal.entry_price == pytest.approx(0.05)
        target_pct = abs(signal.take_profit - signal.entry_price) / signal.entry_price
        assert 0.0005 <= target_pct <= 0.01

    def test_panic_regime_blocks_before_any_strategy_runs(self):
        pipeline = build_pipeline()
        view = build_view(
            {"3m": impulse_prices(220), "5m": impulse_prices(220)}, MarketRegime.PANIC
        )
        result = pipeline.evaluate(view, market_score(), LIQUID, 600.0)
        assert not result.accepted
        assert result.rejection is RejectionReason.REGIME_BLOCKED
        assert result.stage == "regime"

    def test_flat_market_is_rejected_with_no_signal(self):
        pipeline = build_pipeline()
        flat = flat_prices(320)
        view = build_view(
            {"1m": flat, "3m": flat, "5m": flat, "15m": flat, "1h": flat}, MarketRegime.STRONG_TREND
        )
        result = pipeline.evaluate(view, market_score(), LIQUID, 600.0)
        assert not result.accepted
        assert result.rejection in {
            RejectionReason.NO_SIGNAL,
            RejectionReason.LOW_CONFIDENCE,
            RejectionReason.INSUFFICIENT_CONSENSUS,
        }

    def test_every_rejection_names_its_stage_and_reason(self):
        """'Why didn't it trade?' must always be answerable."""
        pipeline = build_pipeline()
        for prices, regime in (
            (flat_prices(320), MarketRegime.STRONG_TREND),
            (choppy_prices(320, seed=17), MarketRegime.SIDEWAYS),
            (impulse_prices(220), MarketRegime.PANIC),
        ):
            view = build_view(
                {"1m": prices, "3m": prices, "5m": prices, "15m": prices, "1h": prices}, regime
            )
            result = pipeline.evaluate(view, market_score(), LIQUID, 600.0)
            if not result.accepted:
                assert result.rejection is not None
                assert result.detail
                assert result.stage

    def test_a_strong_setup_reaches_the_edge_filter(self):
        """It need not pass — but it must be REJECTED FOR COST, not for signal."""
        pipeline = build_pipeline()
        result = pipeline.evaluate(trending_view(), market_score(95.0), LIQUID, 600.0)
        assert result.stage in {"complete", "edge", "opportunity"}
        if not result.accepted:
            assert result.rejection in {
                RejectionReason.NEGATIVE_EXPECTED_EDGE,
                RejectionReason.LOW_OPPORTUNITY_SCORE,
            }

    def test_illiquidity_can_reject_an_otherwise_good_setup(self):
        pipeline = build_pipeline()
        view = trending_view()
        liquid = pipeline.evaluate(view, market_score(95.0), LIQUID, 600.0)
        illiquid = pipeline.evaluate(view, market_score(95.0), ILLIQUID, 600.0)

        def edge_of(result):
            if result.accepted:
                return result.opportunity.expected_net_edge
            return result.audit.get("expected_net_edge", -1.0)

        assert edge_of(illiquid) < edge_of(liquid)

    def test_correlation_penalty_can_reject_a_crowded_trade(self):
        pipeline = build_pipeline()
        view = trending_view()
        alone = pipeline.evaluate(view, market_score(95.0), LIQUID, 600.0, correlation=0.0)
        crowded = pipeline.evaluate(view, market_score(95.0), LIQUID, 600.0, correlation=1.0)

        def score_of(result):
            if result.accepted:
                return result.opportunity.opportunity_score.total
            return result.audit.get("opportunity_score", 0.0)

        assert score_of(crowded) <= score_of(alone)


def pipeline_with_history(win_rate: float, trades: int = 200) -> SignalPipeline:
    """A pipeline whose strategies have a realised track record.

    Without history the edge calculator shrinks every win probability to the
    prior, so nothing can clear the gate — which is correct behaviour for an
    unproven strategy, and inconvenient for testing the accepted path.
    """
    pipeline = build_pipeline()
    for name in ("momentum", "trend_following", "breakout", "vwap"):
        for i in range(trades):
            pipeline.edge_calculator.record_result(
                name, (i % 100) < win_rate * 100, 0.01, 0.001, 0.001
            )
    return pipeline


class TestRequiredEdge:
    """What the shipped defaults actually demand of a strategy."""

    @pytest.mark.parametrize(
        ("win_rate", "expected"),
        [(0.50, False), (0.58, False), (0.65, True), (0.70, True)],
    )
    def test_acceptance_depends_on_a_real_track_record(self, win_rate, expected):
        pipeline = pipeline_with_history(win_rate)
        result = pipeline.evaluate(trending_view(), market_score(95.0), LIQUID, 600.0)
        assert result.accepted is expected

    def test_an_unproven_strategy_cannot_clear_the_gate(self):
        """Shrinkage toward the prior means a new strategy must earn its size."""
        result = build_pipeline().evaluate(trending_view(), market_score(95.0), LIQUID, 600.0)
        assert not result.accepted


class TestAcceptedOpportunity:
    def accepted_result(self):
        pipeline = pipeline_with_history(0.65)
        result = pipeline.evaluate(trending_view(), market_score(95.0), LIQUID, 600.0)
        assert result.accepted, f"expected acceptance, got: {result.detail}"
        return pipeline, result

    def test_an_accepted_opportunity_carries_a_full_audit_record(self):
        """The brief's 'why did the bot enter?' requirement."""
        _, result = self.accepted_result()

        required = {
            "regime",
            "direction",
            "consensus_score",
            "confidence",
            "agreeing_strategies",
            "opposing_strategies",
            "reason_codes",
            "opportunity_score",
            "opportunity_components",
            "expected_net_edge",
            "win_probability",
            "costs",
            "entry",
            "stop_loss",
            "take_profit",
            "risk_reward",
            "spread_bps",
            "funding_rate",
        }
        assert required <= set(result.audit)

    def test_an_accepted_opportunity_has_sound_geometry(self):
        _, result = self.accepted_result()
        opportunity = result.opportunity
        signal = opportunity.signal
        assert signal.stop_distance > 0
        if signal.direction is Direction.LONG:
            assert signal.stop_loss < signal.entry_price < signal.take_profit
        else:
            assert signal.take_profit < signal.entry_price < signal.stop_loss
        assert opportunity.expected_net_edge > CONFIG.edge.min_expected_edge
        assert opportunity.strategy in signal.strategies

    def test_pipeline_statistics_are_recorded(self):
        pipeline, _ = self.accepted_result()
        stats = pipeline.stats()
        assert stats["evaluated"] >= 1
        assert "rejections" in stats
        assert 0.0 <= stats["acceptance_rate"] <= 1.0


class TestNoForcedTrading:
    """The brief's hardest constraint: zero opportunities means zero trades."""

    def test_a_dead_market_produces_no_opportunity_however_often_it_is_polled(self):
        pipeline = build_pipeline()
        flat = flat_prices(320)
        view = build_view(
            {"1m": flat, "3m": flat, "5m": flat, "15m": flat, "1h": flat},
            MarketRegime.LOW_VOLATILITY,
        )
        accepted = sum(
            pipeline.evaluate(view, market_score(40.0), LIQUID, 600.0).accepted for _ in range(50)
        )
        assert accepted == 0

    def test_a_high_market_score_alone_does_not_produce_a_trade(self):
        """Ranking first in the scanner is not a reason to enter."""
        pipeline = build_pipeline()
        flat = flat_prices(320)
        view = build_view(
            {"1m": flat, "3m": flat, "5m": flat, "15m": flat, "1h": flat}, MarketRegime.STRONG_TREND
        )
        result = pipeline.evaluate(view, market_score(99.0), LIQUID, 600.0)
        assert not result.accepted
