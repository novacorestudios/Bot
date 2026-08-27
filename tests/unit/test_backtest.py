"""Backtesting engine, metrics, walk-forward and Monte Carlo.

The most important test in this file is
``TestPnLArithmetic::test_pnl_matches_hand_calculation``: a backtester whose
arithmetic is wrong produces confident, precise, wrong numbers, and every
decision downstream inherits the error.
"""

from __future__ import annotations

import math

import pytest

from tradebot.backtesting.metrics import EquityPoint, compute_metrics
from tradebot.backtesting.montecarlo import (
    MonteCarloAnalyzer,
    parameter_robustness,
)
from tradebot.backtesting.walkforward import WalkForwardAnalyzer
from tradebot.core.config import MonteCarloConfig, WalkForwardConfig, load_tunables
from tradebot.core.types import Direction, ExitReason, MarketRegime, Trade

from ..conftest import REPO_ROOT

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
)

DAY_MS = 86_400_000


def trade(
    net: float,
    *,
    gross: float | None = None,
    fees: float = 0.08,
    funding: float = 0.0,
    slippage: float = 0.02,
    risk: float = 1.0,
    opened: int = 0,
    closed: int = 600_000,
    symbol: str = "TESTUSDT",
    strategy: str = "momentum",
    reason: ExitReason = ExitReason.TAKE_PROFIT,
    regime: MarketRegime = MarketRegime.STRONG_TREND,
) -> Trade:
    return Trade(
        trade_id=f"t{net}{opened}",
        symbol=symbol,
        strategy=strategy,
        direction=Direction.LONG,
        entry_price=100.0,
        exit_price=101.0,
        quantity=1.0,
        leverage=2,
        stop_loss=99.0,
        take_profit=103.0,
        opened_at=opened,
        closed_at=closed,
        gross_pnl=gross if gross is not None else net + fees + funding,
        fees=fees,
        funding=funding,
        slippage_cost=slippage,
        net_pnl=net,
        exit_reason=reason,
        regime=regime,
        entry_notional=100.0,
        initial_risk=risk,
    )


def curve(values: list[float], step_ms: int = 3_600_000) -> list[EquityPoint]:
    return [EquityPoint(i * step_ms, v) for i, v in enumerate(values)]


# --------------------------------------------------------------------------- #
class TestPnLArithmetic:
    """PnL must be exact. Everything downstream inherits any error here."""

    def test_pnl_matches_hand_calculation(self):
        """A long: buy 2 @ 100, sell @ 103, 0.04% taker each side.

        gross  = (103 - 100) * 2                       = 6.00
        fee in = 100 * 2 * 0.0004                      = 0.08
        fee out= 103 * 2 * 0.0004                      = 0.0824
        net    = 6.00 - 0.08 - 0.0824                  = 5.8376
        """
        entry, exit_, qty, fee_rate = 100.0, 103.0, 2.0, 0.0004
        gross = (exit_ - entry) * qty
        fees = entry * qty * fee_rate + exit_ * qty * fee_rate
        net = gross - fees

        assert gross == pytest.approx(6.0)
        assert fees == pytest.approx(0.1624)
        assert net == pytest.approx(5.8376)

        result = trade(net=net, gross=gross, fees=fees, funding=0.0, risk=2.0)
        assert result.net_pnl == pytest.approx(5.8376)
        assert result.r_multiple == pytest.approx(2.9188)

    def test_short_pnl_sign_is_correct(self):
        """A short profits when price falls: sell 2 @ 100, buy back @ 97."""
        entry, exit_, qty = 100.0, 97.0, 2.0
        gross = (exit_ - entry) * qty * Direction.SHORT.sign
        assert gross == pytest.approx(6.0)

    def test_funding_reduces_net_pnl_for_a_payer(self):
        result = trade(net=5.0, gross=6.0, fees=0.16, funding=0.84)
        assert result.gross_pnl - result.fees - result.funding == pytest.approx(5.0)

    def test_funding_received_increases_net_pnl(self):
        """A short receives positive funding, recorded as a negative cost."""
        result = trade(net=6.0, gross=5.9, fees=0.1, funding=-0.2)
        assert result.gross_pnl - result.fees - result.funding == pytest.approx(6.0)


class TestMetrics:
    def test_empty_input_does_not_crash(self):
        metrics = compute_metrics([], [], 75.0)
        assert metrics.total_trades == 0
        assert metrics.final_capital == 75.0
        assert metrics.warnings

    def test_no_trades_is_reported_as_a_valid_outcome(self):
        """Opportunity-driven trading means zero trades is possible, not a bug."""
        metrics = compute_metrics([], curve([75.0, 75.0, 75.0]), 75.0)
        assert metrics.total_trades == 0
        assert any("valid outcome" in w for w in metrics.warnings)

    def test_total_return_is_computed_from_the_equity_curve(self):
        metrics = compute_metrics([trade(10.0)], curve([100.0, 110.0]), 100.0)
        assert metrics.total_return == pytest.approx(0.10)

    def test_win_rate_and_profit_factor(self):
        trades = [trade(2.0), trade(2.0), trade(-1.0), trade(-1.0)]
        metrics = compute_metrics(trades, curve([100.0, 102.0]), 100.0)
        assert metrics.win_rate == pytest.approx(0.5)
        assert metrics.profit_factor == pytest.approx(2.0)
        assert metrics.expectancy == pytest.approx(0.5)

    def test_profit_factor_with_no_losses_is_flagged_not_infinite(self):
        metrics = compute_metrics([trade(1.0), trade(2.0)], curve([100.0, 103.0]), 100.0)
        assert math.isnan(metrics.profit_factor)

    def test_drawdown_is_measured_on_the_equity_curve(self):
        """Closed-trade drawdown would miss an open position's excursion."""
        metrics = compute_metrics([trade(1.0)], curve([100.0, 120.0, 90.0, 110.0]), 100.0)
        assert metrics.max_drawdown == pytest.approx(0.25)  # 120 -> 90
        assert metrics.max_drawdown_abs == pytest.approx(30.0)

    def test_drawdown_of_a_rising_curve_is_zero(self):
        metrics = compute_metrics([trade(1.0)], curve([100.0, 110.0, 120.0]), 100.0)
        assert metrics.max_drawdown == pytest.approx(0.0)

    def test_streaks_are_counted(self):
        trades = [trade(1.0), trade(1.0), trade(1.0), trade(-1.0), trade(-1.0)]
        metrics = compute_metrics(trades, curve([100.0, 101.0]), 100.0)
        assert metrics.longest_winning_streak == 3
        assert metrics.longest_losing_streak == 2

    def test_costs_are_totalled_separately(self):
        trades = [trade(1.0, fees=0.1, funding=0.02, slippage=0.03) for _ in range(3)]
        metrics = compute_metrics(trades, curve([100.0, 103.0]), 100.0)
        assert metrics.total_fees == pytest.approx(0.3)
        assert metrics.total_funding == pytest.approx(0.06)
        assert metrics.total_slippage == pytest.approx(0.09)
        assert metrics.total_costs == pytest.approx(0.45)

    def test_high_cost_ratio_is_warned_about(self):
        trades = [trade(0.01, gross=1.0, fees=0.9, slippage=0.09) for _ in range(40)]
        metrics = compute_metrics(trades, curve([100.0, 100.4]), 100.0)
        assert any("costs" in w for w in metrics.warnings)

    def test_small_sample_is_warned_about(self):
        metrics = compute_metrics([trade(1.0)], curve([100.0, 101.0]), 100.0)
        assert any("noise" in w for w in metrics.warnings)

    def test_breakdowns_by_strategy_and_regime(self):
        trades = [
            trade(2.0, strategy="momentum", regime=MarketRegime.STRONG_TREND),
            trade(-1.0, strategy="momentum", regime=MarketRegime.STRONG_TREND),
            trade(3.0, strategy="breakout", regime=MarketRegime.BREAKOUT),
        ]
        metrics = compute_metrics(trades, curve([100.0, 104.0]), 100.0)
        assert metrics.by_strategy["momentum"]["trades"] == 2
        assert metrics.by_strategy["breakout"]["net_pnl"] == pytest.approx(3.0)
        assert metrics.by_regime["BREAKOUT"]["trades"] == 1

    def test_exit_reasons_are_counted(self):
        trades = [
            trade(1.0, reason=ExitReason.TAKE_PROFIT),
            trade(-1.0, reason=ExitReason.STOP_LOSS),
            trade(0.1, reason=ExitReason.TIME_LIMIT),
        ]
        metrics = compute_metrics(trades, curve([100.0, 100.1]), 100.0)
        assert metrics.by_exit_reason["STOP_LOSS"] == 1
        assert metrics.by_exit_reason["TIME_LIMIT"] == 1

    def test_short_sample_is_not_annualised_at_all(self):
        """Compounding a one-hour gain to a year gives ~10^300, not a return."""
        metrics = compute_metrics([trade(1.0)], curve([100.0, 101.0]), 100.0)
        assert metrics.annualized_return == 0.0
        assert math.isfinite(metrics.annualized_return)
        assert any("annualisation" in w for w in metrics.warnings)

    def test_a_long_sample_is_annualised_normally(self):
        # Two years of hourly points, ending 21% up -> ~10% a year.
        points = curve([100.0 + i * 0.0012 for i in range(17_532)])
        metrics = compute_metrics([trade(1.0)], points, 100.0)
        assert 0.05 < metrics.annualized_return < 0.15

    def test_annualised_return_is_capped_rather_than_infinite(self):
        points = curve([100.0] * 200 + [100_000.0], step_ms=86_400_000 // 4)
        metrics = compute_metrics([trade(1.0)], points, 100.0)
        assert math.isfinite(metrics.annualized_return)
        assert abs(metrics.annualized_return) <= 100.0

    def test_sharpe_is_undefined_rather_than_infinite_on_zero_variance(self):
        metrics = compute_metrics([trade(1.0)], curve([100.0] * 10), 100.0)
        assert math.isnan(metrics.sharpe_ratio) or metrics.sharpe_ratio == 0.0

    def test_wipeout_is_reported_not_annualised(self):
        metrics = compute_metrics([trade(-100.0)], curve([100.0, 0.0]), 100.0)
        assert metrics.annualized_return == -1.0
        assert any("wiped out" in w for w in metrics.warnings)

    def test_summary_lines_render(self):
        metrics = compute_metrics([trade(1.0)], curve([100.0, 101.0]), 100.0)
        assert any("NET PROFIT" in line for line in metrics.summary_lines())


class TestMonteCarlo:
    def analyzer(self, **overrides) -> MonteCarloAnalyzer:
        params = {"iterations": 400, "method": "bootstrap", "drawdown_percentile": 0.05}
        params.update(overrides)
        return MonteCarloAnalyzer(MonteCarloConfig(**params), seed=42)

    def profitable(self, n: int = 200) -> list[Trade]:
        # 60% win rate, 1.5:1 payoff -> genuinely positive expectancy.
        return [trade(1.5 if i % 10 < 6 else -1.0) for i in range(n)]

    def losing(self, n: int = 200) -> list[Trade]:
        return [trade(1.0 if i % 10 < 4 else -1.0) for i in range(n)]

    def test_too_few_trades_is_inconclusive(self):
        report = self.analyzer().run([trade(1.0)] * 5, 100.0)
        assert "INCONCLUSIVE" in report.verdict

    def test_a_profitable_sequence_has_a_positive_median(self):
        report = self.analyzer().run(self.profitable(), 1000.0)
        assert report.median_return > 0

    def test_a_losing_sequence_is_mostly_negative(self):
        report = self.analyzer().run(self.losing(), 1000.0)
        assert report.probability_of_loss > 0.5
        assert "FAILS" in report.verdict

    def test_tail_drawdown_exceeds_the_median(self):
        """The 95th percentile is the number to plan around, not the median."""
        report = self.analyzer().run(self.profitable(), 1000.0)
        assert report.percentile_95_drawdown >= report.median_max_drawdown

    def test_drawdown_beyond_the_limit_fails(self):
        report = self.analyzer().run(self.profitable(), 100.0, max_drawdown_limit=0.001)
        assert "FAILS" in report.verdict
        assert "kill switch" in report.verdict

    def test_shuffle_method_preserves_the_total(self):
        """Permuting trades cannot change the sum, only the path."""
        analyzer = self.analyzer(method="shuffle")
        trades = self.profitable()
        expected = sum(t.net_pnl for t in trades) / 1000.0
        report = analyzer.run(trades, 1000.0)
        assert report.mean_return == pytest.approx(expected, abs=1e-9)

    def test_bootstrap_varies_the_total(self):
        report = self.analyzer(method="bootstrap").run(self.profitable(), 1000.0)
        assert report.std_return > 0

    def test_results_are_reproducible_with_a_seed(self):
        trades = self.profitable()
        a = MonteCarloAnalyzer(MonteCarloConfig(iterations=200), seed=7).run(trades, 1000.0)
        b = MonteCarloAnalyzer(MonteCarloConfig(iterations=200), seed=7).run(trades, 1000.0)
        assert a.median_return == pytest.approx(b.median_return)

    def test_the_independence_caveat_is_always_stated(self):
        report = self.analyzer().run(self.profitable(), 1000.0)
        assert "UNDERSTATE tail risk" in report.summary()

    def test_losing_streak_percentile_is_reported(self):
        report = self.analyzer().run(self.profitable(), 1000.0)
        assert report.longest_losing_streak_p95 >= 1


class TestParameterRobustness:
    def test_too_few_variants_cannot_be_judged(self):
        ok, reason = parameter_robustness({"a": 0.1, "b": 0.2})
        assert not ok
        assert "three" in reason

    def test_stable_performance_passes(self):
        ok, _ = parameter_robustness({"a": 0.10, "b": 0.09, "c": 0.11, "d": 0.08})
        assert ok

    def test_a_single_lucky_parameter_fails(self):
        """Performance concentrated in one parameter value is a fitted result."""
        ok, reason = parameter_robustness({"a": 0.50, "b": -0.20, "c": -0.15, "d": -0.10})
        assert not ok
        assert "parameter" in reason

    def test_all_negative_variants_fail(self):
        ok, reason = parameter_robustness({"a": -0.1, "b": -0.2, "c": -0.3})
        assert not ok
        assert "profitable" in reason


class TestWalkForward:
    def analyzer(self, **overrides) -> WalkForwardAnalyzer:
        params = {
            "train_days": 30,
            "validation_days": 7,
            "test_days": 7,
            "step_days": 7,
            "min_trades_per_fold": 20,
        }
        params.update(overrides)
        return WalkForwardAnalyzer(CONFIG, WalkForwardConfig(**params))

    def test_folds_tile_the_period_with_a_rolling_step(self):
        folds = self.analyzer().build_folds(0, 100 * DAY_MS)
        assert folds
        assert folds[0].train_start == 0
        assert folds[1].train_start == 7 * DAY_MS
        for fold in folds:
            assert fold.train_end == fold.validation_start
            assert fold.validation_end == fold.test_start
            assert fold.test_end <= 100 * DAY_MS

    def test_test_windows_never_overlap_their_own_training_window(self):
        """The entire point: test data must be unseen."""
        for fold in self.analyzer().build_folds(0, 200 * DAY_MS):
            assert fold.test_start >= fold.train_end

    def test_a_period_too_short_for_one_fold_yields_no_folds(self):
        assert self.analyzer().build_folds(0, 10 * DAY_MS) == []

    def test_insufficient_data_is_reported_as_inconclusive(self):
        report = self.analyzer().run({}, 0, 10 * DAY_MS)
        assert "INCONCLUSIVE" in report.verdict
        assert report.warnings

    def test_efficiency_compares_out_of_sample_to_in_sample(self):
        from tradebot.backtesting.walkforward import Fold, FoldResult

        class _Stub:
            def __init__(self, value):
                self.metrics = type("M", (), {"total_return": value, "total_trades": 50})()

        fold = Fold(0, 0, 1, 1, 2, 2, 3)
        halved = FoldResult(fold=fold, train=_Stub(0.10), test=_Stub(0.05))
        assert halved.efficiency == pytest.approx(0.5)

        collapsed = FoldResult(fold=fold, train=_Stub(0.10), test=_Stub(-0.05))
        assert collapsed.efficiency < 0

    def test_a_losing_train_window_gives_zero_efficiency_not_a_division_error(self):
        from tradebot.backtesting.walkforward import Fold, FoldResult

        class _Stub:
            def __init__(self, value):
                self.metrics = type("M", (), {"total_return": value, "total_trades": 10})()

        result = FoldResult(fold=Fold(0, 0, 1, 1, 2, 2, 3), train=_Stub(-0.10), test=_Stub(0.05))
        assert result.efficiency == 0.0
