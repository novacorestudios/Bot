"""Strategy x regime and symbol x strategy performance matrices.

The question the allocator alone cannot answer: mean reversion in a strong trend
and momentum in chop are not weak strategies, they are strategies in the wrong
conditions — and an aggregate win rate hides both facts.

The governing constraint, asserted repeatedly: a matrix may only ever say
"do less of this". A cell that looks excellent on twelve trades is far more
likely to be luck than skill, and betting up on it is how a small account turns
a good run into a drawdown.
"""

from __future__ import annotations

import inspect

import pytest

from tradebot.core.config import MatricesConfig
from tradebot.core.types import MarketRegime
from tradebot.risk import matrices as matrices_module
from tradebot.risk.matrices import MatrixSet, PerformanceMatrix


@pytest.fixture
def matrix() -> PerformanceMatrix:
    return PerformanceMatrix("test", min_trades=10, floor_expectancy_r=-0.25)


def fill(matrix: PerformanceMatrix, row: str, column: str, r: float, count: int) -> None:
    for _ in range(count):
        matrix.record(row, column, won=r > 0, r_multiple=r, pnl=r * 0.375)


class TestEvidenceThreshold:
    def test_an_unseen_cell_has_no_opinion(self, matrix: PerformanceMatrix) -> None:
        """No evidence is not evidence of a problem."""
        assert matrix.multiplier("momentum", "STRONG_TREND") == 1.0
        assert matrix.has_evidence("momentum", "STRONG_TREND") is False
        assert matrix.blocked("momentum", "STRONG_TREND") is False

    def test_a_thin_cell_has_no_opinion_either(self, matrix: PerformanceMatrix) -> None:
        """Three losing trades in a cell is noise; suppressing on it is curve
        fitting against your own history."""
        fill(matrix, "mean_reversion", "STRONG_TREND", r=-1.0, count=3)
        assert matrix.has_evidence("mean_reversion", "STRONG_TREND") is False
        assert matrix.multiplier("mean_reversion", "STRONG_TREND") == 1.0

    def test_evidence_arrives_at_the_threshold(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "mean_reversion", "STRONG_TREND", r=-1.0, count=10)
        assert matrix.has_evidence("mean_reversion", "STRONG_TREND") is True


class TestMultipliers:
    def test_a_profitable_cell_is_left_alone(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "momentum", "STRONG_TREND", r=1.4, count=20)
        assert matrix.multiplier("momentum", "STRONG_TREND") == 1.0

    def test_an_exceptional_cell_is_still_only_left_alone(self, matrix: PerformanceMatrix) -> None:
        """Never above 1.0, however good the record looks."""
        fill(matrix, "momentum", "STRONG_TREND", r=8.0, count=50)
        assert matrix.multiplier("momentum", "STRONG_TREND") == 1.0

    def test_a_losing_cell_is_suppressed_proportionally(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "vwap", "HIGH_VOLATILITY", r=-0.10, count=15)
        multiplier = matrix.multiplier("vwap", "HIGH_VOLATILITY")
        assert 0.0 < multiplier < 1.0

    def test_a_badly_losing_cell_is_suppressed_entirely(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "mean_reversion", "STRONG_TREND", r=-0.9, count=15)
        assert matrix.multiplier("mean_reversion", "STRONG_TREND") == 0.0
        assert matrix.blocked("mean_reversion", "STRONG_TREND") is True

    def test_suppression_deepens_as_expectancy_falls(self, matrix: PerformanceMatrix) -> None:
        mild = PerformanceMatrix("mild", min_trades=10)
        harsh = PerformanceMatrix("harsh", min_trades=10)
        fill(mild, "s", "r", r=-0.05, count=12)
        fill(harsh, "s", "r", r=-0.20, count=12)
        assert mild.multiplier("s", "r") > harsh.multiplier("s", "r")

    def test_breaking_even_is_not_suppressed(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "vwap", "SIDEWAYS", r=0.0, count=20)
        assert matrix.multiplier("vwap", "SIDEWAYS") == 1.0


class TestCellStatistics:
    def test_win_rate_and_expectancy(self, matrix: PerformanceMatrix) -> None:
        for r in (2.0, 2.0, -1.0, -1.0):
            matrix.record("momentum", "STRONG_TREND", won=r > 0, r_multiple=r)
        cell = matrix.cell("momentum", "STRONG_TREND")
        assert cell is not None
        assert cell.trades == 4
        assert cell.win_rate == pytest.approx(0.5)
        assert cell.expectancy_r == pytest.approx(0.5)

    def test_best_and_worst_are_tracked(self, matrix: PerformanceMatrix) -> None:
        for r in (0.5, 3.2, -1.0):
            matrix.record("momentum", "BREAKOUT", won=r > 0, r_multiple=r)
        cell = matrix.cell("momentum", "BREAKOUT")
        assert cell is not None
        assert cell.best_r == 3.2
        assert cell.worst_r == -1.0

    def test_a_single_trade_sets_both_extremes(self, matrix: PerformanceMatrix) -> None:
        """The seeded-max bug: starting from 0.0 would report a best of 0.0 for
        a cell whose only trade lost."""
        matrix.record("momentum", "PANIC", won=False, r_multiple=-1.0)
        cell = matrix.cell("momentum", "PANIC")
        assert cell is not None
        assert cell.best_r == -1.0
        assert cell.worst_r == -1.0


class TestReporting:
    def test_the_table_lists_every_recorded_cell(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "momentum", "STRONG_TREND", r=1.0, count=12)
        fill(matrix, "momentum", "SIDEWAYS", r=-0.5, count=12)
        table = matrix.as_table()
        assert set(table["momentum"]) == {"STRONG_TREND", "SIDEWAYS"}
        assert table["momentum"]["STRONG_TREND"]["multiplier"] == 1.0
        assert table["momentum"]["SIDEWAYS"]["sufficient_evidence"] is True

    def test_worst_cells_only_include_ones_with_evidence(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "vwap", "PANIC", r=-3.0, count=2)  # terrible, but thin
        fill(matrix, "vwap", "SIDEWAYS", r=-0.4, count=12)  # bad, with evidence
        worst = matrix.worst_cells()
        assert [(w["row"], w["column"]) for w in worst] == [("vwap", "SIDEWAYS")]

    def test_rows_and_columns(self, matrix: PerformanceMatrix) -> None:
        matrix.record("a", "x", True, 1.0)
        matrix.record("b", "y", True, 1.0)
        assert matrix.rows() == ["a", "b"]
        assert matrix.columns() == ["x", "y"]

    def test_stats(self, matrix: PerformanceMatrix) -> None:
        fill(matrix, "momentum", "STRONG_TREND", r=1.0, count=12)
        fill(matrix, "vwap", "PANIC", r=1.0, count=2)
        stats = matrix.stats()
        assert stats["cells"] == 2
        assert stats["cells_with_evidence"] == 1
        assert stats["total_trades"] == 14


ENABLED = MatricesConfig(
    feedback_enabled=True, strategy_regime_min_trades=10, symbol_strategy_min_trades=10
)


class TestFeedbackIsOptIn:
    """The plan's rule: the tables diagnose by default, they do not select.

    Suppressing a combination on a handful of trades is overfitting against your
    own history, and on a 75 USDT account a cell takes weeks to fill.
    """

    def test_the_shipped_default_has_no_influence_on_sizing(self) -> None:
        assert MatricesConfig().feedback_enabled is False
        matrices = MatrixSet(MatricesConfig())
        for _ in range(200):
            matrices.record("vwap", MarketRegime.PANIC, "BADUSDT", False, -3.0)
        assert matrices.multiplier("vwap", MarketRegime.PANIC, "BADUSDT") == 1.0
        assert matrices.blocked("vwap", MarketRegime.PANIC, "BADUSDT") is False

    def test_recording_still_happens_so_the_tables_can_be_read(self) -> None:
        matrices = MatrixSet(MatricesConfig())
        matrices.record("vwap", MarketRegime.PANIC, "BADUSDT", False, -3.0)
        cell = matrices.strategy_regime.cell("vwap", "PANIC")
        assert cell is not None and cell.trades == 1

    def test_enabling_it_makes_the_evidence_bite(self) -> None:
        matrices = MatrixSet(ENABLED)
        for _ in range(20):
            matrices.record("vwap", MarketRegime.PANIC, "BADUSDT", False, -3.0)
        assert matrices.blocked("vwap", MarketRegime.PANIC, "BADUSDT") is True

    def test_the_default_thresholds_are_not_a_handful_of_trades(self) -> None:
        config = MatricesConfig()
        assert config.strategy_regime_min_trades >= 25
        assert config.symbol_strategy_min_trades >= 20


class TestMatrixSet:
    def test_a_trade_lands_in_both_matrices(self) -> None:
        matrices = MatrixSet(ENABLED)
        matrices.record("momentum", MarketRegime.STRONG_TREND, "BTCUSDT", won=True, r_multiple=1.5)
        assert matrices.strategy_regime.cell("momentum", "STRONG_TREND") is not None
        assert matrices.symbol_strategy.cell("BTCUSDT", "momentum") is not None

    def test_it_accepts_a_regime_enum_or_its_name(self) -> None:
        matrices = MatrixSet(ENABLED)
        matrices.record("momentum", MarketRegime.BREAKOUT, "BTCUSDT", True, 1.0)
        matrices.record("momentum", "BREAKOUT", "ETHUSDT", True, 1.0)
        cell = matrices.strategy_regime.cell("momentum", "BREAKOUT")
        assert cell is not None
        assert cell.trades == 2

    def test_the_two_penalties_compound(self) -> None:
        """Bad in this regime AND bad on this symbol is worse than either
        alone; averaging would let one good half excuse the other."""
        matrices = MatrixSet(ENABLED)
        for _ in range(20):
            matrices.record("vwap", MarketRegime.PANIC, "WEIRDUSDT", False, -0.12)

        both = matrices.multiplier("vwap", MarketRegime.PANIC, "WEIRDUSDT")
        regime_only = matrices.strategy_regime.multiplier("vwap", "PANIC")
        assert both < regime_only
        assert both == pytest.approx(
            regime_only * matrices.symbol_strategy.multiplier("WEIRDUSDT", "vwap")
        )

    def test_an_untested_combination_is_unaffected(self) -> None:
        matrices = MatrixSet(ENABLED)
        assert matrices.multiplier("momentum", MarketRegime.STRONG_TREND, "BTCUSDT") == 1.0
        assert matrices.blocked("momentum", MarketRegime.STRONG_TREND, "BTCUSDT") is False

    def test_the_report_covers_both_matrices(self) -> None:
        matrices = MatrixSet(ENABLED)
        matrices.record("momentum", MarketRegime.STRONG_TREND, "BTCUSDT", True, 1.0)
        report = matrices.report()
        assert {"strategy_regime", "symbol_strategy"} <= set(report)
        assert report["feedback_enabled"] is True
        assert report["strategy_regime"]["stats"]["total_trades"] == 1


class TestMatricesHaveNoAuthority:
    def test_nothing_here_can_place_an_order(self) -> None:
        source = inspect.getsource(matrices_module)
        for forbidden in ("OrderIntent", "place_order", "gateway", "execution"):
            assert forbidden not in source

    def test_a_multiplier_can_never_exceed_one(self) -> None:
        """Property check across a wide range of records."""
        for r in (-5.0, -1.0, -0.25, -0.01, 0.0, 0.5, 3.0, 20.0):
            matrix = PerformanceMatrix("p", min_trades=5)
            fill(matrix, "s", "c", r=r, count=10)
            assert 0.0 <= matrix.multiplier("s", "c") <= 1.0
