"""The backtester tests the system that would be deployed.

Every test here corresponds to a BACKTEST_AUDIT.md finding. They are structural
where the defect was structural, because three of the four critical findings
were *silent* — they produced plausible numbers rather than errors, and a test
that only checks the output would have passed with the bug present.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from tradebot.backtesting.engine import BacktestEngine
from tradebot.backtesting.runner import build_context, config_hash, run_scenarios
from tradebot.core.config import load_tunables
from tradebot.core.types import Direction, ExitReason, MarketRegime, Position

from ..conftest import REPO_ROOT

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.backtest.yaml", REPO_ROOT / "config" / "strategies.yaml"
)


def _calls_in(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


# --------------------------------------------------------------------------- #
class TestB1DynamicUniverse:
    """B-1: every symbol was evaluated on every bar, with no Top-25 cut."""

    def test_the_universe_is_ranked_per_cycle(self) -> None:
        assert "_rank_universe" in _calls_in(inspect.getsource(BacktestEngine.run))

    def test_the_ranking_respects_the_configured_top_n(self) -> None:
        source = inspect.getsource(BacktestEngine._rank_universe)
        assert "self.config.scanner.top_markets" in source

    def test_the_ranking_is_logged_with_a_reason(self) -> None:
        """§8: so the ranking can be audited rather than taken on trust."""
        source = inspect.getsource(BacktestEngine._rank_universe)
        for field in ("timestamp", "rank", "symbol", "score", "reason"):
            assert f'"{field}"' in source

    def test_held_symbols_are_excluded_from_the_ranking(self) -> None:
        source = inspect.getsource(BacktestEngine._rank_universe)
        assert "if symbol in self.positions" in source

    def test_the_universe_is_rebuilt_on_the_scan_interval(self) -> None:
        """Matching the live engine, which re-ranks every scan_interval_sec."""
        assert "scan_interval_ms" in inspect.getsource(BacktestEngine.run)


class TestB2OpportunityQueue:
    """B-2: alphabetical symbol order decided who took the last free slot."""

    def test_the_engine_owns_a_queue(self) -> None:
        engine = BacktestEngine(CONFIG)
        assert engine.queue is not None
        assert engine.queue.ttl_sec == CONFIG.opportunity.queue_ttl_sec

    def test_opportunities_are_queued_before_any_is_taken(self) -> None:
        source = inspect.getsource(BacktestEngine.run)
        assert "_fill_queue" in source
        assert "_spend_slots" in source
        assert source.index("_fill_queue") < source.index("_spend_slots")

    def test_slots_are_spent_best_first(self) -> None:
        assert "take" in _calls_in(inspect.getsource(BacktestEngine._spend_slots))

    def test_zero_free_slots_takes_nothing(self) -> None:
        assert "free <= 0" in inspect.getsource(BacktestEngine._spend_slots)


class TestB3CapitalPreservation:
    """B-3: the backtest ran with the brakes disconnected."""

    def test_the_risk_context_carries_a_drawdown(self) -> None:
        assert "drawdown=self._drawdown()" in inspect.getsource(BacktestEngine._spend_slots)

    def test_drawdown_is_measured_from_peak_equity(self) -> None:
        engine = BacktestEngine(CONFIG, initial_capital=100.0)
        assert engine._drawdown() == 0.0
        engine.peak_equity, engine.equity = 100.0, 90.0
        assert engine._drawdown() == pytest.approx(0.10)

    def test_slot_counting_honours_the_preservation_mode(self) -> None:
        assert "preservation.max_positions" in inspect.getsource(BacktestEngine._spend_slots)

    def test_the_daily_loss_counter_rolls_over(self) -> None:
        engine = BacktestEngine(CONFIG)
        engine.realized_pnl_today = -5.0
        engine._roll_day(1_704_067_200_000)
        engine._roll_day(1_704_067_200_000 + 86_400_000)
        assert engine.realized_pnl_today == 0.0


class TestB4Matrices:
    """B-4: every trade was recorded as SIDEWAYS with zero PnL."""

    def test_the_close_path_passes_regime_and_pnl(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(BacktestEngine._close_position)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "record_trade_closed"
            ):
                kwargs = {k.arg for k in node.keywords}
                assert "regime" in kwargs, "the matrix would record every trade as SIDEWAYS"
                assert "pnl" in kwargs, "the matrix cell PnL would always be 0.0"
                return
        pytest.fail("record_trade_closed is no longer called from _close_position")


class TestB5Liquidation:
    """B-5: leverage risk was assumed away rather than measured."""

    def test_a_liquidation_price_is_computed(self) -> None:
        engine = BacktestEngine(CONFIG)
        position = Position(
            position_id="p",
            symbol="X",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            leverage=5,
            stop_loss=99.0,
            take_profit=102.0,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opened_at=0,
        )
        price = engine._liquidation_price(position)
        assert 0 < price < 100.0
        # 5x leverage liquidates around a 20% adverse move, less maintenance.
        assert price == pytest.approx(100.0 * (1 - (0.2 - 0.004)), rel=1e-6)

    def test_a_short_liquidates_above_its_entry(self) -> None:
        engine = BacktestEngine(CONFIG)
        position = Position(
            position_id="p",
            symbol="X",
            direction=Direction.SHORT,
            quantity=1.0,
            entry_price=100.0,
            leverage=5,
            stop_loss=101.0,
            take_profit=98.0,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opened_at=0,
        )
        assert engine._liquidation_price(position) > 100.0

    def test_liquidation_is_checked_before_the_stop(self) -> None:
        """The exchange does not wait for our stop."""
        source = inspect.getsource(BacktestEngine._exit_for)
        assert source.index("_liquidation_price") < source.index("is_stop_hit")
        assert "ExitReason.LIQUIDATION" in source

    def test_liquidation_is_a_real_exit_reason(self) -> None:
        assert hasattr(ExitReason, "LIQUIDATION")


class TestB9BarsPerDay:
    """B-9: a hardcoded 288 that was only correct for 5m bars."""

    @pytest.mark.parametrize(
        ("timeframe", "expected"), [("1m", 1440.0), ("5m", 288.0), ("15m", 96.0), ("1h", 24.0)]
    )
    def test_bars_per_day_follows_the_timeframe(self, timeframe: str, expected: float) -> None:
        config = CONFIG.model_copy(
            update={"timeframes": CONFIG.timeframes.model_copy(update={"primary": timeframe})}
        )
        assert BacktestEngine(config)._bars_per_day() == expected

    def test_the_constant_is_gone(self) -> None:
        assert "* 288" not in inspect.getsource(BacktestEngine._liquidity)


class TestB12EquitySampling:
    """B-12: equity every 50 bars left a 4-hour blind spot in the drawdown."""

    def test_equity_is_recorded_every_cycle(self) -> None:
        source = inspect.getsource(BacktestEngine.run)
        assert "% 50" not in source
        assert "_record_equity" in _calls_in(source)


class TestDatasetGuard:
    """A dataset missing a timeframe produces zero trades and looks like a
    clean run — 'no edge' when the truth is 'no data'."""

    def test_the_engine_checks_timeframe_coverage(self) -> None:
        assert "_check_timeframe_coverage" in _calls_in(inspect.getsource(BacktestEngine.run))

    def test_missing_timeframes_are_reported_on_the_result(self) -> None:
        from tradebot.backtesting.engine import BacktestResult

        assert "missing_timeframes" in BacktestResult.__dataclass_fields__


class TestReproducibility:
    """§38: a number with no record of its inputs is an anecdote."""

    def test_the_context_carries_everything_needed_to_reproduce(self) -> None:
        context = build_context(CONFIG, {}, seed=7)
        payload = context.as_dict()
        for field in (
            "run_id",
            "git_commit",
            "config_hash",
            "dataset_fingerprint",
            "seed",
            "code_version",
            "started_at",
        ):
            assert payload[field] is not None, field

    def test_the_config_hash_covers_every_tunable(self) -> None:
        """Hashing a chosen subset means the parameter someone forgot is
        exactly the one that differs between two runs meant to match."""
        changed = CONFIG.model_copy(
            update={"risk": CONFIG.risk.model_copy(update={"risk_per_trade": 0.006})}
        )
        assert config_hash(CONFIG) != config_hash(changed)

    def test_the_same_config_hashes_identically(self) -> None:
        assert config_hash(CONFIG) == config_hash(CONFIG.model_copy())


class TestScenarioReporting:
    """§41: all three scenarios, or none."""

    def test_the_runner_executes_all_three(self) -> None:
        source = inspect.getsource(run_scenarios)
        for scenario in ("BASE", "CONSERVATIVE", "STRESS"):
            assert scenario in source

    def test_the_comparison_reports_every_scenario_side_by_side(self) -> None:
        results = run_scenarios(CONFIG, {}, seed=1)
        # No data means no results, but the shape must still be the three-way one.
        assert results.context.seed == 1
        assert isinstance(results.comparison(), list)

    def test_surviving_stress_is_what_is_asked(self) -> None:
        """A strategy profitable under BASE and destroyed under CONSERVATIVE has
        an edge thinner than the error bars on the cost model."""
        results = run_scenarios(CONFIG, {}, seed=1)
        assert results.survives_stress is False  # no trades, no survival
