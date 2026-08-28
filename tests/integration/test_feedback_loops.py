"""The engine's feedback loops: exits, execution quality, and the matrices.

Each of these is a component whose entire value is in being consulted. The V1
audit's lesson was that `exit_on_signal_flip` and `exit_on_negative_edge` were
configurable, documented and never read — so these tests assert the CALLS, not
just the behaviour of the components in isolation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from tradebot.app.runner import TradingEngine
from tradebot.core.clock import VirtualClock
from tradebot.core.config import load_tunables
from tradebot.core.types import Direction, ExitReason, MarketRegime, RejectionReason
from tradebot.execution.engine import ExecutionEngine
from tradebot.market.candles import CandleStore
from tradebot.risk.engine import RiskContext, RiskEngine

from ..conftest import REPO_ROOT
from ..fakes import make_symbol_info
from ..unit.test_opportunity_queue import make_opportunity

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
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
class TestEveryExitFlagIsActuallyRead:
    """The V1 failure: three flags in YAML, one implementation."""

    @pytest.mark.parametrize(
        "flag",
        ["exit_on_regime_change", "exit_on_signal_flip", "exit_on_negative_edge"],
    )
    def test_the_flag_is_consulted(self, flag: str) -> None:
        from tradebot.execution import exits as exits_module

        assert f"self.config.trade.{flag}" in inspect.getsource(exits_module), (
            f"{flag} is configurable but never read"
        )

    def test_the_monitor_loop_uses_the_evaluator(self) -> None:
        calls = _calls_in(inspect.getsource(TradingEngine._manage_positions))
        assert "evaluate" in calls
        assert "_exit_context" in calls

    @pytest.mark.parametrize(
        ("field", "builder"),
        [
            ("signal_direction", "_record_consensus"),
            ("holding_edge", "_holding_edge"),
            ("stop_order_missing", "_stop_order_missing"),
        ],
    )
    def test_each_context_field_has_a_real_source(self, field: str, builder: str) -> None:
        """A context field the engine always leaves at its default is a dead
        rule wearing a live one's clothes."""
        source = inspect.getsource(TradingEngine._exit_context)
        assert field in source
        assert hasattr(TradingEngine, builder)

    def test_consensus_is_recorded_during_evaluation(self) -> None:
        assert "_record_consensus" in _calls_in(
            inspect.getsource(TradingEngine._evaluate_candidates)
        )

    def test_an_exit_is_never_blocked_by_safe_mode_or_preservation(self) -> None:
        from tradebot.execution import exits as exits_module

        source = inspect.getsource(exits_module)
        for forbidden in ("safe_mode", "preservation", "entries_blocked", "kill_switch"):
            assert forbidden not in source


class TestExecutionQualityIsWiredIn:
    def test_entries_are_recorded(self) -> None:
        assert "_record_execution" in _calls_in(inspect.getsource(TradingEngine._attempt_trade))

    def test_exits_are_recorded(self) -> None:
        assert "_record_exit_execution" in _calls_in(
            inspect.getsource(TradingEngine._on_trade_completed)
        )

    def test_the_measurement_feeds_back_into_the_cost_model(self) -> None:
        """Without this the loop is a dashboard, not a correction."""
        source = inspect.getsource(TradingEngine._recalibrate_costs)
        assert "self.cost_model.set_slippage_adjustment(" in source
        assert "is_calibrated" in source

    def test_recalibration_runs_after_every_recorded_fill(self) -> None:
        for method in (
            TradingEngine._record_execution,
            TradingEngine._record_exit_execution,
        ):
            assert "_recalibrate_costs" in _calls_in(inspect.getsource(method))


class TestMatricesAreWiredIn:
    def test_results_are_recorded_with_their_regime(self) -> None:
        source = inspect.getsource(TradingEngine._on_trade_completed)
        assert "regime=trade.regime" in source

    def test_the_risk_engine_consults_them(self) -> None:
        source = inspect.getsource(RiskEngine.evaluate)
        assert "self.matrices.multiplier(" in source
        assert "matrix_multiplier" in source

    def test_the_multiplier_scales_the_risk_fraction(self) -> None:
        assert "risk_fraction *= matrix_multiplier" in inspect.getsource(RiskEngine.evaluate)

    def test_the_shipped_config_records_but_does_not_select(self) -> None:
        """IMPLEMENTATION_PLAN_V2 V2-10: diagnosis, not selection. Suppressing a
        combination on thin data is overfitting with extra steps."""
        assert CONFIG.matrices.feedback_enabled is False
        risk = RiskEngine(CONFIG, CandleStore(500), VirtualClock(0))
        for _ in range(200):
            risk.matrices.record("momentum", MarketRegime.STRONG_TREND, "BTCUSDT", False, -3.0)
        assert risk.matrices.multiplier("momentum", MarketRegime.STRONG_TREND, "BTCUSDT") == 1.0


# --------------------------------------------------------------------------- #
class TestMatricesChangeRiskDecisions:
    def _risk(self, clock: VirtualClock) -> RiskEngine:
        # Feedback is off in the shipped config by design; these tests are
        # about what happens once an operator turns it on.
        config = CONFIG.model_copy(
            update={
                "matrices": CONFIG.matrices.model_copy(
                    update={
                        "feedback_enabled": True,
                        "strategy_regime_min_trades": 10,
                        "symbol_strategy_min_trades": 10,
                    }
                )
            }
        )
        return RiskEngine(config, CandleStore(500), clock)

    def _context(self) -> RiskContext:
        return RiskContext(
            equity=75.0,
            available_balance=75.0,
            positions={},
            prices={},
            symbol_info=make_symbol_info("BTCUSDT", min_notional=1.0),
        )

    def test_a_combination_with_a_losing_record_is_refused(self, clock: VirtualClock) -> None:
        risk = self._risk(clock)
        for _ in range(25):
            risk.matrices.record(
                "momentum",
                MarketRegime.STRONG_TREND,
                "BTCUSDT",
                won=False,
                r_multiple=-1.0,
            )

        decision = risk.evaluate(make_opportunity("BTCUSDT", score=95.0), self._context())
        assert decision.approved is False
        assert decision.reason is RejectionReason.STRATEGY_DISABLED
        assert "losing record" in decision.detail

    def test_a_good_record_does_not_block(self, clock: VirtualClock) -> None:
        risk = self._risk(clock)
        for _ in range(25):
            risk.matrices.record(
                "momentum",
                MarketRegime.STRONG_TREND,
                "BTCUSDT",
                won=True,
                r_multiple=1.6,
            )
        decision = risk.evaluate(make_opportunity("BTCUSDT", score=95.0), self._context())
        assert decision.reason is not RejectionReason.STRATEGY_DISABLED

    def test_the_same_strategy_is_judged_separately_per_regime(self, clock: VirtualClock) -> None:
        """The whole point: a strategy losing in one regime says nothing about
        how it does in another."""
        risk = self._risk(clock)
        for _ in range(25):
            risk.matrices.record(
                "momentum", MarketRegime.SIDEWAYS, "BTCUSDT", won=False, r_multiple=-1.0
            )
        regimes = risk.matrices.strategy_regime
        assert regimes.multiplier("momentum", "SIDEWAYS") == 0.0
        assert regimes.multiplier("momentum", "STRONG_TREND") == 1.0

    def test_a_symbol_verdict_follows_the_strategy_across_regimes(
        self, clock: VirtualClock
    ) -> None:
        """The other half: a symbol on which a strategy loses is a symbol on
        which that strategy loses, whatever the regime says — so the combined
        multiplier stays suppressed even where the regime matrix is neutral."""
        risk = self._risk(clock)
        for _ in range(25):
            risk.matrices.record(
                "momentum", MarketRegime.SIDEWAYS, "WEIRDUSDT", won=False, r_multiple=-1.0
            )
        assert risk.matrices.blocked("momentum", MarketRegime.STRONG_TREND, "WEIRDUSDT")
        assert not risk.matrices.blocked("momentum", MarketRegime.STRONG_TREND, "BTCUSDT")


# --------------------------------------------------------------------------- #
class TestExitsAreNeverGated:
    def test_closing_is_reachable_with_every_switch_thrown(self) -> None:
        source = inspect.getsource(ExecutionEngine.close_position)
        for forbidden in ("entries_blocked", "safe_mode", "preservation", "kill_switch"):
            assert forbidden not in source

    def test_every_exit_reason_the_evaluator_can_return_is_a_real_one(self) -> None:
        from tradebot.execution import exits as exits_module

        source = inspect.getsource(exits_module)
        for name in (
            "STOP_LOSS",
            "TAKE_PROFIT",
            "TIME_LIMIT",
            "REGIME_CHANGE",
            "SIGNAL_FLIP",
            "NEGATIVE_EDGE",
            "RISK_EVENT",
        ):
            assert f"ExitReason.{name}" in source
            assert hasattr(ExitReason, name)

    def test_a_flipped_signal_needs_a_real_direction(self) -> None:
        """Guards against the engine reading a default and closing everything."""
        source = inspect.getsource(TradingEngine._exit_context)
        assert "direction: Direction | None = None" in source
        assert "signal_interval_sec" in source, (
            "a stale consensus must not be treated as a live flip"
        )


def test_direction_wait_is_never_treated_as_a_reversal() -> None:
    from tradebot.execution.exits import ExitContext, ExitEvaluator

    from ..unit.test_exits import make_position

    evaluator = ExitEvaluator(CONFIG)
    decision = evaluator.evaluate(
        make_position(),
        ExitContext(
            price=100.5,
            now_ms=1_700_000_060_000,
            signal_direction=Direction.WAIT,
            signal_confidence=100.0,
        ),
    )
    assert decision.should_exit is False
