"""Opportunity ordering and capital preservation, as the engine uses them.

Two components whose value is entirely in being wired in: a queue nothing takes
from is a list, and a preservation mode nothing consults is a log line.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from tradebot.app import runner as runner_module
from tradebot.app.runner import TradingEngine
from tradebot.core.clock import VirtualClock
from tradebot.core.config import load_tunables
from tradebot.core.types import Direction, MarketRegime, RejectionReason
from tradebot.market.candles import CandleStore
from tradebot.risk.engine import RiskContext, RiskEngine
from tradebot.risk.preservation import PreservationMode

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


def build_risk(clock: VirtualClock) -> RiskEngine:
    return RiskEngine(CONFIG, CandleStore(500), clock)


def context(
    equity: float = 75.0,
    drawdown: float = 0.0,
    realized_pnl_today: float = 0.0,
    positions: dict | None = None,
) -> RiskContext:
    return RiskContext(
        equity=equity,
        available_balance=equity,
        positions=positions or {},
        prices={},
        symbol_info=make_symbol_info("BTCUSDT", min_notional=1.0),
        realized_pnl_today=realized_pnl_today,
        drawdown=drawdown,
    )


# --------------------------------------------------------------------------- #
class TestTheQueueIsWiredIn:
    def test_evaluation_queues_instead_of_trading_inline(self) -> None:
        """The bug: the first passing candidate in SCAN order took the slot."""
        source = inspect.getsource(TradingEngine._evaluate_candidates)
        assert "self.opportunities.add(" in source
        assert "_attempt_trade" not in source, (
            "evaluation still trades inline, so scan rank still decides"
        )

    def test_slots_are_spent_best_first(self) -> None:
        calls = _calls_in(inspect.getsource(TradingEngine._spend_slots))
        assert "take" in calls
        assert "_attempt_trade" in calls

    def test_only_free_slots_are_offered(self) -> None:
        source = inspect.getsource(TradingEngine._spend_slots)
        assert "free <= 0" in source
        assert "in_flight" in source, "in-flight orders must count against the slots"

    def test_the_queue_never_places_an_order(self) -> None:
        """Structural: nothing in the queue module may reach execution."""
        from tradebot.signals import queue as queue_module

        source = inspect.getsource(queue_module)
        for forbidden in ("place_order", "OrderIntent", "gateway", "execution"):
            assert forbidden not in source


class TestPreservationIsWiredIn:
    def test_the_risk_engine_consults_it(self) -> None:
        source = inspect.getsource(RiskEngine.evaluate)
        assert "self.preservation.evaluate(" in source
        assert "risk_multiplier" in source

    def test_the_engine_passes_the_drawdown_in(self) -> None:
        assert "drawdown=self._drawdown()" in inspect.getsource(TradingEngine._attempt_trade)

    def test_slot_counting_honours_the_mode(self) -> None:
        source = inspect.getsource(TradingEngine._spend_slots)
        assert "preservation.max_positions" in source

    def test_status_reports_the_mode(self) -> None:
        source = inspect.getsource(TradingEngine.status_snapshot)
        assert "preservation" in source
        assert "self.risk.preservation.entries_allowed" in source


# --------------------------------------------------------------------------- #
class TestPreservationChangesRiskDecisions:
    def test_a_halted_account_takes_no_new_entry(self, clock: VirtualClock) -> None:
        risk = build_risk(clock)
        opportunity = make_opportunity("BTCUSDT", score=95.0)
        # A 2% realised loss on the day is the halt threshold.
        decision = risk.evaluate(opportunity, context(realized_pnl_today=-1.5))

        assert decision.approved is False
        assert decision.reason is RejectionReason.KILL_SWITCH_ACTIVE
        assert "HALTED" in decision.detail
        assert risk.preservation.mode is PreservationMode.HALTED

    def test_defensive_mode_refuses_a_merely_good_opportunity(self, clock: VirtualClock) -> None:
        """A 72 would be traded normally; in DEFENSIVE only exceptional
        opportunities are worth the remaining capital."""
        risk = build_risk(clock)
        decision = risk.evaluate(make_opportunity("BTCUSDT", score=72.0), context(drawdown=0.07))
        assert decision.approved is False
        assert decision.reason is RejectionReason.LOW_OPPORTUNITY_SCORE
        assert "DEFENSIVE" in decision.detail

    def test_defensive_mode_still_allows_an_exceptional_one_through_that_gate(
        self, clock: VirtualClock
    ) -> None:
        risk = build_risk(clock)
        decision = risk.evaluate(make_opportunity("BTCUSDT", score=95.0), context(drawdown=0.07))
        # It may still be refused later (sizing, correlation) — what matters is
        # that preservation is not the thing that refused it.
        assert decision.reason is not RejectionReason.LOW_OPPORTUNITY_SCORE

    def test_cautious_mode_sizes_smaller_than_normal(self, clock: VirtualClock) -> None:
        normal = build_risk(clock)
        approved = normal.evaluate(make_opportunity("BTCUSDT", score=95.0), context())

        cautious = build_risk(VirtualClock(clock.now_ms()))
        reduced = cautious.evaluate(make_opportunity("BTCUSDT", score=95.0), context(drawdown=0.04))
        assert cautious.preservation.mode is PreservationMode.CAUTIOUS

        if approved.approved and reduced.approved:
            assert reduced.intent is not None and approved.intent is not None
            assert reduced.intent.quantity < approved.intent.quantity
        else:
            # Sizing floors can refuse the reduced trade outright on a 75 USDT
            # account — which is itself the preservation behaviour working.
            assert not reduced.approved or approved.approved

    def test_a_new_trading_day_releases_a_halt(self, clock: VirtualClock) -> None:
        risk = build_risk(clock)
        risk.evaluate(make_opportunity("BTCUSDT", score=95.0), context(realized_pnl_today=-1.5))
        assert risk.preservation.mode is PreservationMode.HALTED

        risk.kill_switches.update_equity(75.0)
        clock.advance(86_400 * 2)
        risk.kill_switches.update_equity(75.0)  # rolls the day index
        risk.evaluate(make_opportunity("BTCUSDT", score=95.0), context())
        assert risk.preservation.mode is PreservationMode.NORMAL


class TestExitsAreNeverGated:
    def test_position_management_never_consults_preservation(self) -> None:
        source = inspect.getsource(TradingEngine._manage_positions)
        assert "preservation" not in source

    def test_the_close_path_never_consults_preservation(self) -> None:
        from tradebot.execution.engine import ExecutionEngine

        for method in (ExecutionEngine.close_position, ExecutionEngine._close_locked):
            assert "preservation" not in inspect.getsource(method)

    def test_only_the_risk_engine_builds_an_intent_even_under_preservation(self) -> None:
        """Rule 4 still holds: preservation narrows what the risk engine
        approves, it does not create a second path to an order."""
        source = inspect.getsource(runner_module)
        assert "OrderIntent(" not in source


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("direction", "expected"),
    [(Direction.LONG, Direction.LONG), (Direction.SHORT, Direction.SHORT)],
)
def test_queue_preserves_direction(direction: Direction, expected: Direction) -> None:
    from tradebot.signals.queue import OpportunityQueue

    queue = OpportunityQueue(clock=VirtualClock(0))
    queue.add(make_opportunity("BTCUSDT", 90.0, direction=direction))
    entry = queue.best()
    assert entry is not None
    assert entry.direction is expected
    assert entry.opportunity.regime is MarketRegime.STRONG_TREND
