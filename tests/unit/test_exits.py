"""Exit conditions the exchange cannot evaluate.

Two of these rules were configurable in V1 and implemented in none of it:
`exit_on_signal_flip` and `exit_on_negative_edge` were read from YAML and never
consulted. A flag that does nothing is worse than a missing feature, because it
reads as a guarantee.

The third — the local safety net — exists because a position whose protective
stop has quietly gone is the single most expensive failure this system can have.
"""

from __future__ import annotations

import pytest

from tradebot.core.config import TunableConfig, load_tunables
from tradebot.core.types import Direction, ExitReason, MarketRegime, Position
from tradebot.execution.exits import ExitContext, ExitEvaluator, holding_edge

from ..conftest import REPO_ROOT

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
)
OPENED_MS = 1_700_000_000_000


def make_position(
    direction: Direction = Direction.LONG,
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 102.0,
    strategy: str = "momentum",
) -> Position:
    return Position(
        position_id="p_test",
        symbol="TESTUSDT",
        direction=direction,
        quantity=1.0,
        entry_price=entry,
        leverage=2,
        stop_loss=stop,
        take_profit=target,
        strategy=strategy,
        regime=MarketRegime.STRONG_TREND,
        opened_at=OPENED_MS,
        entry_notional=entry,
        initial_stop=stop,
        initial_risk=abs(entry - stop),
        highest_price=entry,
        lowest_price=entry,
        stop_order_id="tb_sl_test",
    )


@pytest.fixture
def evaluator() -> ExitEvaluator:
    return ExitEvaluator(CONFIG)


def ctx(price: float = 100.5, held_sec: float = 60.0, **kwargs) -> ExitContext:
    return ExitContext(price=price, now_ms=OPENED_MS + int(held_sec * 1000), **kwargs)


class TestHolding:
    def test_a_healthy_position_is_held(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(make_position(), ctx(price=100.5))
        assert decision.should_exit is False
        assert decision.reason is None

    def test_a_price_of_zero_does_not_trigger_the_safety_net(
        self, evaluator: ExitEvaluator
    ) -> None:
        """A missing price is missing information, not a breached stop."""
        decision = evaluator.evaluate(make_position(), ctx(price=0.0))
        assert decision.reason is not ExitReason.STOP_LOSS


class TestLocalSafetyNet:
    def test_a_long_through_its_stop_is_closed_at_market(self, evaluator: ExitEvaluator) -> None:
        """The exchange stop did not fire. Closing here is worse than the stop
        price we wanted and far better than an unbounded loss."""
        decision = evaluator.evaluate(make_position(), ctx(price=98.5))
        assert decision.reason is ExitReason.STOP_LOSS
        assert decision.urgent is True

    def test_a_short_through_its_stop_is_closed(self, evaluator: ExitEvaluator) -> None:
        position = make_position(Direction.SHORT, entry=100.0, stop=101.0, target=98.0)
        decision = evaluator.evaluate(position, ctx(price=101.5))
        assert decision.reason is ExitReason.STOP_LOSS

    def test_exactly_at_the_stop_counts_as_through_it(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(make_position(), ctx(price=99.0))
        assert decision.reason is ExitReason.STOP_LOSS

    @pytest.mark.parametrize(
        ("direction", "entry", "stop", "target", "price"),
        [
            (Direction.LONG, 100.0, 99.0, 102.0, 102.5),
            (Direction.SHORT, 100.0, 101.0, 98.0, 97.5),
        ],
    )
    def test_a_reached_target_is_booked(
        self,
        evaluator: ExitEvaluator,
        direction: Direction,
        entry: float,
        stop: float,
        target: float,
        price: float,
    ) -> None:
        position = make_position(direction, entry=entry, stop=stop, target=target)
        decision = evaluator.evaluate(position, ctx(price=price))
        assert decision.reason is ExitReason.TAKE_PROFIT

    def test_a_vanished_stop_order_closes_the_position(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(make_position(), ctx(price=100.5, stop_order_missing=True))
        assert decision.reason is ExitReason.RISK_EVENT
        assert decision.urgent is True

    def test_a_breached_stop_outranks_everything_else(self, evaluator: ExitEvaluator) -> None:
        """Worst-first ordering: the loss being taken right now beats a time
        limit and a deteriorating thesis."""
        decision = evaluator.evaluate(
            make_position(),
            ctx(
                price=98.0,
                held_sec=99_999,
                regime_blocks=True,
                signal_direction=Direction.SHORT,
                signal_confidence=99.0,
                holding_edge=-0.01,
            ),
        )
        assert decision.reason is ExitReason.STOP_LOSS


class TestTimeLimit:
    def test_the_sixty_minute_cap_is_enforced(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(
            make_position(), ctx(price=100.5, held_sec=CONFIG.trade.max_duration_sec)
        )
        assert decision.reason is ExitReason.TIME_LIMIT

    def test_one_second_short_of_the_cap_is_held(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(
            make_position(), ctx(price=100.5, held_sec=CONFIG.trade.max_duration_sec - 1)
        )
        assert decision.should_exit is False

    def test_the_configured_cap_never_exceeds_sixty_minutes(self) -> None:
        assert CONFIG.trade.max_duration_sec <= 3600


class TestRegimeChange:
    def test_a_blocking_regime_closes_the_position(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(make_position(), ctx(regime_blocks=True))
        assert decision.reason is ExitReason.REGIME_CHANGE

    def test_it_can_be_switched_off(self) -> None:
        config = CONFIG.model_copy(
            update={"trade": CONFIG.trade.model_copy(update={"exit_on_regime_change": False})}
        )
        decision = ExitEvaluator(config).evaluate(make_position(), ctx(regime_blocks=True))
        assert decision.should_exit is False


class TestSignalFlip:
    def test_a_confident_reversal_closes_a_long(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(
            make_position(),
            ctx(signal_direction=Direction.SHORT, signal_confidence=80.0),
        )
        assert decision.reason is ExitReason.SIGNAL_FLIP

    def test_agreement_holds_the_position(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(
            make_position(),
            ctx(signal_direction=Direction.LONG, signal_confidence=90.0),
        )
        assert decision.should_exit is False

    def test_wait_is_not_a_flip(self, evaluator: ExitEvaluator) -> None:
        """WAIT is the absence of an opinion, not an opinion against us. Closing
        on every lull would churn the account into its own fees."""
        decision = evaluator.evaluate(
            make_position(),
            ctx(signal_direction=Direction.WAIT, signal_confidence=95.0),
        )
        assert decision.should_exit is False

    def test_no_signal_at_all_is_not_a_flip(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(make_position(), ctx(signal_direction=None))
        assert decision.should_exit is False

    def test_a_weak_reversal_is_ignored(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(
            make_position(),
            ctx(
                signal_direction=Direction.SHORT,
                signal_confidence=CONFIG.aggregator.min_signal_confidence - 1,
            ),
        )
        assert decision.should_exit is False

    def test_it_can_be_switched_off(self) -> None:
        config = CONFIG.model_copy(
            update={"trade": CONFIG.trade.model_copy(update={"exit_on_signal_flip": False})}
        )
        decision = ExitEvaluator(config).evaluate(
            make_position(), ctx(signal_direction=Direction.SHORT, signal_confidence=95.0)
        )
        assert decision.should_exit is False


class TestNegativeEdge:
    def test_a_negative_holding_edge_closes_the_position(self) -> None:
        config = CONFIG.model_copy(
            update={"trade": CONFIG.trade.model_copy(update={"exit_on_negative_edge": True})}
        )
        decision = ExitEvaluator(config).evaluate(make_position(), ctx(holding_edge=-0.0004))
        assert decision.reason is ExitReason.NEGATIVE_EDGE

    def test_a_positive_edge_holds(self, evaluator: ExitEvaluator) -> None:
        decision = evaluator.evaluate(make_position(), ctx(holding_edge=0.0012))
        assert decision.should_exit is False

    def test_an_unknown_edge_holds(self, evaluator: ExitEvaluator) -> None:
        """No estimate is not the same as a bad estimate."""
        decision = evaluator.evaluate(make_position(), ctx(holding_edge=None))
        assert decision.should_exit is False

    def test_it_can_be_switched_off(self) -> None:
        config = CONFIG.model_copy(
            update={"trade": CONFIG.trade.model_copy(update={"exit_on_negative_edge": False})}
        )
        decision = ExitEvaluator(config).evaluate(make_position(), ctx(holding_edge=-0.05))
        assert decision.should_exit is False


class TestHoldingEdgeArithmetic:
    def test_it_measures_from_the_current_price_not_the_entry(self) -> None:
        """The sunk-cost test. A position deep in loss must not be held on the
        strength of the loss it has already taken."""
        position = make_position(entry=100.0, stop=99.0, target=102.0)
        # Price at 99.1: 2.9% still to win, 0.1% still to lose.
        near_stop = holding_edge(position, 99.1, win_probability=0.5, round_trip_cost=0.0)
        # Price at 101.9: 0.1% left to win, 2.9% left to lose.
        near_target = holding_edge(position, 101.9, win_probability=0.5, round_trip_cost=0.0)
        assert near_stop > 0
        assert near_target < 0

    def test_costs_are_subtracted(self) -> None:
        position = make_position()
        free = holding_edge(position, 100.5, 0.5, round_trip_cost=0.0)
        costly = holding_edge(position, 100.5, 0.5, round_trip_cost=0.001)
        assert costly == pytest.approx(free - 0.001)

    def test_a_higher_win_probability_raises_the_edge(self) -> None:
        position = make_position()
        assert holding_edge(position, 100.5, 0.7, 0.0) > holding_edge(position, 100.5, 0.4, 0.0)

    def test_a_short_position_is_measured_the_right_way_round(self) -> None:
        position = make_position(Direction.SHORT, entry=100.0, stop=101.0, target=98.0)
        # Price fell to 98.5: mostly won, little left to gain, much left to risk.
        assert holding_edge(position, 98.5, 0.5, 0.0) < 0
        # Price at 100.5: still most of the move ahead.
        assert holding_edge(position, 100.5, 0.5, 0.0) > 0

    @pytest.mark.parametrize("price", [0.0, -5.0])
    def test_a_missing_price_returns_zero_not_a_signal(self, price: float) -> None:
        assert holding_edge(make_position(), price, 0.5, 0.0) == 0.0

    def test_past_the_levels_returns_zero(self) -> None:
        """The level rules own that case; the arithmetic would be meaningless."""
        position = make_position()
        assert holding_edge(position, 103.0, 0.5, 0.0) == 0.0
        assert holding_edge(position, 98.0, 0.5, 0.0) == 0.0

    def test_the_probability_is_clamped(self) -> None:
        position = make_position()
        assert holding_edge(position, 100.5, 5.0, 0.0) == holding_edge(position, 100.5, 1.0, 0.0)
        assert holding_edge(position, 100.5, -3.0, 0.0) == holding_edge(position, 100.5, 0.0, 0.0)


class TestBookkeeping:
    def test_exit_reasons_are_counted(self, evaluator: ExitEvaluator) -> None:
        evaluator.evaluate(make_position(), ctx(regime_blocks=True))
        evaluator.evaluate(make_position(), ctx(regime_blocks=True))
        evaluator.evaluate(make_position(), ctx(held_sec=99_999))
        assert evaluator.stats() == {"REGIME_CHANGE": 2, "TIME_LIMIT": 1}

    def test_holding_is_not_counted(self, evaluator: ExitEvaluator) -> None:
        evaluator.evaluate(make_position(), ctx())
        assert evaluator.stats() == {}


def test_config_type_is_what_the_evaluator_expects() -> None:
    assert isinstance(CONFIG, TunableConfig)
