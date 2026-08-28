"""The order state machine, driven by the execution engine.

The unit tests in `tests/unit/test_order_state.py` prove the machine's rules.
These prove the ENGINE drives it — that a real submission, a real fill and a
real protective-order fill all move an order through it, and that a stop filled
on the exchange becomes a completed trade in milliseconds rather than at the
next reconciliation sweep.
"""

from __future__ import annotations

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.config import load_tunables
from tradebot.core.errors import NetworkError, TimeoutError_
from tradebot.core.events import EventBus
from tradebot.core.types import (
    Direction,
    ExitReason,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
)
from tradebot.execution.engine import ExecutionEngine
from tradebot.execution.state import OrderState

from ..conftest import REPO_ROOT
from ..fakes import FakeGateway, make_symbol_info

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
)


def intent(
    symbol: str = "TESTUSDT",
    direction: Direction = Direction.LONG,
    quantity: float = 0.75,
    entry: float = 100.0,
) -> OrderIntent:
    stop = entry * 0.995 if direction is Direction.LONG else entry * 1.005
    target = entry * 1.01 if direction is Direction.LONG else entry * 0.99
    return OrderIntent(
        intent_id="abc123def456",
        symbol=symbol,
        direction=direction,
        side=OrderSide.for_entry(direction),
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=None,
        stop_loss=stop,
        take_profit=target,
        leverage=2,
        notional=quantity * entry,
        risk_amount=0.375,
        strategy="momentum",
        regime=MarketRegime.STRONG_TREND,
        opportunity_score=82.0,
        expected_net_edge=0.0015,
        metadata={"reference_price": entry},
    )


def build() -> tuple[ExecutionEngine, FakeGateway]:
    gateway = FakeGateway()
    gateway.symbols.setdefault("TESTUSDT", make_symbol_info("TESTUSDT", min_notional=1.0))
    engine = ExecutionEngine(CONFIG, gateway, EventBus(), VirtualClock(1_700_000_000_000))
    return engine, gateway


# --------------------------------------------------------------------------- #
class TestTheEngineDrivesTheMachine:
    async def test_a_successful_entry_walks_to_filled(self):
        engine, _ = build()
        request = intent()
        result = await engine.open_position(request)
        assert result.success

        tracked = engine.tracker.get(request.client_order_id)
        assert tracked is not None
        assert tracked.state is OrderState.FILLED
        assert [t.to_state for t in tracked.history][:2] == [
            OrderState.SUBMITTED,
            OrderState.FILLED,
        ]

    async def test_the_order_is_registered_before_it_is_sent(self):
        """If the process dies mid-submission, the client order id must still
        exist for reconciliation to look up."""
        engine, gateway = build()
        request = intent()
        seen: list[bool] = []
        original = gateway.place_order

        async def spy(sent):
            seen.append(sent.client_order_id in engine.tracker)
            return await original(sent)

        gateway.place_order = spy  # type: ignore[method-assign]
        await engine.open_position(request)
        assert seen == [True]

    async def test_a_rejection_is_terminal(self):
        engine, gateway = build()
        gateway.reject_next_order = True
        request = intent()
        result = await engine.open_position(request)

        assert not result.success
        tracked = engine.tracker.get(request.client_order_id)
        assert tracked is not None
        assert tracked.state is OrderState.REJECTED
        assert tracked.is_terminal

    @pytest.mark.parametrize("failure", [TimeoutError_("timed out"), NetworkError("reset")])
    async def test_a_timeout_leaves_the_order_indeterminate(self, failure: Exception):
        """Never guessed either way: a duplicate position and an unprotected
        one are both worse than blocking entries until reconciliation."""
        engine, gateway = build()

        async def fail(_sent):
            raise failure

        gateway.place_order = fail  # type: ignore[method-assign]
        request = intent()
        result = await engine.open_position(request)

        assert not result.success
        tracked = engine.tracker.get(request.client_order_id)
        assert tracked is not None
        assert tracked.state is OrderState.INDETERMINATE
        assert engine.tracker.indeterminate() == [tracked]
        assert engine.entries_blocked is True

    async def test_protective_orders_are_tracked_too(self):
        engine, gateway = build()
        await engine.open_position(intent())
        position = engine.positions["TESTUSDT"]

        stop = engine.tracker.get(position.stop_order_id or "")
        assert stop is not None
        assert stop.purpose == "STOP"
        assert stop.state is OrderState.ACKNOWLEDGED

        target = engine.tracker.get(position.take_profit_order_id or "")
        assert target is not None
        assert target.purpose == "TARGET"

    async def test_closing_cancels_the_resting_bracket_in_the_machine(self):
        engine, _ = build()
        await engine.open_position(intent())
        position = engine.positions["TESTUSDT"]
        stop_id = position.stop_order_id or ""

        await engine.close_position("TESTUSDT", ExitReason.MANUAL)
        assert engine.tracker.get(stop_id).state is OrderState.CANCELLED  # type: ignore[union-attr]
        assert engine.tracker.open_for("TESTUSDT") == []


class TestUserStreamFills:
    async def test_a_stop_fill_closes_the_position_immediately(self):
        """The point of the user stream: a stop that fills on the exchange
        becomes a completed trade now, not at the next reconciliation sweep —
        which is how a flat account ends up refusing trades on
        'max concurrent positions'."""
        engine, _ = build()
        await engine.open_position(intent(entry=100.0))
        position = engine.positions["TESTUSDT"]
        stop_id = position.stop_order_id or ""

        await engine.on_order_update(
            {
                "client_order_id": stop_id,
                "symbol": "TESTUSDT",
                "status": "FILLED",
                "filled_quantity": position.quantity,
                "average_price": 99.5,
                "commission": 0.03,
            }
        )

        assert "TESTUSDT" not in engine.positions
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade.exit_reason is ExitReason.STOP_LOSS
        assert trade.exit_price == 99.5
        assert trade.net_pnl < 0

    async def test_a_target_fill_books_a_take_profit(self):
        engine, _ = build()
        await engine.open_position(intent(entry=100.0))
        position = engine.positions["TESTUSDT"]

        await engine.on_order_update(
            {
                "client_order_id": position.take_profit_order_id,
                "symbol": "TESTUSDT",
                "status": "FILLED",
                "filled_quantity": position.quantity,
                "average_price": 101.0,
            }
        )
        assert engine.trades[0].exit_reason is ExitReason.TAKE_PROFIT
        assert engine.trades[0].net_pnl > 0

    async def test_a_partial_fill_does_not_close_anything(self):
        engine, _ = build()
        await engine.open_position(intent())
        position = engine.positions["TESTUSDT"]

        await engine.on_order_update(
            {
                "client_order_id": position.stop_order_id,
                "symbol": "TESTUSDT",
                "status": "PARTIALLY_FILLED",
                "filled_quantity": position.quantity / 3,
                "average_price": 99.5,
            }
        )
        assert "TESTUSDT" in engine.positions
        assert engine.trades == []

    async def test_a_duplicate_fill_event_books_one_trade(self):
        """The same event can arrive twice; the machine's terminal rule is what
        stops the second one booking a phantom second trade."""
        engine, _ = build()
        await engine.open_position(intent())
        position = engine.positions["TESTUSDT"]
        update = {
            "client_order_id": position.stop_order_id,
            "symbol": "TESTUSDT",
            "status": "FILLED",
            "filled_quantity": position.quantity,
            "average_price": 99.5,
        }
        await engine.on_order_update(update)
        await engine.on_order_update(update)
        assert len(engine.trades) == 1

    async def test_an_update_for_someone_elses_order_is_ignored(self):
        """A manual order placed from the Binance app, or one from before a
        restart. Reconciliation owns that case; guessing here would be worse."""
        engine, _ = build()
        await engine.open_position(intent())
        await engine.on_order_update(
            {
                "client_order_id": "placed-by-hand",
                "symbol": "TESTUSDT",
                "status": "FILLED",
                "filled_quantity": 99.0,
                "average_price": 50.0,
            }
        )
        assert "TESTUSDT" in engine.positions
        assert engine.trades == []

    async def test_the_entry_fill_event_does_not_close_the_position(self):
        """The entry's own FILLED echo arrives on the same stream as the stop's."""
        engine, _ = build()
        request = intent()
        await engine.open_position(request)
        await engine.on_order_update(
            {
                "client_order_id": request.client_order_id,
                "symbol": "TESTUSDT",
                "status": "FILLED",
                "filled_quantity": request.quantity,
                "average_price": 100.0,
            }
        )
        assert "TESTUSDT" in engine.positions
        assert engine.trades == []
