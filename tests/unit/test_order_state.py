"""The order state machine.

Three properties matter more than the enum itself, and each corresponds to a
way a real bot loses track of a real position:

1. A terminal state is final — a late update cannot resurrect a filled order.
2. Filled quantity only moves forward — out-of-order updates cannot shrink it.
3. INDETERMINATE is recorded, not guessed — a timed-out submission may or may
   not have reached the exchange.
"""

from __future__ import annotations

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.types import OrderStatus
from tradebot.execution.state import (
    TRANSITIONS,
    OrderState,
    OrderTracker,
    state_for_status,
)


@pytest.fixture
def tracker(clock: VirtualClock) -> OrderTracker:
    return OrderTracker(clock)


def new_entry(tracker: OrderTracker, cid: str = "tb_e_1") -> str:
    tracker.create(cid, "BTCUSDT", "BUY", 0.01, "ENTRY")
    return cid


class TestStateClassification:
    def test_terminal_states(self) -> None:
        terminal = {s for s in OrderState if s.is_terminal}
        assert terminal == {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        }

    def test_open_states_are_the_ones_that_can_still_fill(self) -> None:
        assert OrderState.ACKNOWLEDGED.is_open
        assert OrderState.PARTIALLY_FILLED.is_open
        assert OrderState.CANCEL_REQUESTED.is_open  # may still fill mid-cancel
        assert not OrderState.FILLED.is_open
        assert not OrderState.CREATED.is_open  # not sent, so no exposure

    def test_indeterminate_is_neither_open_nor_terminal(self) -> None:
        """It is genuinely unknown, and that is the point: it must not be
        collapsed into either bucket, because both would be a guess."""
        assert not OrderState.INDETERMINATE.is_open
        assert not OrderState.INDETERMINATE.is_terminal
        assert OrderState.INDETERMINATE.needs_resolution

    def test_every_state_has_a_transition_entry(self) -> None:
        assert set(TRANSITIONS) == set(OrderState)

    def test_terminal_states_have_no_exits(self) -> None:
        for state in OrderState:
            if state.is_terminal:
                assert TRANSITIONS[state] == frozenset(), state


class TestExchangeStatusMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("NEW", OrderState.ACKNOWLEDGED),
            ("PARTIALLY_FILLED", OrderState.PARTIALLY_FILLED),
            ("FILLED", OrderState.FILLED),
            ("CANCELED", OrderState.CANCELLED),  # Binance's single-L spelling
            ("CANCELLED", OrderState.CANCELLED),
            ("REJECTED", OrderState.REJECTED),
            ("EXPIRED", OrderState.EXPIRED),
            ("EXPIRED_IN_MATCH", OrderState.EXPIRED),
        ],
    )
    def test_documented_statuses(self, status: str, expected: OrderState) -> None:
        assert state_for_status(status) == expected

    def test_accepts_the_enum_too(self) -> None:
        assert state_for_status(OrderStatus.FILLED) is OrderState.FILLED

    def test_unknown_status_is_none_not_a_guess(self) -> None:
        assert state_for_status("SOMETHING_NEW") is None


class TestTransitions:
    def test_the_happy_path(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        for state in (
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
        ):
            assert tracker.transition(cid, state) is True
        order = tracker.get(cid)
        assert order is not None
        assert order.state is OrderState.FILLED
        assert [t.to_state for t in order.history][-1] is OrderState.FILLED

    def test_a_market_order_may_fill_without_an_acknowledgement(
        self, tracker: OrderTracker
    ) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        assert tracker.transition(cid, OrderState.FILLED) is True

    def test_terminal_is_final(self, tracker: OrderTracker) -> None:
        """Rule 1. REST polling and the user stream race constantly, so a stale
        NEW after a FILLED is expected traffic, not corruption."""
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.transition(cid, OrderState.FILLED)

        assert tracker.transition(cid, OrderState.ACKNOWLEDGED, "rest_poll") is False
        order = tracker.get(cid)
        assert order is not None
        assert order.state is OrderState.FILLED
        assert tracker.late_updates == 1
        assert tracker.illegal_transitions == 0  # a late update is not an error

    def test_an_illegal_jump_is_refused_and_counted(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        # CREATED -> FILLED skips the submission entirely: our model would be
        # claiming a fill for an order that was never sent.
        assert tracker.transition(cid, OrderState.FILLED) is False
        assert tracker.illegal_transitions == 1
        order = tracker.get(cid)
        assert order is not None
        assert order.state is OrderState.CREATED

    def test_repeated_partial_fills_are_all_recorded(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.transition(cid, OrderState.PARTIALLY_FILLED)
        tracker.transition(cid, OrderState.PARTIALLY_FILLED)
        order = tracker.get(cid)
        assert order is not None
        assert sum(1 for t in order.history if t.to_state is OrderState.PARTIALLY_FILLED) == 2

    def test_duplicate_non_fill_updates_are_idempotent(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.transition(cid, OrderState.ACKNOWLEDGED)
        before = len(tracker.get(cid).history)  # type: ignore[union-attr]
        assert tracker.transition(cid, OrderState.ACKNOWLEDGED) is True
        assert len(tracker.get(cid).history) == before  # type: ignore[union-attr]

    def test_an_update_for_an_unknown_order_is_refused(self, tracker: OrderTracker) -> None:
        assert tracker.transition("never-sent", OrderState.FILLED) is False
        assert tracker.unknown_orders == 1

    def test_a_cancel_that_loses_the_race_still_fills(self, tracker: OrderTracker) -> None:
        """Cancelling is a request, not a result: the order may fill first."""
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.transition(cid, OrderState.ACKNOWLEDGED)
        tracker.transition(cid, OrderState.CANCEL_REQUESTED)
        assert tracker.transition(cid, OrderState.FILLED) is True


class TestIndeterminateOrders:
    def test_a_timed_out_submission_is_recorded_as_unknown(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        assert tracker.transition(cid, OrderState.INDETERMINATE, "execution", "timeout") is True
        assert [o.client_order_id for o in tracker.indeterminate()] == [cid]

    @pytest.mark.parametrize(
        "outcome",
        [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED],
    )
    def test_reconciliation_can_resolve_it_any_way(
        self, tracker: OrderTracker, outcome: OrderState
    ) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.transition(cid, OrderState.INDETERMINATE)
        assert tracker.resolve_indeterminate(cid, outcome, "reconciled") is True
        assert tracker.indeterminate() == []
        assert tracker.get(cid).state is outcome  # type: ignore[union-attr]


class TestExchangeUpdates:
    def test_a_fill_update_records_price_and_commission(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        assert tracker.apply_exchange_update(
            {
                "client_order_id": cid,
                "exchange_order_id": "9911",
                "status": "FILLED",
                "filled_quantity": 0.01,
                "average_price": 60000.5,
                "commission": 0.24,
                "realized_pnl": 0.0,
            }
        )
        order = tracker.get(cid)
        assert order is not None
        assert order.state is OrderState.FILLED
        assert order.filled_quantity == 0.01
        assert order.average_price == 60000.5
        assert order.commission == 0.24
        assert order.exchange_order_id == "9911"
        assert order.remaining == 0.0
        assert order.fill_fraction == 1.0

    def test_partial_then_full(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.apply_exchange_update(
            {
                "client_order_id": cid,
                "status": "PARTIALLY_FILLED",
                "filled_quantity": 0.004,
                "average_price": 60000.0,
                "commission": 0.1,
            }
        )
        assert tracker.get(cid).remaining == pytest.approx(0.006)  # type: ignore[union-attr]
        tracker.apply_exchange_update(
            {
                "client_order_id": cid,
                "status": "FILLED",
                "filled_quantity": 0.01,
                "average_price": 60000.2,
                "commission": 0.14,
            }
        )
        order = tracker.get(cid)
        assert order is not None
        assert order.state is OrderState.FILLED
        assert order.commission == pytest.approx(0.24)  # commissions accumulate

    def test_an_out_of_order_update_cannot_shrink_the_fill(self, tracker: OrderTracker) -> None:
        """Rule 2. Two updates for the same order can arrive in either order;
        letting the older one win would make the engine believe it holds less
        than it does — and size the exit accordingly."""
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.apply_exchange_update(
            {
                "client_order_id": cid,
                "status": "FILLED",
                "filled_quantity": 0.01,
                "average_price": 60000.0,
            }
        )
        tracker.apply_exchange_update(
            {
                "client_order_id": cid,
                "status": "PARTIALLY_FILLED",
                "filled_quantity": 0.004,
                "average_price": 59000.0,
            }
        )
        order = tracker.get(cid)
        assert order is not None
        assert order.filled_quantity == 0.01
        assert order.average_price == 60000.0
        assert tracker.late_updates >= 1

    def test_an_unmapped_status_changes_nothing(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        assert (
            tracker.apply_exchange_update({"client_order_id": cid, "status": "SOMETHING_NEW"})
            is False
        )
        assert tracker.get(cid).state is OrderState.SUBMITTED  # type: ignore[union-attr]

    def test_updates_without_an_id_are_ignored(self, tracker: OrderTracker) -> None:
        assert tracker.apply_exchange_update({"status": "FILLED"}) is False


class TestBookkeeping:
    def test_open_orders_are_listed_per_symbol(self, tracker: OrderTracker) -> None:
        tracker.create("a", "BTCUSDT", "BUY", 1.0)
        tracker.create("b", "ETHUSDT", "BUY", 1.0)
        for cid in ("a", "b"):
            tracker.transition(cid, OrderState.SUBMITTED)
            tracker.transition(cid, OrderState.ACKNOWLEDGED)
        assert [o.client_order_id for o in tracker.open_for("BTCUSDT")] == ["a"]
        assert len(tracker.open_orders()) == 2

    def test_completed_orders_do_not_accumulate_forever(self, clock: VirtualClock) -> None:
        """A bot running for weeks submits thousands of orders. Holding every
        one is a leak that only shows up in production."""
        bounded = OrderTracker(clock, keep_terminal=5)
        for index in range(20):
            cid = f"o{index}"
            bounded.create(cid, "BTCUSDT", "BUY", 1.0)
            bounded.transition(cid, OrderState.SUBMITTED)
            bounded.transition(cid, OrderState.FILLED)
        assert len(bounded) == 5
        assert bounded.get("o19") is not None
        assert bounded.get("o0") is None

    def test_creating_the_same_id_twice_returns_the_same_order(self, tracker: OrderTracker) -> None:
        """Client order ids are deterministic so retries are idempotent; the
        tracker must not fork a second record for a retried submission."""
        first = tracker.create("dup", "BTCUSDT", "BUY", 1.0)
        tracker.transition("dup", OrderState.SUBMITTED)
        second = tracker.create("dup", "BTCUSDT", "BUY", 1.0)
        assert second is first
        assert second.state is OrderState.SUBMITTED

    def test_stats_and_report(self, tracker: OrderTracker) -> None:
        cid = new_entry(tracker)
        tracker.transition(cid, OrderState.SUBMITTED)
        tracker.transition(cid, OrderState.ACKNOWLEDGED)
        stats = tracker.stats()
        assert stats["tracked"] == 1
        assert stats["open"] == 1
        assert stats["by_state"]["ACKNOWLEDGED"] == 1
        assert tracker.report()[0]["client_order_id"] == cid
