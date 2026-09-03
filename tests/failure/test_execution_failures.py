"""Execution and reconciliation under failure.

Everything here is a scenario that WILL happen in production. The tests assert
the system's response is safe, not merely that it does not crash.

The governing rules:

* a timed-out order is never blindly re-sent
* a position is never left without a stop
* an exit is never blocked, by anything
* the exchange is the source of truth
"""

from __future__ import annotations

import asyncio

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.config import load_tunables
from tradebot.core.errors import (
    ExchangeError,
    FilterViolationError,
    NetworkError,
    TimeoutError_,
)
from tradebot.core.events import EventBus
from tradebot.core.types import (
    Direction,
    ExitReason,
    MarketRegime,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
)
from tradebot.execution.engine import ExecutionEngine, positions_without_stops
from tradebot.execution.reconciliation import Reconciler

from ..conftest import REPO_ROOT
from ..fakes import FakeGateway, make_symbol_info, position_for

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


def build(gateway: FakeGateway | None = None) -> tuple[ExecutionEngine, FakeGateway]:
    gw = gateway or FakeGateway()
    gw.symbols.setdefault("TESTUSDT", make_symbol_info("TESTUSDT", min_notional=1.0))
    engine = ExecutionEngine(CONFIG, gw, EventBus(), VirtualClock(1_700_000_000_000))
    return engine, gw


# --------------------------------------------------------------------------- #
class TestOrderSubmissionFailures:
    async def test_a_rejected_order_opens_no_position(self):
        engine, gateway = build()
        gateway.reject_next_order = True
        result = await engine.open_position(intent())
        assert not result.success
        assert engine.positions == {}
        assert engine.rejected == 1

    async def test_a_filter_violation_is_reported_not_retried(self):
        engine, gateway = build()

        async def refuse(_intent):
            raise FilterViolationError("min notional", symbol="TESTUSDT")

        gateway.place_order = refuse
        result = await engine.open_position(intent())
        assert not result.success
        assert "filter violation" in result.reason

    async def test_an_indeterminate_submission_blocks_entries(self):
        """We do not know whether we have a position. Stop trading until we do."""
        engine, gateway = build()

        async def timeout(_intent):
            raise TimeoutError_("order timed out; state INDETERMINATE")

        gateway.place_order = timeout
        result = await engine.open_position(intent())
        assert not result.success
        assert engine.entries_blocked
        assert "indeterminate" in engine.entries_blocked_reason.lower()

    async def test_a_network_failure_on_submit_also_blocks_entries(self):
        engine, gateway = build()

        async def fail(_intent):
            raise NetworkError("connection reset")

        gateway.place_order = fail
        await engine.open_position(intent())
        assert engine.entries_blocked

    async def test_an_unknown_status_blocks_entries(self):
        engine, gateway = build()

        async def unknown(order_intent):
            return Order(
                client_order_id=order_intent.client_order_id,
                symbol=order_intent.symbol,
                side=order_intent.side,
                order_type=order_intent.order_type,
                quantity=order_intent.quantity,
                status=OrderStatus.UNKNOWN,
            )

        gateway.place_order = unknown
        result = await engine.open_position(intent())
        assert not result.success
        assert engine.entries_blocked


class TestStopLossIsMandatory:
    async def test_a_position_that_cannot_be_protected_is_closed(self):
        """The rule with no exceptions: no position without a stop."""
        engine, gateway = build()

        async def refuse_stop(**_kwargs):
            raise ExchangeError("stop rejected by the exchange")

        gateway.place_protective_order = refuse_stop
        result = await engine.open_position(intent())

        assert not result.success
        assert "protective stop" in result.reason
        assert engine.positions == {}, "an unprotected position must not persist"
        assert engine.unprotected_closures == 1
        assert "TESTUSDT" in gateway.closed

    async def test_a_successful_entry_places_a_stop(self):
        engine, gateway = build()
        result = await engine.open_position(intent())
        assert result.success
        assert result.protected
        assert result.position.stop_order_id
        stop_orders = [p for p in gateway.protective if p["type"] is OrderType.STOP_MARKET]
        assert len(stop_orders) == 1
        assert stop_orders[0]["stop_price"] == pytest.approx(99.5)

    async def test_a_missing_take_profit_does_not_block_the_trade(self):
        """A missing STOP is fatal. A missing target is not."""
        engine, gateway = build()
        calls = {"n": 0}
        original = gateway.place_protective_order

        async def stop_only(**kwargs):
            calls["n"] += 1
            if kwargs["order_type"] is OrderType.TAKE_PROFIT_MARKET:
                raise ExchangeError("take profit rejected")
            return await original(**kwargs)

        gateway.place_protective_order = stop_only
        result = await engine.open_position(intent())
        assert result.success
        assert result.position.stop_order_id
        assert result.position.take_profit_order_id is None

    async def test_a_failed_stop_update_closes_the_position(self):
        """Better flat than unprotected."""
        engine, gateway = build()
        await engine.open_position(intent())

        async def refuse(**_kwargs):
            raise ExchangeError("stop replacement rejected")

        gateway.place_protective_order = refuse
        assert not await engine.update_stop("TESTUSDT", 99.8)
        assert "TESTUSDT" not in engine.positions

    async def test_unprotected_positions_are_detectable(self):
        positions = {"A": position_for("A"), "B": position_for("B")}
        orders = [
            Order(
                client_order_id="sl_a",
                symbol="A",
                side=OrderSide.SELL,
                order_type=OrderType.STOP_MARKET,
                quantity=1.0,
                status=OrderStatus.NEW,
            ),
        ]
        unprotected = positions_without_stops(positions, orders)
        assert [u.symbol for u in unprotected] == ["B"]

    async def test_a_filled_stop_no_longer_counts_as_protection(self):
        """A stop that already filled protects nothing."""
        positions = {"A": position_for("A")}
        orders = [
            Order(
                client_order_id="sl_a",
                symbol="A",
                side=OrderSide.SELL,
                order_type=OrderType.STOP_MARKET,
                quantity=1.0,
                status=OrderStatus.FILLED,
            ),
        ]
        assert [u.symbol for u in positions_without_stops(positions, orders)] == ["A"]


class TestSlippageProtection:
    async def test_fill_above_intent_margin_cap_is_closed(self):
        engine, gateway = build()
        gateway.fill_price_override["TESTUSDT"] = 100.05
        capped = intent(quantity=0.1, entry=100.0)
        capped.metadata["margin_per_trade_cap"] = 5.0
        result = await engine.open_position(capped)
        assert not result.success
        assert "margin" in result.reason
        assert engine.positions == {}

    async def test_a_fill_far_from_the_decision_price_is_closed(self):
        """The edge that justified the trade was priced away before we filled."""
        engine, gateway = build()
        gateway.fill_price_override["TESTUSDT"] = 105.0  # 5% away
        result = await engine.open_position(intent(entry=100.0))
        assert not result.success
        assert "slippage" in result.reason
        assert engine.positions == {}

    async def test_a_normal_fill_is_accepted(self):
        engine, gateway = build()
        gateway.fill_price_override["TESTUSDT"] = 100.05  # 5 bps
        result = await engine.open_position(intent(entry=100.0))
        assert result.success
        assert result.slippage == pytest.approx(0.0005, abs=1e-6)


class TestRaceConditions:
    async def test_concurrent_intents_for_one_symbol_open_one_position(self):
        """Two signals in the same cycle must not double up."""
        engine, gateway = build()
        results = await asyncio.gather(
            engine.open_position(intent()),
            engine.open_position(intent()),
            engine.open_position(intent()),
        )
        assert sum(1 for r in results if r.success) == 1
        assert len(engine.positions) == 1
        assert len(gateway.placed) == 1

    async def test_a_second_entry_is_refused_while_one_is_held(self):
        engine, _ = build()
        assert (await engine.open_position(intent())).success
        second = await engine.open_position(intent())
        assert not second.success
        assert "already holding" in second.reason

    async def test_the_client_order_id_is_deterministic(self):
        """This is what makes a retry collide instead of double-filling."""
        assert intent().client_order_id == intent().client_order_id
        assert intent().client_order_id == "tb_abc123def456"


class TestPartialFills:
    async def test_the_position_tracks_the_actual_filled_quantity(self):
        """A stop sized for a position that does not exist is not protection."""
        engine, gateway = build()

        async def partial(order_intent):
            return Order(
                client_order_id=order_intent.client_order_id,
                symbol=order_intent.symbol,
                side=order_intent.side,
                order_type=order_intent.order_type,
                quantity=order_intent.quantity,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_quantity=order_intent.quantity * 0.4,
                average_price=100.0,
                exchange_order_id="x1",
            )

        gateway.place_order = partial
        result = await engine.open_position(intent(quantity=1.0))
        assert result.success
        assert result.position.quantity == pytest.approx(0.4)
        assert result.position.metadata["partial_fill"] is True
        assert gateway.protective[0]["quantity"] == pytest.approx(0.4)

    async def test_risk_is_recomputed_from_the_actual_fill(self):
        engine, gateway = build()

        async def partial(order_intent):
            return Order(
                client_order_id=order_intent.client_order_id,
                symbol=order_intent.symbol,
                side=order_intent.side,
                order_type=order_intent.order_type,
                quantity=order_intent.quantity,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_quantity=0.5,
                average_price=100.0,
            )

        gateway.place_order = partial
        result = await engine.open_position(intent(quantity=1.0, entry=100.0))
        # stop at 99.5 -> 0.5 distance * 0.5 filled = 0.25 at risk
        assert result.position.initial_risk == pytest.approx(0.25)


class TestExitsAreNeverBlocked:
    async def test_an_exit_works_while_entries_are_blocked(self):
        engine, _ = build()
        await engine.open_position(intent())
        engine.block_entries("kill switch tripped")

        trade = await engine.close_position("TESTUSDT", ExitReason.KILL_SWITCH)
        assert trade is not None
        assert "TESTUSDT" not in engine.positions

    async def test_close_all_flattens_everything(self):
        engine, gateway = build()
        for symbol in ("AAAUSDT", "BBBUSDT"):
            gateway.symbols[symbol] = make_symbol_info(symbol, min_notional=1.0)
            await engine.open_position(intent(symbol=symbol))
        assert len(engine.positions) == 2

        closed = await engine.close_all(ExitReason.EMERGENCY, "shutdown")
        assert len(closed) == 2
        assert engine.positions == {}

    async def test_a_failed_close_is_reported_loudly_and_keeps_the_position(self):
        """Silently dropping a position we failed to close would be catastrophic."""
        engine, gateway = build()
        await engine.open_position(intent())

        async def refuse(*_args, **_kwargs):
            raise ExchangeError("close rejected")

        gateway.close_position = refuse
        assert await engine.close_position("TESTUSDT", ExitReason.STOP_LOSS) is None
        assert "TESTUSDT" in engine.positions, (
            "a position that could not be closed must remain tracked"
        )


class TestReconciliation:
    def reconciler(self, gateway, engine):
        return Reconciler(gateway, engine, CONFIG, VirtualClock(1_700_000_000_000))

    async def test_a_clean_state_reconciles_without_change(self):
        engine, gateway = build()
        report = await self.reconciler(gateway, engine).reconcile()
        assert report.clean
        assert not engine.entries_blocked

    async def test_an_unexpected_exchange_position_is_adopted_and_protected(self):
        """The most dangerous case: a real position nothing is watching."""
        engine, gateway = build()
        gateway.positions["ETHUSDT"] = position_for(
            "ETHUSDT", Direction.LONG, quantity=0.5, entry=3000.0, stop=0.0
        )
        gateway.symbols["ETHUSDT"] = make_symbol_info("ETHUSDT", min_notional=1.0)

        report = await self.reconciler(gateway, engine).reconcile()

        assert "ETHUSDT" in report.adopted
        assert "ETHUSDT" in engine.positions
        adopted = engine.positions["ETHUSDT"]
        assert adopted.adopted
        assert adopted.stop_loss > 0, "an adopted position must be given a stop"
        assert adopted.stop_loss < adopted.entry_price
        assert any(p["symbol"] == "ETHUSDT" for p in gateway.protective)

    async def test_an_adopted_position_that_cannot_be_protected_is_closed(self):
        engine, gateway = build()
        gateway.positions["ETHUSDT"] = position_for("ETHUSDT", stop=0.0)
        gateway.symbols["ETHUSDT"] = make_symbol_info("ETHUSDT", min_notional=1.0)

        async def refuse(**_kwargs):
            raise ExchangeError("cannot place stop")

        gateway.place_protective_order = refuse
        report = await self.reconciler(gateway, engine).reconcile()
        assert report.errors
        assert "ETHUSDT" not in engine.positions

    async def test_a_phantom_local_position_is_closed(self):
        """The stop filled while we were disconnected."""
        engine, gateway = build()
        engine.positions["TESTUSDT"] = position_for("TESTUSDT")

        gateway.get_user_trades = _fills([{"side": "SELL", "qty": "1.0", "price": "98.0"}])
        report = await self.reconciler(gateway, engine).reconcile()

        assert "TESTUSDT" in report.closed_locally
        assert "TESTUSDT" not in engine.positions
        assert engine.trades
        assert engine.trades[-1].exit_price == pytest.approx(98.0)

    async def test_phantom_pnl_falls_back_safely_when_fills_are_unavailable(self):
        engine, gateway = build()
        engine.positions["TESTUSDT"] = position_for("TESTUSDT", entry=100.0)

        async def broken(*_args, **_kwargs):
            raise ExchangeError("userTrades unavailable")

        gateway.get_user_trades = broken
        report = await self.reconciler(gateway, engine).reconcile()
        assert "TESTUSDT" in report.closed_locally
        assert report.errors

    async def test_a_quantity_mismatch_trusts_the_exchange(self):
        engine, gateway = build()
        engine.positions["TESTUSDT"] = position_for("TESTUSDT", quantity=1.0)
        gateway.positions["TESTUSDT"] = position_for("TESTUSDT", quantity=0.6)
        gateway.get_user_trades = _fills([])

        report = await self.reconciler(gateway, engine).reconcile()
        assert "TESTUSDT" in report.quantity_mismatches
        assert engine.positions["TESTUSDT"].quantity == pytest.approx(0.6)

    async def test_orphan_protective_orders_are_cancelled(self):
        engine, gateway = build()
        gateway.open_orders = [
            Order(
                client_order_id="orphan_sl",
                symbol="GONEUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.STOP_MARKET,
                quantity=1.0,
                status=OrderStatus.NEW,
            ),
        ]
        report = await self.reconciler(gateway, engine).reconcile()
        assert "orphan_sl" in report.orphan_orders_cancelled

    async def test_an_unprotected_local_position_gets_a_stop(self):
        engine, gateway = build()
        position = position_for("TESTUSDT")
        engine.positions["TESTUSDT"] = position
        gateway.positions["TESTUSDT"] = position

        report = await self.reconciler(gateway, engine).reconcile()
        assert "TESTUSDT" in report.unprotected_fixed
        assert any(p["symbol"] == "TESTUSDT" for p in gateway.protective)

    async def test_entries_stay_blocked_when_reconciliation_fails(self):
        engine, gateway = build()

        async def unreachable():
            raise ExchangeError("exchange unreachable")

        gateway.get_positions = unreachable
        report = await self.reconciler(gateway, engine).reconcile()

        assert report.errors
        assert engine.entries_blocked, "unverified local state must never permit new entries"

    async def test_startup_blocks_entries_until_state_is_verified(self):
        engine, gateway = build()
        reconciler = self.reconciler(gateway, engine)
        report = await reconciler.startup()
        assert report.clean
        assert not engine.entries_blocked

    async def test_repeated_mismatches_are_counted(self):
        engine, gateway = build()
        reconciler = self.reconciler(gateway, engine)
        gateway.get_user_trades = _fills([])

        for _ in range(2):
            engine.positions["TESTUSDT"] = position_for("TESTUSDT")
            await reconciler.reconcile()
        assert reconciler.consecutive_mismatches == 2

    async def test_a_clean_run_resets_the_mismatch_counter(self):
        engine, gateway = build()
        reconciler = self.reconciler(gateway, engine)
        gateway.get_user_trades = _fills([])

        engine.positions["TESTUSDT"] = position_for("TESTUSDT")
        await reconciler.reconcile()
        assert reconciler.consecutive_mismatches == 1

        await reconciler.reconcile()
        assert reconciler.consecutive_mismatches == 0


def _fills(rows: list[dict]):
    async def _get(*_args, **_kwargs):
        return rows

    return _get
