"""Domain type invariants."""

from __future__ import annotations

import pytest

from tradebot.core.types import (
    BookTicker,
    Direction,
    ExitReason,
    MarketRegime,
    OrderSide,
    OrderStatus,
    Position,
    Signal,
    Timeframe,
    Trade,
)


class TestDirection:
    def test_signs(self):
        assert Direction.LONG.sign == 1
        assert Direction.SHORT.sign == -1
        assert Direction.WAIT.sign == 0

    def test_opposite(self):
        assert Direction.LONG.opposite() is Direction.SHORT
        assert Direction.WAIT.opposite() is Direction.WAIT

    def test_order_side_derivation(self):
        assert OrderSide.for_entry(Direction.LONG) is OrderSide.BUY
        assert OrderSide.for_exit(Direction.LONG) is OrderSide.SELL
        assert OrderSide.for_exit(Direction.SHORT) is OrderSide.BUY

    def test_wait_cannot_become_an_order(self):
        with pytest.raises(ValueError):
            OrderSide.for_entry(Direction.WAIT)


class TestSignalValidation:
    def _signal(self, direction, entry, stop, target):
        return Signal(
            symbol="TESTUSDT",
            strategy="unit",
            direction=direction,
            confidence=70.0,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            timeframe="3m",
            signal_timestamp=0,
        )

    def test_well_formed_long_passes(self):
        assert self._signal(Direction.LONG, 100, 99, 103).validate() == []

    def test_long_stop_above_entry_is_rejected(self):
        problems = self._signal(Direction.LONG, 100, 101, 103).validate()
        assert any("stop" in p for p in problems)

    def test_long_target_below_entry_is_rejected(self):
        problems = self._signal(Direction.LONG, 100, 99, 98).validate()
        assert any("target" in p for p in problems)

    def test_short_stop_below_entry_is_rejected(self):
        problems = self._signal(Direction.SHORT, 100, 99, 97).validate()
        assert any("stop" in p for p in problems)

    def test_wait_signal_needs_no_levels(self):
        assert self._signal(Direction.WAIT, 0, 0, 0).validate() == []

    def test_risk_reward_computation(self):
        assert self._signal(Direction.LONG, 100, 99, 103).risk_reward == pytest.approx(3.0)

    def test_risk_reward_with_zero_stop_distance_does_not_divide_by_zero(self):
        assert self._signal(Direction.LONG, 100, 100, 103).risk_reward == 0.0

    def test_confidence_out_of_range_is_flagged(self):
        s = Signal("X", "u", Direction.LONG, 150.0, 100, 99, 103, "3m", 0)
        assert any("confidence" in p for p in s.validate())


class TestPosition:
    def _position(self, direction=Direction.LONG):
        return Position(
            position_id="p1",
            symbol="TESTUSDT",
            direction=direction,
            quantity=2.0,
            entry_price=100.0,
            leverage=3,
            stop_loss=98.0,
            take_profit=104.0,
            strategy="unit",
            regime=MarketRegime.STRONG_TREND,
            opened_at=0,
            entry_notional=200.0,
            initial_risk=4.0,
            initial_stop=98.0,
        )

    def test_long_pnl_sign(self):
        assert self._position().unrealized_pnl(101.0) == pytest.approx(2.0)
        assert self._position().unrealized_pnl(99.0) == pytest.approx(-2.0)

    def test_short_pnl_sign(self):
        p = self._position(Direction.SHORT)
        assert p.unrealized_pnl(99.0) == pytest.approx(2.0)
        assert p.unrealized_pnl(101.0) == pytest.approx(-2.0)

    def test_r_multiple_uses_initial_risk(self):
        assert self._position().r_multiple(104.0) == pytest.approx(2.0)

    def test_r_multiple_without_recorded_risk_is_zero_not_infinite(self):
        p = self._position()
        p.initial_risk = 0.0
        assert p.r_multiple(104.0) == 0.0

    def test_margin_uses_leverage(self):
        assert self._position().margin(100.0) == pytest.approx(200.0 / 3)

    def test_stop_detection_long(self):
        p = self._position()
        assert p.is_stop_hit(low=97.9, high=100.0)
        assert not p.is_stop_hit(low=98.1, high=100.0)

    def test_stop_detection_short(self):
        p = self._position(Direction.SHORT)
        p.stop_loss = 102.0
        assert p.is_stop_hit(low=100.0, high=102.5)
        assert not p.is_stop_hit(low=100.0, high=101.5)

    def test_duration_is_never_negative(self):
        assert self._position().duration_sec(-1000) == 0.0


class TestTrade:
    def _trade(self, net):
        return Trade(
            trade_id="t1",
            symbol="TESTUSDT",
            strategy="unit",
            direction=Direction.LONG,
            entry_price=100.0,
            exit_price=101.0,
            quantity=1.0,
            leverage=2,
            stop_loss=99.0,
            take_profit=103.0,
            opened_at=0,
            closed_at=501_000,
            gross_pnl=1.0,
            fees=0.08,
            funding=0.0,
            slippage_cost=0.02,
            net_pnl=net,
            exit_reason=ExitReason.TAKE_PROFIT,
            regime=MarketRegime.STRONG_TREND,
            entry_notional=100.0,
            initial_risk=1.0,
        )

    def test_duration(self):
        assert self._trade(0.9).duration_sec == pytest.approx(501.0)

    def test_win_classification(self):
        assert self._trade(0.9).is_win
        assert not self._trade(-0.1).is_win
        assert not self._trade(0.0).is_win

    def test_r_multiple(self):
        assert self._trade(0.9).r_multiple == pytest.approx(0.9)


class TestBookTicker:
    def test_spread_and_imbalance(self):
        b = BookTicker("X", 99.9, 10.0, 100.1, 5.0, 0)
        assert b.mid == pytest.approx(100.0)
        assert b.spread_bps == pytest.approx(20.0)
        assert b.imbalance == pytest.approx(1 / 3)

    def test_empty_book_does_not_divide_by_zero(self):
        b = BookTicker("X", 0.0, 0.0, 0.0, 0.0, 0)
        assert b.imbalance == 0.0
        assert b.spread_bps == float("inf")


class TestEnums:
    def test_timeframe_seconds(self):
        assert Timeframe.M1.seconds == 60
        assert Timeframe.M15.seconds == 900
        assert Timeframe.H1.milliseconds == 3_600_000

    def test_panic_blocks_entries(self):
        assert MarketRegime.PANIC.blocks_entries
        assert not MarketRegime.STRONG_TREND.blocks_entries

    def test_terminal_and_open_statuses_are_disjoint(self):
        for status in OrderStatus:
            assert not (status.is_terminal and status.is_open)
