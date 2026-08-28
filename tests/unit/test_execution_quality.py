"""Execution quality and the cost-model feedback loop.

Every input to the edge filter is an estimate. If those estimates are
systematically optimistic, the filter approves trades that were never
profitable, and the losses read as strategy failure rather than as the
measurement error they are. This is the measurement.
"""

from __future__ import annotations

import pytest

from tradebot.core.config import EdgeConfig
from tradebot.core.types import Direction
from tradebot.execution.quality import ExecutionQuality, ExecutionRecord
from tradebot.market.microstructure import CostModel, LiquiditySnapshot


def record(
    symbol: str = "BTCUSDT",
    direction: Direction = Direction.LONG,
    is_entry: bool = True,
    reference: float = 100.0,
    fill: float = 100.05,
    expected_cost: float = 0.0002,
    order_type: str = "MARKET",
) -> ExecutionRecord:
    return ExecutionRecord(
        symbol=symbol,
        direction=direction,
        order_type=order_type,
        is_entry=is_entry,
        reference_price=reference,
        fill_price=fill,
        quantity=1.0,
        expected_cost=expected_cost,
    )


class TestSlippageSign:
    def test_a_long_entry_filled_higher_is_adverse(self) -> None:
        assert record(fill=100.05).slippage == pytest.approx(0.0005)

    def test_a_long_entry_filled_lower_is_favourable(self) -> None:
        """A better-than-quoted fill is not a cost to compensate for. An
        unsigned magnitude could not tell the two apart."""
        assert record(fill=99.95).slippage == pytest.approx(-0.0005)

    def test_a_short_entry_filled_lower_is_adverse(self) -> None:
        """A short entry SELLS, so a lower price is the bad direction."""
        entry = record(direction=Direction.SHORT, fill=99.95)
        assert entry.slippage == pytest.approx(0.0005)

    def test_a_long_exit_sells_so_lower_is_adverse(self) -> None:
        """The exit leg trades the other way round; getting this wrong would
        make every profitable exit look like a cost."""
        exit_leg = record(direction=Direction.LONG, is_entry=False, fill=99.95)
        assert exit_leg.is_buy is False
        assert exit_leg.slippage == pytest.approx(0.0005)

    def test_a_short_exit_buys(self) -> None:
        exit_leg = record(direction=Direction.SHORT, is_entry=False, fill=100.05)
        assert exit_leg.is_buy is True
        assert exit_leg.slippage == pytest.approx(0.0005)

    def test_a_missing_reference_price_is_zero_not_infinite(self) -> None:
        assert record(reference=0.0).slippage == 0.0

    def test_cost_error_is_actual_minus_predicted(self) -> None:
        under = record(fill=100.05, expected_cost=0.0002)  # 5 bps actual, 2 predicted
        assert under.cost_error == pytest.approx(0.0003)


class TestCalibration:
    def test_no_adjustment_before_the_minimum_sample(self) -> None:
        """Three fills are not evidence of a bias; reacting to them chases noise."""
        quality = ExecutionQuality(min_samples=10)
        for _ in range(9):
            quality.record(record(fill=100.10, expected_cost=0.0001))
        assert quality.is_calibrated() is False
        assert quality.slippage_adjustment() == 0.0

    def test_a_persistent_under_estimate_produces_an_adjustment(self) -> None:
        quality = ExecutionQuality(min_samples=10)
        for _ in range(12):
            # 10 bps actual against a 2 bps prediction: 8 bps of bias.
            quality.record(record(fill=100.10, expected_cost=0.0002))
        assert quality.is_calibrated() is True
        assert quality.slippage_adjustment() == pytest.approx(0.0008, abs=1e-6)

    def test_the_adjustment_is_never_negative(self) -> None:
        """A run of lucky fills must not make the edge filter more permissive."""
        quality = ExecutionQuality(min_samples=5)
        for _ in range(10):
            quality.record(record(fill=99.90, expected_cost=0.0010))
        assert quality.slippage_adjustment() == 0.0

    def test_the_adjustment_is_capped(self) -> None:
        """One pathological session must not make the model so pessimistic that
        nothing ever trades again."""
        quality = ExecutionQuality(min_samples=5, max_adjustment=0.002)
        for _ in range(10):
            quality.record(record(fill=110.0, expected_cost=0.0))  # 10% slippage
        assert quality.slippage_adjustment() == 0.002

    def test_one_outlier_does_not_move_the_median(self) -> None:
        """A single news-spike fill would drag a mean estimate for hours."""
        quality = ExecutionQuality(min_samples=5)
        for _ in range(10):
            quality.record(record(fill=100.02, expected_cost=0.0002))
        baseline = quality.slippage_adjustment()
        quality.record(record(fill=140.0, expected_cost=0.0002))
        assert quality.slippage_adjustment() == pytest.approx(baseline, abs=1e-5)

    def test_per_symbol_calibration_is_independent(self) -> None:
        quality = ExecutionQuality(min_samples=5)
        for _ in range(6):
            quality.record(record(symbol="CLEANUSDT", fill=100.01, expected_cost=0.0001))
            quality.record(record(symbol="NASTYUSDT", fill=100.20, expected_cost=0.0001))
        assert quality.slippage_adjustment("NASTYUSDT") > quality.slippage_adjustment("CLEANUSDT")

    def test_an_unseen_symbol_gets_no_adjustment(self) -> None:
        quality = ExecutionQuality(min_samples=5)
        assert quality.slippage_adjustment("NEVERUSDT") == 0.0
        assert quality.expected_slippage("NEVERUSDT") == 0.0


class TestReporting:
    def test_junk_records_are_ignored(self) -> None:
        quality = ExecutionQuality()
        quality.record(record(reference=0.0))
        quality.record(record(fill=0.0))
        assert quality.recorded == 0

    def test_worst_symbols_are_ranked(self) -> None:
        quality = ExecutionQuality(min_samples=3)
        for _ in range(4):
            quality.record(record(symbol="GOODUSDT", fill=100.01))
            quality.record(record(symbol="BADUSDT", fill=100.30))
        worst = quality.worst_symbols()
        assert worst[0]["symbol"] == "BADUSDT"
        assert worst[0]["median_slippage_bps"] > worst[-1]["median_slippage_bps"]

    def test_adverse_rate_is_tracked(self) -> None:
        quality = ExecutionQuality()
        for _ in range(3):
            quality.record(record(fill=100.05))  # adverse
        quality.record(record(fill=99.95))  # favourable
        row = next(r for r in quality.report() if r["symbol"] == "BTCUSDT")
        assert row["fills"] == 4
        assert row["adverse_rate"] == pytest.approx(0.75)

    def test_stats_break_down_by_order_type(self) -> None:
        quality = ExecutionQuality()
        quality.record(record(order_type="MARKET", fill=100.10))
        quality.record(record(order_type="LIMIT", fill=100.01))
        by_type = quality.stats()["by_order_type"]
        assert by_type["MARKET"] > by_type["LIMIT"]


class TestCostModelFeedback:
    def test_the_adjustment_raises_the_estimated_cost(self) -> None:
        model = CostModel(EdgeConfig())
        liquidity = LiquiditySnapshot("BTCUSDT", 2.0, 1e5, 1e5, 0.0, 1e7)

        before = model.estimate(Direction.LONG, 1000.0, liquidity).total
        model.set_slippage_adjustment(0.0008)
        after = model.estimate(Direction.LONG, 1000.0, liquidity).total

        # Both legs pay it.
        assert after == pytest.approx(before + 0.0016, abs=1e-9)

    def test_a_symbol_adjustment_beats_the_global_one(self) -> None:
        model = CostModel(EdgeConfig())
        model.set_slippage_adjustment(0.0002)
        model.set_slippage_adjustment(0.0009, "NASTYUSDT")
        assert model.slippage_adjustment("NASTYUSDT") == 0.0009
        assert model.slippage_adjustment("OTHERUSDT") == 0.0002

    def test_a_negative_adjustment_is_clamped_to_zero(self) -> None:
        """The loop may make the model more careful, never less."""
        model = CostModel(EdgeConfig())
        model.set_slippage_adjustment(-0.005)
        assert model.slippage_adjustment() == 0.0

    def test_an_uncalibrated_model_behaves_exactly_as_before(self) -> None:
        model = CostModel(EdgeConfig())
        liquidity = LiquiditySnapshot("BTCUSDT", 2.0, 1e5, 1e5, 0.0, 1e7)
        assert model.slippage_adjustment("BTCUSDT") == 0.0
        assert model.estimate(Direction.LONG, 1000.0, liquidity).slippage > 0
