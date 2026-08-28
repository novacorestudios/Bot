"""The realistic execution model and the three scenarios.

This is the module that decides whether an edge survives friction, which is the
question V3 exists to answer. The properties asserted here are the ones that,
if broken, would make a backtest flatter itself:

* every cost is adverse, on both sides and both legs;
* the scenarios are monotonic, so CONSERVATIVE is never cheaper than BASE;
* the simulator is deterministic given a seed, so a run can be compared with
  itself.
"""

from __future__ import annotations

import pytest

from tradebot.backtesting.execution import (
    CostBreakdown,
    ExecutionSimulator,
    Scenario,
    scenarios,
)
from tradebot.core.types import Candle, Direction

BAR = Candle(0, 100.0, 101.0, 99.0, 100.5, 1000.0, 59_999, quote_volume=100_000.0)
ALL = (Scenario.BASE, Scenario.CONSERVATIVE, Scenario.STRESS)


def sim(scenario: Scenario, seed: int = 0) -> ExecutionSimulator:
    return ExecutionSimulator(scenarios()[scenario], seed=seed)


class TestCostsAreAlwaysAdverse:
    """The single most important property. A cost model that is ever
    favourable turns a losing strategy into a winning backtest."""

    @pytest.mark.parametrize("scenario", ALL)
    def test_a_buy_fills_above_the_reference(self, scenario: Scenario) -> None:
        fill = sim(scenario).execute(100.0, 1.0, Direction.LONG, True, BAR)
        assert fill.price > 100.0

    @pytest.mark.parametrize("scenario", ALL)
    def test_a_sell_fills_below_the_reference(self, scenario: Scenario) -> None:
        fill = sim(scenario).execute(100.0, 1.0, Direction.SHORT, True, BAR)
        assert fill.price < 100.0

    @pytest.mark.parametrize("scenario", ALL)
    def test_exiting_a_long_sells_and_so_fills_below(self, scenario: Scenario) -> None:
        """The exit leg trades the other way. Getting this wrong would make
        every profitable exit look like a cost."""
        fill = sim(scenario).execute(100.0, 1.0, Direction.LONG, False, BAR)
        assert fill.price < 100.0

    @pytest.mark.parametrize("scenario", ALL)
    def test_exiting_a_short_buys_and_so_fills_above(self, scenario: Scenario) -> None:
        fill = sim(scenario).execute(100.0, 1.0, Direction.SHORT, False, BAR)
        assert fill.price > 100.0

    @pytest.mark.parametrize("scenario", ALL)
    def test_every_cost_component_is_non_negative(self, scenario: Scenario) -> None:
        fill = sim(scenario).execute(100.0, 1.0, Direction.LONG, True, BAR, 50_000)
        assert fill.fee >= 0
        assert fill.spread_cost >= 0
        assert fill.slippage_cost >= 0
        assert fill.latency_cost >= 0
        assert fill.total_cost > 0


class TestScenarioMonotonicity:
    def test_costs_increase_with_severity(self) -> None:
        costs = [
            sim(s).execute(100.0, 1.0, Direction.LONG, True, BAR, 50_000).total_cost for s in ALL
        ]
        assert costs == sorted(costs), f"scenarios are not monotonic: {costs}"
        assert costs[0] < costs[-1]

    def test_fill_prices_get_worse_with_severity(self) -> None:
        prices = [sim(s).execute(100.0, 1.0, Direction.LONG, True, BAR, 50_000).price for s in ALL]
        assert prices == sorted(prices)

    @pytest.mark.parametrize(
        "field", ["spread_bps", "slippage_bps", "latency_ms", "reject_probability"]
    )
    def test_each_assumption_increases_with_severity(self, field: str) -> None:
        values = [getattr(scenarios()[s], field) for s in ALL]
        assert values == sorted(values), f"{field} is not monotonic: {values}"

    def test_base_is_derived_from_the_configured_baseline(self) -> None:
        """CONSERVATIVE and STRESS are multiples of BASE, so changing the
        baseline moves all three together and the comparison stays meaningful."""
        built = scenarios(base_spread_bps=2.0, base_slippage_bps=3.0)
        assert built[Scenario.BASE].spread_bps == 2.0
        assert built[Scenario.CONSERVATIVE].spread_bps == 4.0
        assert built[Scenario.STRESS].spread_bps == 8.0


class TestSizeAndVolatility:
    def test_a_larger_order_pays_more_impact(self) -> None:
        simulator = sim(Scenario.BASE)
        small = simulator.execute(100.0, 1.0, Direction.LONG, True, BAR, depth_notional=50_000)
        large = simulator.execute(100.0, 100.0, Direction.LONG, True, BAR, depth_notional=50_000)
        assert large.price > small.price

    def test_a_faster_market_costs_more_latency(self) -> None:
        """The latency term: a delayed fill in a fast market is worse than in a
        quiet one, and the bar's range is the only evidence kline data holds."""
        quiet = Candle(0, 100.0, 100.05, 99.95, 100.0, 1000.0, 59_999)
        fast = Candle(0, 100.0, 108.0, 92.0, 100.0, 1000.0, 59_999)
        simulator = sim(Scenario.STRESS)
        assert (
            simulator.execute(100.0, 1.0, Direction.LONG, True, fast).latency_cost
            > simulator.execute(100.0, 1.0, Direction.LONG, True, quiet).latency_cost
        )

    def test_no_bar_means_no_latency_cost_rather_than_a_guess(self) -> None:
        fill = sim(Scenario.STRESS).execute(100.0, 1.0, Direction.LONG, True, None)
        assert fill.latency_cost == 0.0
        assert fill.filled


class TestRejectionsAndPartials:
    def test_base_never_rejects(self) -> None:
        simulator = sim(Scenario.BASE)
        for _ in range(200):
            simulator.execute(100.0, 1.0, Direction.LONG, True, BAR)
        assert simulator.rejections == 0

    def test_stress_rejects_some(self) -> None:
        simulator = sim(Scenario.STRESS, seed=1)
        for _ in range(500):
            simulator.execute(100.0, 1.0, Direction.LONG, True, BAR)
        assert simulator.rejections > 0
        assert simulator.stats()["reject_rate"] < 0.15

    def test_a_partial_fill_reduces_the_quantity(self) -> None:
        simulator = sim(Scenario.STRESS, seed=3)
        fills = [simulator.execute(100.0, 10.0, Direction.LONG, True, BAR) for _ in range(300)]
        partial = [f for f in fills if f.partial]
        assert partial, "STRESS should produce some partial fills"
        assert all(f.quantity < 10.0 for f in partial)

    def test_an_urgent_exit_is_never_rejected_or_partially_filled(self) -> None:
        """A reduce-only market order in a liquid perpetual gets done. Modelling
        an un-exitable position would understate risk in the one direction that
        matters."""
        simulator = sim(Scenario.STRESS, seed=5)
        fills = [
            simulator.execute(100.0, 1.0, Direction.LONG, False, BAR, is_exit_urgent=True)
            for _ in range(500)
        ]
        assert all(f.filled for f in fills)
        assert not any(f.rejected or f.partial for f in fills)

    def test_determinism_given_a_seed(self) -> None:
        """A backtest whose result changes between runs cannot be compared with
        itself, so the seed is part of the reproducibility record."""

        def run(seed: int) -> list[float]:
            simulator = sim(Scenario.STRESS, seed=seed)
            return [
                simulator.execute(100.0, 1.0, Direction.LONG, True, BAR).price for _ in range(50)
            ]

        assert run(11) == run(11)
        assert run(11) != run(12)


class TestNoFill:
    @pytest.mark.parametrize(("price", "quantity"), [(0.0, 1.0), (-5.0, 1.0), (100.0, 0.0)])
    def test_invalid_requests_do_not_fill(self, price: float, quantity: float) -> None:
        fill = sim(Scenario.BASE).execute(price, quantity, Direction.LONG, True, BAR)
        assert not fill.filled
        assert fill.total_cost == 0.0


class TestCostBreakdown:
    def test_net_is_gross_minus_every_component(self) -> None:
        breakdown = CostBreakdown(
            gross_pnl=10.0,
            fees=1.0,
            spread_cost=0.5,
            slippage_cost=0.25,
            latency_cost=0.15,
            funding=0.1,
            trades=4,
            hours=8.0,
        )
        assert breakdown.total_costs == pytest.approx(2.0)
        assert breakdown.net_pnl == pytest.approx(8.0)
        assert breakdown.cost_per_trade == pytest.approx(0.5)
        assert breakdown.cost_per_hour == pytest.approx(0.25)
        assert breakdown.cost_per_day == pytest.approx(6.0)

    def test_the_cost_ratio_exposes_friction_eating_the_edge(self) -> None:
        """Above 1.0 means the edge was real in price terms and did not survive
        contact with the exchange."""
        eaten = CostBreakdown(gross_pnl=1.0, fees=1.5, trades=10)
        assert eaten.cost_ratio > 1.0
        assert eaten.net_pnl < 0

    def test_zero_trades_does_not_divide_by_zero(self) -> None:
        empty = CostBreakdown()
        assert empty.cost_per_trade == 0.0
        assert empty.cost_per_hour == 0.0
        assert empty.cost_ratio == 0.0

    def test_the_table_lists_every_component_separately(self) -> None:
        """§29: a single blended cost cannot tell you which assumption a result
        depends on."""
        lines = "\n".join(CostBreakdown(gross_pnl=1.0, fees=0.1).table())
        for component in (
            "Gross PnL",
            "Fees",
            "Spread",
            "Slippage",
            "Latency",
            "Funding",
            "Net PnL",
        ):
            assert component in lines
