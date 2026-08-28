"""Realistic execution simulation for the backtester.

The question V3 exists to answer is not "do the strategies produce signals" —
they do — but "does anything survive the friction". So the friction gets its own
module, with every component priced separately and reported separately, because
a single blended "cost" number cannot tell you *which* assumption a result
depends on.

### The scenarios

Brief §41 requires the same signals executed under three sets of assumptions.
They are not three guesses at the truth; they are a **sensitivity test**. A
strategy that is profitable under BASE and destroyed under CONSERVATIVE has an
edge thinner than the error bars on the cost model, which is the same thing as
having no edge you can rely on.

| | spread | slippage | latency | rejects |
|---|---|---|---|---|
| `BASE` | as measured/configured | size-aware base | small | none |
| `CONSERVATIVE` | 2× | 2× + volatility term | 3× | occasional |
| `STRESS` | 4× | 4× + volatility term | 10× | frequent |

**None of the three is the "right" one, and results are reported side by side.**
Picking the flattering scenario is the easiest way to fabricate an edge, so the
runner refuses to report one without the others.

### Latency

Brief §17 wants `signal → decision → submission → execution` modelled. Bar data
cannot resolve sub-bar timing, so latency is applied as an **adverse price
adjustment** proportional to the bar's own realised volatility: a fill delayed
into a fast market is worse than one delayed into a quiet one, and the bar's
range is the only evidence about speed that kline data contains.

This is a model, not a measurement, and its size is a configuration choice. It
is stated here rather than buried so nobody mistakes it for the real thing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tradebot.core.logging import get_logger
from tradebot.core.mathutil import from_bps, safe_div
from tradebot.core.types import Candle, Direction

log = get_logger(__name__)


class Scenario(StrEnum):
    BASE = "BASE"
    CONSERVATIVE = "CONSERVATIVE"
    STRESS = "STRESS"


@dataclass(frozen=True, slots=True)
class ExecutionAssumptions:
    """One scenario's cost and execution parameters.

    Everything here is an assumption about the exchange, not about the strategy.
    Changing these numbers is a legitimate sensitivity test; changing strategy
    parameters to improve a result is not.
    """

    name: Scenario

    #: Half-spread paid on each marketable leg, in basis points.
    spread_bps: float
    #: Baseline slippage per leg, in basis points, before the size and
    #: volatility terms.
    slippage_bps: float
    #: Extra slippage proportional to the fraction of visible depth consumed.
    impact_coefficient: float
    #: Extra slippage proportional to the bar's own realised range. This is the
    #: latency term: a delayed fill in a fast market is worse than in a slow one.
    volatility_coefficient: float
    #: Modelled signal-to-fill delay, in milliseconds. Reported, and used to
    #: scale the volatility term.
    latency_ms: float
    #: Probability an order is rejected outright and the opportunity is lost.
    reject_probability: float = 0.0
    #: Probability a fill is partial, and the fraction filled when it is.
    partial_fill_probability: float = 0.0
    partial_fill_fraction: float = 1.0

    taker_fee: float = 0.0004
    maker_fee: float = 0.0002

    def describe(self) -> str:
        return (
            f"{self.name.value}: spread {self.spread_bps}bps, "
            f"slippage {self.slippage_bps}bps, latency {self.latency_ms:.0f}ms, "
            f"reject {self.reject_probability:.1%}, partial {self.partial_fill_probability:.1%}"
        )


def scenarios(
    base_spread_bps: float = 1.0,
    base_slippage_bps: float = 1.5,
    taker_fee: float = 0.0004,
    maker_fee: float = 0.0002,
) -> dict[Scenario, ExecutionAssumptions]:
    """The three scenarios, derived from the configured baseline.

    CONSERVATIVE and STRESS are multiples of BASE rather than independent
    numbers, so that changing the baseline moves all three together and the
    comparison stays meaningful.
    """
    return {
        Scenario.BASE: ExecutionAssumptions(
            name=Scenario.BASE,
            spread_bps=base_spread_bps,
            slippage_bps=base_slippage_bps,
            impact_coefficient=0.5,
            volatility_coefficient=0.05,
            latency_ms=250.0,
            reject_probability=0.0,
            partial_fill_probability=0.0,
            taker_fee=taker_fee,
            maker_fee=maker_fee,
        ),
        Scenario.CONSERVATIVE: ExecutionAssumptions(
            name=Scenario.CONSERVATIVE,
            spread_bps=base_spread_bps * 2.0,
            slippage_bps=base_slippage_bps * 2.0,
            impact_coefficient=1.0,
            volatility_coefficient=0.15,
            latency_ms=750.0,
            reject_probability=0.01,
            partial_fill_probability=0.05,
            partial_fill_fraction=0.6,
            taker_fee=taker_fee,
            maker_fee=maker_fee,
        ),
        Scenario.STRESS: ExecutionAssumptions(
            name=Scenario.STRESS,
            spread_bps=base_spread_bps * 4.0,
            slippage_bps=base_slippage_bps * 4.0,
            impact_coefficient=2.0,
            volatility_coefficient=0.35,
            latency_ms=2_500.0,
            reject_probability=0.05,
            partial_fill_probability=0.15,
            partial_fill_fraction=0.4,
            taker_fee=taker_fee,
            maker_fee=maker_fee,
        ),
    }


@dataclass(slots=True)
class Fill:
    """The outcome of trying to execute one leg."""

    filled: bool
    price: float = 0.0
    quantity: float = 0.0
    #: Every cost component, separately, in quote currency.
    fee: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    latency_cost: float = 0.0
    reference_price: float = 0.0
    rejected: bool = False
    partial: bool = False

    @property
    def total_cost(self) -> float:
        return self.fee + self.spread_cost + self.slippage_cost + self.latency_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "filled": self.filled,
            "price": self.price,
            "quantity": self.quantity,
            "fee": self.fee,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "latency_cost": self.latency_cost,
            "total_cost": self.total_cost,
            "reference_price": self.reference_price,
            "rejected": self.rejected,
            "partial": self.partial,
        }


class ExecutionSimulator:
    """Turns a requested trade into a fill, priced component by component.

    Deterministic given a seed. Rejections and partial fills are random events,
    and a backtest whose result changes between runs cannot be compared against
    itself — so the seed is part of the run's reproducibility record.
    """

    def __init__(self, assumptions: ExecutionAssumptions, seed: int = 0) -> None:
        self.assumptions = assumptions
        self._random = random.Random(seed)  # noqa: S311  # nosec B311 - simulation, not crypto

        self.attempts = 0
        self.rejections = 0
        self.partials = 0

    # ------------------------------------------------------------------ #
    def execute(
        self,
        reference_price: float,
        quantity: float,
        direction: Direction,
        is_entry: bool,
        bar: Candle | None = None,
        depth_notional: float = 0.0,
        is_exit_urgent: bool = False,
    ) -> Fill:
        """Execute one leg against ``reference_price``.

        ``is_exit_urgent`` marks a stop or liquidation: those are never rejected
        and never partially filled, because in reality a reduce-only market
        order in a liquid perpetual gets done — the question is only at what
        price. Modelling an un-exitable position would understate risk in the
        one direction that matters.
        """
        self.attempts += 1
        cfg = self.assumptions

        if reference_price <= 0 or quantity <= 0:
            return Fill(filled=False, reference_price=reference_price)

        # -- rejection -------------------------------------------------------
        if (
            not is_exit_urgent
            and cfg.reject_probability > 0
            and self._random.random() < cfg.reject_probability
        ):
            self.rejections += 1
            return Fill(filled=False, rejected=True, reference_price=reference_price)

        # -- partial fill ----------------------------------------------------
        partial = False
        if (
            not is_exit_urgent
            and cfg.partial_fill_probability > 0
            and self._random.random() < cfg.partial_fill_probability
        ):
            quantity *= cfg.partial_fill_fraction
            partial = True
            self.partials += 1

        # -- price components, each adverse by construction ------------------
        # A buy pays up; a sell receives less. `sign` is +1 when we are buying.
        sign = 1.0 if (direction is Direction.LONG) == is_entry else -1.0

        half_spread = from_bps(cfg.spread_bps)
        base_slip = from_bps(cfg.slippage_bps)

        notional = reference_price * quantity
        participation = safe_div(notional, depth_notional, 0.0) if depth_notional > 0 else 0.0
        impact = base_slip * cfg.impact_coefficient * participation

        # The latency term. Bar data cannot resolve sub-bar timing, so the bar's
        # own range stands in for how fast the market was moving while we waited.
        bar_range = 0.0
        if bar is not None and bar.close > 0:
            bar_range = safe_div(bar.high - bar.low, bar.close, 0.0)
        latency_scale = cfg.latency_ms / 1000.0
        latency_slip = bar_range * cfg.volatility_coefficient * latency_scale

        spread_cost = reference_price * half_spread * quantity
        slippage_cost = reference_price * (base_slip + impact) * quantity
        latency_cost = reference_price * latency_slip * quantity

        adverse = half_spread + base_slip + impact + latency_slip
        price = reference_price * (1.0 + adverse * sign)

        fee = price * quantity * cfg.taker_fee

        return Fill(
            filled=True,
            price=price,
            quantity=quantity,
            fee=fee,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            latency_cost=latency_cost,
            reference_price=reference_price,
            partial=partial,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "scenario": self.assumptions.name.value,
            "attempts": self.attempts,
            "rejections": self.rejections,
            "reject_rate": safe_div(self.rejections, self.attempts, 0.0),
            "partials": self.partials,
            "partial_rate": safe_div(self.partials, self.attempts, 0.0),
        }


@dataclass(slots=True)
class CostBreakdown:
    """Aggregated costs across a run — the §29 table."""

    gross_pnl: float = 0.0
    fees: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    latency_cost: float = 0.0
    funding: float = 0.0
    trades: int = 0
    hours: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    @property
    def total_costs(self) -> float:
        return self.fees + self.spread_cost + self.slippage_cost + self.latency_cost + self.funding

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_costs

    @property
    def cost_per_trade(self) -> float:
        return safe_div(self.total_costs, self.trades, 0.0)

    @property
    def cost_per_hour(self) -> float:
        return safe_div(self.total_costs, self.hours, 0.0)

    @property
    def cost_per_day(self) -> float:
        return self.cost_per_hour * 24.0

    @property
    def cost_ratio(self) -> float:
        """Costs as a fraction of gross profit.

        Above 1.0 means friction ate more than the strategy made — the edge was
        real in price terms and did not survive contact with the exchange.
        """
        return safe_div(self.total_costs, abs(self.gross_pnl), 0.0)

    def add_trade(self, trade: Any) -> None:
        self.trades += 1
        self.gross_pnl += trade.gross_pnl
        self.fees += trade.fees
        self.funding += trade.funding
        self.slippage_cost += trade.slippage_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "gross_pnl": round(self.gross_pnl, 6),
            "fees": round(self.fees, 6),
            "spread_cost": round(self.spread_cost, 6),
            "slippage_cost": round(self.slippage_cost, 6),
            "latency_cost": round(self.latency_cost, 6),
            "funding": round(self.funding, 6),
            "total_costs": round(self.total_costs, 6),
            "net_pnl": round(self.net_pnl, 6),
            "trades": self.trades,
            "cost_per_trade": round(self.cost_per_trade, 6),
            "cost_per_hour": round(self.cost_per_hour, 6),
            "cost_per_day": round(self.cost_per_day, 6),
            "cost_ratio": round(self.cost_ratio, 4),
        }

    def table(self) -> list[str]:
        """The §29 breakdown, as lines."""
        return [
            f"  Gross PnL          {self.gross_pnl:>12.4f}",
            f"  Fees               {-self.fees:>12.4f}",
            f"  Spread cost        {-self.spread_cost:>12.4f}",
            f"  Slippage cost      {-self.slippage_cost:>12.4f}",
            f"  Latency cost       {-self.latency_cost:>12.4f}",
            f"  Funding            {-self.funding:>12.4f}",
            f"  {'-' * 30}",
            f"  Net PnL            {self.net_pnl:>12.4f}",
            "",
            f"  Cost / trade       {self.cost_per_trade:>12.4f}",
            f"  Cost / hour        {self.cost_per_hour:>12.4f}",
            f"  Cost / day         {self.cost_per_day:>12.4f}",
            f"  Costs / gross      {self.cost_ratio:>12.2%}",
        ]
