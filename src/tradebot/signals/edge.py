"""Expected net edge.

This is the gate that decides whether a technically valid signal is worth
actually trading. It is the most consequential piece of arithmetic in the
system, and the one most likely to reject trades that "look good".

    expected_net = p·win − (1−p)·loss − fees − spread − slippage − funding

with everything expressed as a fraction of notional.

Three parts, each with a way of being wrong:

**The win probability.** Estimated from the strategy's own realised history,
shrunk toward a prior until enough trades exist. This shrinkage matters: a
strategy with 6 wins from 8 trades has an observed rate of 75 %, which is noise.
Believing it sizes the account into a strategy that has proven nothing. The
prior weight (default 40 trades) controls how much evidence is required before
the observed rate is trusted.

**The payoff.** Taken from the signal's own stop and target distances, so a
strategy that promises a 3:1 target it never reaches will show a good expected
edge and a bad realised one. That gap is exactly what the post-trade analysis
compares, and why `realised_vs_expected` exists.

**The costs.** From the cost model. At a five-minute horizon these dominate: a
round trip on a liquid perpetual costs roughly 0.11 %, so a trade targeting
0.15 % gross keeps a quarter of it.

A structural caveat worth stating plainly: p is the probability of the target
being reached *before* the stop. Estimating it from a strategy's historical win
rate is an approximation, because historical wins include time-based and
signal-flip exits that were neither. It is the best estimate available before a
trade, and the backtester measures the real distribution afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tradebot.core.config import EdgeConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp, safe_div
from tradebot.core.types import (
    AggregatedSignal,
    CostEstimate,
    EdgeEstimate,
)
from tradebot.market.microstructure import CostModel, LiquiditySnapshot

log = get_logger(__name__)


@dataclass(slots=True)
class StrategyStats:
    """Realised performance used to estimate win probability."""

    trades: int = 0
    wins: int = 0
    gross_win_sum: float = 0.0  # sum of winning returns, as fractions
    gross_loss_sum: float = 0.0  # sum of losing returns, positive fractions
    expected_edge_sum: float = 0.0
    realised_edge_sum: float = 0.0

    @property
    def observed_win_rate(self) -> float:
        return safe_div(self.wins, self.trades, 0.0)

    @property
    def average_win(self) -> float:
        return safe_div(self.gross_win_sum, self.wins, 0.0)

    @property
    def average_loss(self) -> float:
        return safe_div(self.gross_loss_sum, self.trades - self.wins, 0.0)

    def record(
        self, won: bool, gross_return: float, expected_edge: float, realised_edge: float
    ) -> None:
        self.trades += 1
        if won:
            self.wins += 1
            self.gross_win_sum += abs(gross_return)
        else:
            self.gross_loss_sum += abs(gross_return)
        self.expected_edge_sum += expected_edge
        self.realised_edge_sum += realised_edge


@dataclass(frozen=True, slots=True)
class EdgeDecision:
    """The edge estimate plus the accept/reject verdict."""

    estimate: EdgeEstimate
    accepted: bool
    threshold: float
    detail: str = ""
    inputs: dict[str, float] = field(default_factory=dict)

    @property
    def expected_net(self) -> float:
        return self.estimate.expected_net


class EdgeCalculator:
    """Estimates expected net edge and applies the minimum-edge gate."""

    def __init__(self, config: EdgeConfig, cost_model: CostModel) -> None:
        self.config = config
        self.cost_model = cost_model
        self._stats: dict[str, StrategyStats] = {}
        #: How many edge estimates used an ASSUMED win rate rather than a
        #: measured one. Reported so a backtest result cannot be mistaken for
        #: evidence when it in fact rests on assumptions.
        self.bootstrap_estimates = 0
        self.bootstrap_strategies: set[str] = set()

    # ------------------------------------------------------------------ #
    # Win probability
    # ------------------------------------------------------------------ #
    def stats_for(self, strategy: str) -> StrategyStats:
        found = self._stats.get(strategy)
        if found is None:
            found = StrategyStats()
            self._stats[strategy] = found
        return found

    def record_result(
        self,
        strategy: str,
        won: bool,
        gross_return: float,
        expected_edge: float,
        realised_edge: float,
    ) -> None:
        self.stats_for(strategy).record(won, gross_return, expected_edge, realised_edge)

    def win_probability(self, strategy: str, reward_risk: float = 0.0) -> float:
        """Shrunk estimate of P(target before stop), from realised history.

        With few trades the estimate sits at the prior and moves toward the
        observed rate as evidence accumulates. Without this, a strategy with a
        lucky first ten trades would be trusted with real size.

        When there is no history at all, the prior is additionally adjusted for
        the reward:risk being attempted — a 3:1 target is inherently reached
        less often than a 1:1 one, and a flat prior for both would make every
        ambitious target look artificially attractive.
        """
        stats = self.stats_for(strategy)
        prior = self.config.win_rate_prior
        prior_weight = self.config.win_rate_prior_weight

        if stats.trades == 0 and reward_risk > 0:
            breakeven = 1.0 / (1.0 + reward_risk)
            reference_breakeven = 1.0 / (1.0 + 1.6)  # the configured base RR
            prior = clamp(prior * (breakeven / reference_breakeven), 0.05, 0.95)

        blended = (stats.wins + prior * prior_weight) / (stats.trades + prior_weight)
        return clamp(blended, 0.01, 0.99)

    def bootstrap_probability(
        self, strategy: str, gross_win: float, gross_loss: float, cost_total: float
    ) -> float:
        """The ASSUMED win rate used to break the cold-start deadlock.

        Shrinking an unproven strategy's win rate toward the prior makes every
        candidate negative-edge at a reward:risk of 2.0 or more. Nothing is
        taken, so no evidence accumulates, so the estimate never moves: the
        system cannot start. That is correct for live trading — do not risk
        money on an unproven strategy — but fatal in a backtest, where
        measuring the win rate is the entire purpose.

        So in bootstrap mode the strategy is assumed to be *just good enough to
        be worth measuring*: its exact break-even rate after costs,

            p_breakeven = (loss + costs) / (win + loss)

        plus ``bootstrap_win_rate_margin``. As real trades arrive the assumption
        is blended out in favour of the observed rate, so the estimate migrates
        to reality rather than jumping when the trade count crosses a threshold.

        Every call is counted in ``bootstrap_estimates``. A backtest that used
        them is reporting what WOULD happen if the assumption held — not
        evidence that it does.
        """
        stats = self.stats_for(strategy)
        self.bootstrap_estimates += 1
        self.bootstrap_strategies.add(strategy)

        breakeven = self.breakeven_win_rate(gross_win, gross_loss, _flat_costs(cost_total))
        assumed = clamp(breakeven + self.config.bootstrap_win_rate_margin, 0.01, 0.95)

        if stats.trades > 0:
            weight = clamp(stats.trades / max(1, self.config.bootstrap_min_trades), 0.0, 1.0)
            assumed = assumed * (1 - weight) + stats.observed_win_rate * weight
        return clamp(assumed, 0.01, 0.99)

    def uses_bootstrap(self, strategy: str) -> bool:
        """Whether this strategy's estimate would rest on an assumption."""
        return (
            self.config.bootstrap_enabled
            and self.stats_for(strategy).trades < self.config.bootstrap_min_trades
        )

    # ------------------------------------------------------------------ #
    # Edge
    # ------------------------------------------------------------------ #
    def estimate(
        self,
        signal: AggregatedSignal,
        liquidity: LiquiditySnapshot,
        notional: float,
        funding_rate: float = 0.0,
        seconds_to_funding: float = float("inf"),
        expected_duration_sec: float = 600.0,
        strategy: str | None = None,
        win_probability: float | None = None,
    ) -> EdgeEstimate:
        """Expected value of this trade, per unit of notional."""
        entry = signal.entry_price
        if entry <= 0:
            return _zero_edge()

        gross_win = safe_div(abs(signal.take_profit - entry), entry, 0.0)
        gross_loss = safe_div(abs(entry - signal.stop_loss), entry, 0.0)
        reward_risk = safe_div(gross_win, gross_loss, 0.0)

        # ONE attribution model: AggregatedSignal.primary_strategy. Taking
        # contributing[0] here made edge statistics refer to a different
        # strategy than the one the trade was booked against whenever the
        # aggregator's tuple order differed from confidence order.
        name = strategy or signal.primary_strategy
        p = (
            win_probability
            if win_probability is not None
            else self.win_probability(name, reward_risk)
        )

        costs = self.cost_model.estimate(
            direction=signal.direction,
            notional=notional,
            liquidity=liquidity,
            funding_rate=funding_rate,
            expected_duration_sec=expected_duration_sec,
            seconds_to_funding=seconds_to_funding,
        )

        # Resolved AFTER costs: the bootstrap assumption is anchored to the
        # break-even rate net of those costs, which needs the target size.
        if win_probability is not None:
            p = win_probability
        elif self.uses_bootstrap(name):
            p = self.bootstrap_probability(name, gross_win, gross_loss, costs.total)
        else:
            p = self.win_probability(name, reward_risk)

        expected_gross = p * gross_win - (1.0 - p) * gross_loss
        # Funding can be negative (received), so it is added rather than
        # subtracted as a magnitude.
        expected_net = (
            expected_gross
            - (costs.entry_fee + costs.exit_fee + costs.spread_cost + costs.slippage)
            - costs.funding
        )

        return EdgeEstimate(
            win_probability=p,
            gross_win=gross_win,
            gross_loss=gross_loss,
            costs=costs,
            expected_gross=expected_gross,
            expected_net=expected_net,
        )

    def evaluate(
        self,
        signal: AggregatedSignal,
        liquidity: LiquiditySnapshot,
        notional: float,
        **kwargs: float,
    ) -> EdgeDecision:
        """Estimate the edge and apply the minimum-edge threshold."""
        estimate = self.estimate(signal, liquidity, notional, **kwargs)  # type: ignore[arg-type]
        threshold = self.config.min_expected_edge
        accepted = estimate.expected_net > threshold

        if accepted:
            detail = f"net edge {estimate.expected_net * 100:.4f}% exceeds {threshold * 100:.4f}%"
        else:
            detail = (
                f"net edge {estimate.expected_net * 100:.4f}% does not clear "
                f"{threshold * 100:.4f}% "
                f"(gross {estimate.expected_gross * 100:.4f}%, "
                f"costs {estimate.costs.total * 100:.4f}%, p={estimate.win_probability:.2f})"
            )

        return EdgeDecision(
            estimate=estimate,
            accepted=accepted,
            threshold=threshold,
            detail=detail,
            inputs={
                "notional": notional,
                "spread_bps": liquidity.spread_bps,
                "depth_notional": liquidity.depth_notional,
                "reward_risk": safe_div(estimate.gross_win, estimate.gross_loss, 0.0),
            },
        )

    # ------------------------------------------------------------------ #
    def breakeven_win_rate(self, gross_win: float, gross_loss: float, costs: CostEstimate) -> float:
        """The win rate this trade needs just to break even after costs.

        Surfaced in the audit log because it makes a marginal trade legible:
        "this needs 61 % to break even and the strategy has done 48 %" is a much
        clearer explanation than a bare negative number.
        """
        denominator = gross_win + gross_loss
        if denominator <= 0:
            return 1.0
        return clamp((gross_loss + costs.total) / denominator, 0.0, 1.0)

    def realised_vs_expected(self, strategy: str) -> dict[str, float]:
        """Compare predicted edge against realised edge for a strategy.

        A persistent gap means the model is wrong — usually optimistic slippage
        or a win probability that does not survive contact with the market. This
        is the feedback loop that keeps the edge filter honest.
        """
        stats = self.stats_for(strategy)
        if stats.trades == 0:
            return {"trades": 0}
        expected = safe_div(stats.expected_edge_sum, stats.trades, 0.0)
        realised = safe_div(stats.realised_edge_sum, stats.trades, 0.0)
        return {
            "trades": stats.trades,
            "expected_edge_mean": expected,
            "realised_edge_mean": realised,
            "gap": realised - expected,
            "observed_win_rate": stats.observed_win_rate,
            "shrunk_win_rate": self.win_probability(strategy),
        }

    def summary(self) -> dict[str, dict[str, float]]:
        return {name: self.realised_vs_expected(name) for name in self._stats}

    def export_stats(self) -> dict[str, dict[str, float]]:
        """Measured per-strategy statistics, for seeding another run.

        This is how a validated backtest hands its measurements to live
        trading: live never bootstraps, so it must START from evidence rather
        than gather it with real money.
        """
        return {
            name: {
                "trades": stats.trades,
                "wins": stats.wins,
                "gross_win_sum": stats.gross_win_sum,
                "gross_loss_sum": stats.gross_loss_sum,
            }
            for name, stats in self._stats.items()
        }

    def seed_from(self, exported: dict[str, dict[str, float]]) -> None:
        """Load measured statistics produced by a previous validated run."""
        for name, values in exported.items():
            stats = self.stats_for(name)
            stats.trades = int(values.get("trades", 0))
            stats.wins = int(values.get("wins", 0))
            stats.gross_win_sum = float(values.get("gross_win_sum", 0.0))
            stats.gross_loss_sum = float(values.get("gross_loss_sum", 0.0))
        log.info(
            "edge_stats_seeded",
            strategies=sorted(exported),
            total_trades=sum(int(v.get("trades", 0)) for v in exported.values()),
        )

    @property
    def bootstrap_active(self) -> bool:
        """True when estimates may rest on assumptions rather than measurement."""
        return self.config.bootstrap_enabled


def _flat_costs(total: float) -> CostEstimate:
    """Wrap a scalar cost total in a CostEstimate for the break-even helper."""
    return CostEstimate(total, 0.0, 0.0, 0.0, 0.0)


def _zero_edge() -> EdgeEstimate:
    empty = CostEstimate(0.0, 0.0, 0.0, 0.0, 0.0)
    return EdgeEstimate(0.0, 0.0, 0.0, empty, 0.0, -1.0)


def required_move_for_edge(
    costs: CostEstimate, win_probability: float, reward_risk: float, min_edge: float
) -> float:
    """The gross win, as a fraction, needed to clear ``min_edge``.

    Useful for answering "how far does price have to go for this to be worth
    doing?" — which is the question that makes scalping's difficulty concrete.
    """
    p = clamp(win_probability, 0.01, 0.99)
    rr = max(reward_risk, 1e-9)
    # expected_gross = p*W - (1-p)*W/rr ; solve for W given the target net.
    coefficient = p - (1.0 - p) / rr
    if coefficient <= 0:
        return float("inf")  # this reward:risk cannot produce a positive edge
    return (min_edge + costs.total) / coefficient
