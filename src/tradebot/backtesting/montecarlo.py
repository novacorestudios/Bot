"""Monte Carlo analysis of a trade sequence.

A backtest reports **one** ordering of trades — the one that happened. That
ordering flatters or damns the result largely by luck: the same set of trades
arranged differently produces a very different maximum drawdown, and drawdown is
what decides whether an account survives.

So the trade sequence is resampled many times and the *distribution* of outcomes
is examined. The number that matters is not the mean (which is fixed by the
trades themselves) but the tail:

* **5th-percentile drawdown** — a plausible bad run. If that exceeds the
  configured `max_drawdown`, the strategy will trip its own kill switch in
  normal operation, which is a design failure regardless of expectancy.
* **Probability of ruin** — how often a resampled run loses more than a stated
  fraction of the account.
* **Probability of a losing outcome** — how often the whole run ends negative.

Two resampling methods:

* ``bootstrap`` (default) — sample trades **with replacement**. This treats the
  trade distribution as the population, and is the right choice when trades are
  roughly independent.
* ``shuffle`` — permute the observed trades. This preserves the exact trade set
  and asks only "what if they had arrived in a different order?"

Both assume trades are independent, which is not strictly true: consecutive
trades share a market regime, so real losing streaks cluster more than either
method predicts. **Both therefore understate tail risk**, and a strategy that
only just passes should be treated as failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tradebot.core.config import MonteCarloConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import Trade

log = get_logger(__name__)


@dataclass(slots=True)
class MonteCarloReport:
    iterations: int
    method: str
    trades_per_run: int

    mean_return: float
    median_return: float
    std_return: float
    percentile_5_return: float
    percentile_95_return: float

    mean_max_drawdown: float
    median_max_drawdown: float
    percentile_95_drawdown: float  # a plausible BAD drawdown
    worst_max_drawdown: float

    probability_of_loss: float
    probability_of_ruin: float
    ruin_threshold: float

    longest_losing_streak_p95: int
    verdict: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"MONTE CARLO ({self.method}, {self.iterations} iterations)",
            "=" * 60,
            f"Trades per run         {self.trades_per_run}",
            "",
            f"Median return          {self.median_return * 100:+.2f}%",
            f"5th pct return         {self.percentile_5_return * 100:+.2f}%",
            f"95th pct return        {self.percentile_95_return * 100:+.2f}%",
            "",
            f"Median max drawdown    {self.median_max_drawdown * 100:.2f}%",
            f"95th pct drawdown      {self.percentile_95_drawdown * 100:.2f}%  <- plan for this",
            f"Worst max drawdown     {self.worst_max_drawdown * 100:.2f}%",
            "",
            f"P(losing overall)      {self.probability_of_loss * 100:.1f}%",
            f"P(ruin > {self.ruin_threshold * 100:.0f}% loss)     "
            f"{self.probability_of_ruin * 100:.1f}%",
            f"95th pct loss streak   {self.longest_losing_streak_p95}",
            "",
            f"VERDICT: {self.verdict}",
        ]
        if self.warnings:
            lines += ["", "WARNINGS:"] + [f"  ! {w}" for w in self.warnings]
        lines += [
            "",
            "Resampling assumes trades are independent. They are not: adjacent",
            "trades share a market regime, so real losing streaks cluster more",
            "than this model predicts. These figures UNDERSTATE tail risk.",
            "=" * 60,
        ]
        return "\n".join(lines)


class MonteCarloAnalyzer:
    """Resamples a trade sequence to estimate the distribution of outcomes."""

    def __init__(self, config: MonteCarloConfig, seed: int | None = None) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        trades: list[Trade],
        initial_capital: float,
        max_drawdown_limit: float = 0.10,
        ruin_threshold: float = 0.5,
    ) -> MonteCarloReport:
        """Resample the trade sequence and report the outcome distribution."""
        if len(trades) < 10:
            return _empty_report(
                self.config,
                f"only {len(trades)} trades; resampling needs at least 30 to say "
                f"anything, and several hundred to be useful",
            )

        pnls = np.array([t.net_pnl for t in trades], dtype=np.float64)
        n = len(pnls)
        iterations = self.config.iterations

        returns = np.empty(iterations, dtype=np.float64)
        drawdowns = np.empty(iterations, dtype=np.float64)
        streaks = np.empty(iterations, dtype=np.int64)
        ruined = 0

        for i in range(iterations):
            if self.config.method == "shuffle":
                sample = self._rng.permutation(pnls)
            else:
                sample = self._rng.choice(pnls, size=n, replace=True)

            equity = initial_capital + np.cumsum(sample)
            final = float(equity[-1])
            returns[i] = safe_div(final - initial_capital, initial_capital, 0.0)

            peaks = np.maximum.accumulate(np.concatenate(([initial_capital], equity)))
            curve = np.concatenate(([initial_capital], equity))
            with np.errstate(divide="ignore", invalid="ignore"):
                series = np.where(peaks > 0, (peaks - curve) / peaks, 0.0)
            drawdowns[i] = float(np.max(series))

            streaks[i] = _longest_negative_streak(sample)
            if final <= initial_capital * (1 - ruin_threshold):
                ruined += 1

        report = MonteCarloReport(
            iterations=iterations,
            method=self.config.method,
            trades_per_run=n,
            mean_return=float(returns.mean()),
            median_return=float(np.median(returns)),
            std_return=float(returns.std()),
            percentile_5_return=float(np.percentile(returns, 5)),
            percentile_95_return=float(np.percentile(returns, 95)),
            mean_max_drawdown=float(drawdowns.mean()),
            median_max_drawdown=float(np.median(drawdowns)),
            percentile_95_drawdown=float(np.percentile(drawdowns, 95)),
            worst_max_drawdown=float(drawdowns.max()),
            probability_of_loss=float((returns < 0).mean()),
            probability_of_ruin=ruined / iterations,
            ruin_threshold=ruin_threshold,
            longest_losing_streak_p95=int(np.percentile(streaks, 95)),
            verdict="",
        )
        report.verdict = self._verdict(report, max_drawdown_limit, n)
        self._add_warnings(report, max_drawdown_limit, n)
        return report

    # ------------------------------------------------------------------ #
    @staticmethod
    def _verdict(report: MonteCarloReport, drawdown_limit: float, trades: int) -> str:
        if trades < 100:
            return f"INCONCLUSIVE — {trades} trades is too small a sample to resample meaningfully"
        if report.percentile_95_drawdown > drawdown_limit:
            return (
                f"FAILS — a plausible bad run draws down "
                f"{report.percentile_95_drawdown * 100:.1f}%, beyond the "
                f"{drawdown_limit * 100:.0f}% limit. The strategy would trip its "
                f"own kill switch in normal operation."
            )
        if report.probability_of_loss > 0.35:
            return (
                f"FAILS — {report.probability_of_loss * 100:.0f}% of resampled "
                f"runs end negative; the edge is not robust to ordering"
            )
        if report.probability_of_ruin > 0.01:
            return (
                f"FAILS — {report.probability_of_ruin * 100:.1f}% chance of "
                f"losing over {report.ruin_threshold * 100:.0f}% of the account"
            )
        return (
            "PASSES the Monte Carlo criteria — but see the independence caveat "
            "below; real losing streaks cluster worse than this."
        )

    @staticmethod
    def _add_warnings(report: MonteCarloReport, drawdown_limit: float, trades: int) -> None:
        if trades < 200:
            report.warnings.append(
                f"{trades} trades: the resampled distribution inherits the "
                f"sample's own biases and is only as good as it is"
            )
        if report.percentile_95_drawdown > drawdown_limit * 0.8:
            report.warnings.append(
                f"the 95th-percentile drawdown "
                f"({report.percentile_95_drawdown * 100:.1f}%) is close to the "
                f"{drawdown_limit * 100:.0f}% limit; little margin for error"
            )
        if report.longest_losing_streak_p95 >= 8:
            report.warnings.append(
                f"a plausible losing streak is "
                f"{report.longest_losing_streak_p95} trades — verify the "
                f"consecutive-loss kill switch is set accordingly"
            )


def _longest_negative_streak(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if value <= 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _empty_report(config: MonteCarloConfig, reason: str) -> MonteCarloReport:
    return MonteCarloReport(
        iterations=0,
        method=config.method,
        trades_per_run=0,
        mean_return=0.0,
        median_return=0.0,
        std_return=0.0,
        percentile_5_return=0.0,
        percentile_95_return=0.0,
        mean_max_drawdown=0.0,
        median_max_drawdown=0.0,
        percentile_95_drawdown=0.0,
        worst_max_drawdown=0.0,
        probability_of_loss=0.0,
        probability_of_ruin=0.0,
        ruin_threshold=0.5,
        longest_losing_streak_p95=0,
        verdict=f"INCONCLUSIVE — {reason}",
        warnings=[reason],
    )


def parameter_robustness(results: dict[str, float], tolerance: float = 0.5) -> tuple[bool, str]:
    """Judge whether performance survives parameter perturbation.

    ``results`` maps a parameter-variant label to its return. A strategy whose
    performance collapses under a ±20 % parameter change was fitted to the
    parameter, not to the market — the brief calls this out explicitly, and it
    is one of the most reliable overfitting detectors available.
    """
    if len(results) < 3:
        return False, "at least three parameter variants are needed to judge"

    values = np.array(list(results.values()), dtype=np.float64)
    baseline = float(np.max(values))
    if baseline <= 0:
        return False, "no variant was profitable"

    positive = int((values > 0).sum())
    share = positive / values.size
    worst = float(np.min(values))

    if share < 0.6:
        return False, (
            f"only {share:.0%} of parameter variants were profitable; the "
            f"result depends on the exact parameter value"
        )
    if worst < -abs(baseline) * tolerance:
        return False, (
            f"the worst variant returned {worst * 100:.1f}% against a best of "
            f"{baseline * 100:.1f}%; performance is not stable across parameters"
        )
    return True, (
        f"{share:.0%} of variants profitable, worst {worst * 100:.1f}%: "
        f"performance is reasonably stable across parameters"
    )
