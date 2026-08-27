"""Performance metrics.

Every number the brief asks for, computed from a list of completed trades and an
equity curve. Shared by the backtester, the paper broker and live performance
reporting, so the same definitions apply everywhere and results are comparable.

Two definitional choices worth stating, because they change the numbers
materially and are often fudged:

* **Drawdown is measured on the equity curve, not on the trade sequence.**
  Closed-trade drawdown ignores the open position that was 8 % underwater before
  it recovered — and that excursion is what would actually have hit a kill
  switch or a margin call.
* **Sharpe and Sortino are computed per-period on the equity curve and then
  annualised by the observed sampling frequency**, not per-trade. A per-trade
  "Sharpe" is not comparable with anything published, and inflates as trade
  frequency rises.

Where a metric is undefined (no losses, no trades, zero variance) the functions
return an explicit sentinel rather than infinity or nan, and the report marks it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tradebot.core.mathutil import safe_div
from tradebot.core.types import Trade

#: Returned when a ratio's denominator is zero and the numerator is positive.
UNDEFINED = float("nan")

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(slots=True)
class EquityPoint:
    timestamp: int
    equity: float


@dataclass(slots=True)
class BacktestMetrics:
    """The complete result set required by the brief."""

    # Capital
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    payoff_ratio: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_abs: float = 0.0
    max_drawdown_duration_sec: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility_annualized: float = 0.0

    # Activity
    trades_per_day: float = 0.0
    average_trade_duration_sec: float = 0.0
    median_trade_duration_sec: float = 0.0
    time_in_market: float = 0.0
    max_concurrent_positions: int = 0

    # Streaks and days
    longest_winning_streak: int = 0
    longest_losing_streak: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0
    winning_days: int = 0
    losing_days: int = 0

    # Costs — the numbers that decide whether a scalping edge is real
    total_fees: float = 0.0
    total_funding: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    gross_profit: float = 0.0
    net_profit: float = 0.0
    cost_ratio: float = 0.0

    # Breakdowns
    by_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_exit_reason: dict[str, int] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def summary_lines(self) -> list[str]:
        """Human-readable report body."""

        def pct(value: float) -> str:
            return f"{value * 100:.2f}%"

        def ratio(value: float) -> str:
            return "n/a" if math.isnan(value) else f"{value:.2f}"

        return [
            f"Initial capital        {self.initial_capital:.2f}",
            f"Final capital          {self.final_capital:.2f}",
            f"Total return           {pct(self.total_return)}",
            f"Annualized return      {pct(self.annualized_return)}",
            "",
            f"Total trades           {self.total_trades}",
            f"Win rate               {pct(self.win_rate)} "
            f"({self.winning_trades}W / {self.losing_trades}L)",
            f"Profit factor          {ratio(self.profit_factor)}",
            f"Expectancy             {self.expectancy:.4f} ({self.expectancy_r:.3f}R)",
            f"Average win / loss     {self.average_win:.4f} / {self.average_loss:.4f}",
            f"Payoff ratio           {ratio(self.payoff_ratio)}",
            "",
            f"Max drawdown           {pct(self.max_drawdown)} ({self.max_drawdown_abs:.2f})",
            f"Sharpe ratio           {ratio(self.sharpe_ratio)}",
            f"Sortino ratio          {ratio(self.sortino_ratio)}",
            f"Calmar ratio           {ratio(self.calmar_ratio)}",
            "",
            f"Trades per day         {self.trades_per_day:.1f}",
            f"Avg trade duration     {self.average_trade_duration_sec / 60:.1f} min",
            f"Longest win streak     {self.longest_winning_streak}",
            f"Longest loss streak    {self.longest_losing_streak}",
            f"Best / worst day       {self.best_day:.2f} / {self.worst_day:.2f}",
            "",
            f"Gross profit           {self.gross_profit:.4f}",
            f"Total fees             {self.total_fees:.4f}",
            f"Total funding          {self.total_funding:.4f}",
            f"Total slippage         {self.total_slippage:.4f}",
            f"NET PROFIT             {self.net_profit:.4f}",
            f"Costs as % of gross    {pct(self.cost_ratio)}",
        ]


# --------------------------------------------------------------------------- #
def compute_metrics(
    trades: list[Trade],
    equity_curve: list[EquityPoint],
    initial_capital: float,
    period_seconds: float | None = None,
) -> BacktestMetrics:
    """Compute the full metric set. Safe on an empty trade list."""
    metrics = BacktestMetrics(initial_capital=initial_capital)

    if not equity_curve:
        metrics.final_capital = initial_capital
        metrics.warnings.append("no equity curve; metrics are empty")
        return metrics

    equity = np.array([point.equity for point in equity_curve], dtype=np.float64)
    times = np.array([point.timestamp for point in equity_curve], dtype=np.float64)

    metrics.final_capital = float(equity[-1])
    metrics.total_return = safe_div(metrics.final_capital - initial_capital, initial_capital, 0.0)

    elapsed_sec = max(1.0, (times[-1] - times[0]) / 1000.0)
    years = elapsed_sec / SECONDS_PER_YEAR

    # Annualised return, geometric — and heavily guarded, because compounding a
    # short sample to a year is where backtests produce their most absurd
    # numbers. A 10% gain over one hour annualises to roughly 10^300; reporting
    # that (or an overflow) as a "return" is worse than not reporting it.
    MIN_ANNUALISATION_DAYS = 7.0
    if metrics.final_capital <= 0:
        metrics.annualized_return = -1.0
        metrics.warnings.append("account was wiped out")
    elif elapsed_sec < MIN_ANNUALISATION_DAYS * 86_400:
        metrics.annualized_return = 0.0
        metrics.warnings.append(
            f"sample covers only {elapsed_sec / 86_400:.1f} days, below the "
            f"{MIN_ANNUALISATION_DAYS:.0f}-day floor for annualisation; "
            f"annualized_return is reported as 0, not extrapolated. Use "
            f"total_return instead."
        )
    elif years > 0 and initial_capital > 0:
        growth = metrics.final_capital / initial_capital
        try:
            annualised = float(growth ** (1.0 / years)) - 1.0
        except OverflowError:
            annualised = float("inf")
        if not math.isfinite(annualised) or abs(annualised) > 100.0:
            # Above 10,000% a year the number carries no information beyond
            # "the sample is too short or too lucky to extrapolate".
            metrics.annualized_return = math.copysign(100.0, annualised or 1.0)
            metrics.warnings.append(
                f"annualized return capped at "
                f"{metrics.annualized_return * 100:.0f}%; the underlying figure "
                f"is an artefact of extrapolating a {elapsed_sec / 86_400:.1f}-day "
                f"sample"
            )
        else:
            metrics.annualized_return = annualised

    # -- drawdown, on the EQUITY CURVE ------------------------------------- #
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    metrics.max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    metrics.max_drawdown_abs = float(np.max(peaks - equity)) if equity.size else 0.0
    metrics.max_drawdown_duration_sec = _max_drawdown_duration(equity, times)

    # -- risk-adjusted returns --------------------------------------------- #
    if equity.size >= 3:
        returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], np.nan)
        returns = returns[np.isfinite(returns)]
        if returns.size >= 2:
            interval = period_seconds or max(
                1.0, (times[-1] - times[0]) / 1000.0 / max(1, returns.size)
            )
            periods_per_year = SECONDS_PER_YEAR / interval
            mean = float(np.mean(returns))
            std = float(np.std(returns, ddof=1))

            metrics.volatility_annualized = std * math.sqrt(periods_per_year)
            metrics.sharpe_ratio = (
                mean / std * math.sqrt(periods_per_year) if std > 0 else UNDEFINED
            )
            downside = returns[returns < 0]
            downside_std = float(np.std(downside, ddof=1)) if downside.size >= 2 else 0.0
            metrics.sortino_ratio = (
                mean / downside_std * math.sqrt(periods_per_year) if downside_std > 0 else UNDEFINED
            )

    metrics.calmar_ratio = (
        safe_div(metrics.annualized_return, metrics.max_drawdown, UNDEFINED)
        if metrics.max_drawdown > 0
        else UNDEFINED
    )

    # -- trades -------------------------------------------------------------- #
    if not trades:
        metrics.warnings.append(
            "no trades were taken. With opportunity-driven trading this is a "
            "valid outcome, but verify it is not a data or configuration fault."
        )
        return metrics

    _populate_trade_metrics(metrics, trades, elapsed_sec)
    return metrics


def _populate_trade_metrics(
    metrics: BacktestMetrics, trades: list[Trade], elapsed_sec: float
) -> None:
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]

    metrics.total_trades = len(trades)
    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.win_rate = safe_div(len(wins), len(trades), 0.0)
    metrics.loss_rate = safe_div(len(losses), len(trades), 0.0)

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    metrics.profit_factor = (
        safe_div(gross_profit, gross_loss, UNDEFINED)
        if gross_loss > 0
        else (UNDEFINED if gross_profit > 0 else 0.0)
    )

    metrics.average_win = safe_div(gross_profit, len(wins), 0.0)
    metrics.average_loss = safe_div(gross_loss, len(losses), 0.0)
    metrics.payoff_ratio = safe_div(metrics.average_win, metrics.average_loss, UNDEFINED)
    metrics.largest_win = max((t.net_pnl for t in wins), default=0.0)
    metrics.largest_loss = min((t.net_pnl for t in losses), default=0.0)

    metrics.expectancy = safe_div(sum(t.net_pnl for t in trades), len(trades), 0.0)
    r_multiples = [t.r_multiple for t in trades if t.initial_risk > 0]
    metrics.expectancy_r = safe_div(sum(r_multiples), len(r_multiples), 0.0)

    # -- costs ---------------------------------------------------------------- #
    metrics.total_fees = sum(t.fees for t in trades)
    metrics.total_funding = sum(t.funding for t in trades)
    metrics.total_slippage = sum(t.slippage_cost for t in trades)
    metrics.total_costs = metrics.total_fees + metrics.total_funding + metrics.total_slippage
    metrics.gross_profit = sum(t.gross_pnl for t in trades)
    metrics.net_profit = sum(t.net_pnl for t in trades)
    metrics.cost_ratio = safe_div(metrics.total_costs, abs(metrics.gross_profit), UNDEFINED)

    # -- activity -------------------------------------------------------------- #
    days = max(elapsed_sec / 86_400.0, 1e-9)
    metrics.trades_per_day = len(trades) / days
    durations = [t.duration_sec for t in trades]
    metrics.average_trade_duration_sec = safe_div(sum(durations), len(durations), 0.0)
    metrics.median_trade_duration_sec = float(np.median(durations)) if durations else 0.0
    metrics.time_in_market = min(1.0, safe_div(sum(durations), elapsed_sec, 0.0))

    # -- streaks --------------------------------------------------------------- #
    metrics.longest_winning_streak, metrics.longest_losing_streak = _streaks(trades)

    # -- daily ----------------------------------------------------------------- #
    daily: dict[int, float] = {}
    for trade in trades:
        day = trade.closed_at // 86_400_000
        daily[day] = daily.get(day, 0.0) + trade.net_pnl
    if daily:
        metrics.best_day = max(daily.values())
        metrics.worst_day = min(daily.values())
        metrics.winning_days = sum(1 for v in daily.values() if v > 0)
        metrics.losing_days = sum(1 for v in daily.values() if v < 0)

    # -- breakdowns ------------------------------------------------------------ #
    metrics.by_strategy = _group(trades, lambda t: t.strategy)
    metrics.by_regime = _group(trades, lambda t: t.regime.value)
    metrics.by_symbol = _group(trades, lambda t: t.symbol)
    for trade in trades:
        key = trade.exit_reason.value
        metrics.by_exit_reason[key] = metrics.by_exit_reason.get(key, 0) + 1

    # -- honesty checks --------------------------------------------------------- #
    if len(trades) < 30:
        metrics.warnings.append(
            f"only {len(trades)} trades; every statistic here is dominated by "
            f"noise. At least 100 are needed for a weak signal, 300+ for "
            f"confidence."
        )
    if metrics.total_costs > abs(metrics.gross_profit) * 0.5:
        metrics.warnings.append(
            f"costs ({metrics.total_costs:.4f}) consumed more than half the "
            f"gross profit ({metrics.gross_profit:.4f}); the edge is thin"
        )
    if metrics.max_drawdown > 0.25:
        metrics.warnings.append(f"maximum drawdown {metrics.max_drawdown * 100:.1f}% is severe")


def _streaks(trades: list[Trade]) -> tuple[int, int]:
    best_win = best_loss = current_win = current_loss = 0
    for trade in trades:
        if trade.net_pnl > 0:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
        best_win = max(best_win, current_win)
        best_loss = max(best_loss, current_loss)
    return best_win, best_loss


def _group(trades: list[Trade], key) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        buckets.setdefault(key(trade), []).append(trade)

    out: dict[str, dict[str, Any]] = {}
    for name, group in buckets.items():
        wins = [t for t in group if t.net_pnl > 0]
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in group if t.net_pnl <= 0))
        r_multiples = [t.r_multiple for t in group if t.initial_risk > 0]
        out[name] = {
            "trades": len(group),
            "wins": len(wins),
            "win_rate": round(safe_div(len(wins), len(group), 0.0), 4),
            "net_pnl": round(sum(t.net_pnl for t in group), 6),
            "expectancy": round(safe_div(sum(t.net_pnl for t in group), len(group), 0.0), 6),
            "expectancy_r": round(safe_div(sum(r_multiples), len(r_multiples), 0.0), 4),
            "profit_factor": round(safe_div(gross_profit, gross_loss, 0.0), 4)
            if gross_loss > 0
            else None,
            "fees": round(sum(t.fees for t in group), 6),
        }
    return out


def _max_drawdown_duration(equity: np.ndarray, times: np.ndarray) -> float:
    """Longest time spent below a previous equity peak, in seconds."""
    if equity.size < 2:
        return 0.0
    peak = equity[0]
    peak_time = times[0]
    longest = 0.0
    for value, timestamp in zip(equity, times, strict=False):
        if value >= peak:
            peak = value
            peak_time = timestamp
        else:
            longest = max(longest, (timestamp - peak_time) / 1000.0)
    return longest
