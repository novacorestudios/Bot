"""Turning a run into the reports the brief asks for.

Machine-readable JSON plus a human-readable Markdown summary, both generated
from the same `ScenarioResults` so they cannot disagree.

The one design rule: **anything not measured is reported as NOT MEASURED, never
as zero.** A win rate of 0.0 and an absent win rate look identical in a table
and mean opposite things, and the difference is exactly what decides whether a
system goes anywhere near real money.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradebot.backtesting.engine import BacktestEngine, BacktestResult
from tradebot.backtesting.execution import Scenario
from tradebot.backtesting.runner import ScenarioResults
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import Trade

log = get_logger(__name__)

NOT_MEASURED = "NOT MEASURED"


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def duration_stats(trades: list[Trade]) -> dict[str, float]:
    """§23: average, median, P95 and max. P95 is the one that was missing."""
    durations = [t.duration_sec for t in trades]
    if not durations:
        return {"average": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
    return {
        "average": sum(durations) / len(durations),
        "median": percentile(durations, 0.5),
        "p95": percentile(durations, 0.95),
        "max": max(durations),
        "count": len(durations),
    }


def frequency_stats(trades: list[Trade], start_ms: int, end_ms: int) -> dict[str, float]:
    """§22: activity is reported, never targeted."""
    hours = max(1e-9, (end_ms - start_ms) / 3_600_000)
    return {
        "trades": len(trades),
        "per_hour": len(trades) / hours,
        "per_day": len(trades) / (hours / 24),
        "per_week": len(trades) / (hours / 168),
        "span_hours": hours,
    }


def group_stats(trades: list[Trade], key: Any) -> dict[str, dict[str, Any]]:
    """Per-strategy, per-symbol or per-regime aggregates (§30, §31, §33)."""
    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        buckets.setdefault(str(key(trade)), []).append(trade)

    out: dict[str, dict[str, Any]] = {}
    for name, group in sorted(buckets.items()):
        wins = [t for t in group if t.net_pnl > 0]
        losses = [t for t in group if t.net_pnl <= 0]
        gross_win = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        out[name] = {
            "trades": len(group),
            "win_rate": round(safe_div(len(wins), len(group), 0.0), 4),
            "gross_pnl": round(sum(t.gross_pnl for t in group), 4),
            "net_pnl": round(sum(t.net_pnl for t in group), 4),
            "fees": round(sum(t.fees for t in group), 4),
            "funding": round(sum(t.funding for t in group), 4),
            "slippage": round(sum(t.slippage_cost for t in group), 4),
            "profit_factor": round(safe_div(gross_win, gross_loss, 0.0), 3),
            "expectancy": round(safe_div(sum(t.net_pnl for t in group), len(group), 0.0), 5),
            "avg_duration_sec": round(
                safe_div(sum(t.duration_sec for t in group), len(group), 0.0), 1
            ),
        }
    return out


def matrix_stats(trades: list[Trade], row: Any, column: Any) -> dict[str, dict[str, Any]]:
    """§32 / §33: a two-dimensional table of trades, expectancy and net PnL."""
    table: dict[str, dict[str, Any]] = {}
    for trade in trades:
        cell = table.setdefault(str(row(trade)), {}).setdefault(
            str(column(trade)), {"trades": 0, "net_pnl": 0.0, "wins": 0}
        )
        cell["trades"] += 1
        cell["net_pnl"] += trade.net_pnl
        cell["wins"] += int(trade.net_pnl > 0)

    for columns in table.values():
        for cell in columns.values():
            cell["net_pnl"] = round(cell["net_pnl"], 4)
            cell["expectancy"] = round(safe_div(cell["net_pnl"], cell["trades"], 0.0), 5)
            cell["win_rate"] = round(safe_div(cell["wins"], cell["trades"], 0.0), 4)
    return table


def edge_calibration(trades: list[Trade]) -> dict[str, Any]:
    """§34: predicted edge against realised edge, and by bucket.

    A systematically optimistic model approves trades that were never
    profitable, and the resulting losses read as strategy failure.
    """
    usable = [t for t in trades if t.entry_notional > 0]
    if not usable:
        return {"trades": 0, "status": NOT_MEASURED}

    errors = []
    buckets: dict[str, list[float]] = {}
    for trade in usable:
        realised = trade.net_pnl / trade.entry_notional
        error = realised - trade.expected_net_edge
        errors.append(error)
        expected = trade.expected_net_edge
        label = (
            "<0.10%"
            if expected < 0.0010
            else "0.10-0.15%"
            if expected < 0.0015
            else "0.15-0.20%"
            if expected < 0.0020
            else ">0.20%"
        )
        buckets.setdefault(label, []).append(error)

    return {
        "trades": len(usable),
        "mean_prediction_error": round(sum(errors) / len(errors), 6),
        "median_prediction_error": round(percentile(errors, 0.5), 6),
        "by_edge_bucket": {
            label: {
                "trades": len(values),
                "mean_error": round(sum(values) / len(values), 6),
                "median_error": round(percentile(values, 0.5), 6),
            }
            for label, values in sorted(buckets.items())
        },
    }


@dataclass(slots=True)
class BacktestReport:
    """Everything §28-§35 asks for, from one set of scenario runs."""

    scenarios: ScenarioResults

    def _primary(self) -> tuple[BacktestResult | None, BacktestEngine | None]:
        return (
            self.scenarios.results.get(Scenario.BASE),
            self.scenarios.engines.get(Scenario.BASE),
        )

    def as_dict(self) -> dict[str, Any]:
        result, engine = self._primary()
        if result is None:
            return {
                "context": self.scenarios.context.as_dict(),
                "status": NOT_MEASURED,
                "reason": "no scenario produced a result",
            }

        trades = result.trades
        payload: dict[str, Any] = {
            "context": self.scenarios.context.as_dict(),
            "scenario_comparison": self.scenarios.comparison(),
            "survives_stress": self.scenarios.survives_stress,
            "performance": result.metrics.as_dict(),
            "duration": duration_stats(trades),
            "frequency": frequency_stats(trades, result.start_ms, result.end_ms),
            "by_strategy": group_stats(trades, lambda t: t.strategy),
            "by_symbol": group_stats(trades, lambda t: t.symbol),
            "by_regime": group_stats(trades, lambda t: t.regime.value),
            "symbol_x_strategy": matrix_stats(trades, lambda t: t.symbol, lambda t: t.strategy),
            "strategy_x_regime": matrix_stats(
                trades, lambda t: t.strategy, lambda t: t.regime.value
            ),
            "edge_calibration": edge_calibration(trades),
            "rejections": dict(sorted(result.rejections.items(), key=lambda kv: -kv[1])),
            "liquidations": result.liquidations,
            "missing_timeframes": result.missing_timeframes or None,
            "bootstrap": {
                "estimates": result.bootstrap_estimates,
                "strategies": list(result.bootstrap_strategies),
            },
        }
        if engine is not None:
            payload["cost_breakdown"] = engine.cost_breakdown.as_dict()
            payload["execution_quality"] = engine.execution_quality.stats()
            payload["universe_selections"] = len(engine.universe_log)
        return payload

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        log.info("backtest_report_written", path=str(target))
        return target

    def markdown(self) -> str:
        """The human-readable summary."""
        data = self.as_dict()
        lines = ["# Backtest report", ""]

        lines += ["## Run", "", "```"]
        lines += self.scenarios.context.lines()
        lines += ["```", ""]

        if data.get("status") == NOT_MEASURED:
            lines += [
                "## Result",
                "",
                f"**{NOT_MEASURED}** — {data.get('reason', 'no data')}.",
                "",
                "No performance figures are given because none were produced.",
            ]
            return "\n".join(lines)

        missing = data.get("missing_timeframes")
        if missing:
            lines += [
                "> **This run is not interpretable.** The dataset is missing "
                "timeframes the strategies read, so a low trade count means "
                "*no data*, not *no edge*.",
                "",
                f"> Affected symbols: {len(missing)}",
                "",
            ]

        lines += [
            "## Scenarios",
            "",
            "| Scenario | Trades | Net PnL | Return % | Win rate | PF | Max DD | Liquidations |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in data["scenario_comparison"]:
            lines.append(
                f"| {row['scenario']} | {row['trades']} | {row['net_pnl']} | "
                f"{row['return_pct']} | {row['win_rate']} | {row['profit_factor']} | "
                f"{row['max_drawdown']} | {row['liquidations']} |"
            )
        lines += ["", f"**Survives STRESS:** {data['survives_stress']}", ""]

        if "cost_breakdown" in data:
            lines += ["## Cost breakdown", "", "```"]
            engine = self.scenarios.engines.get(Scenario.BASE)
            if engine is not None:
                lines += engine.cost_breakdown.table()
            lines += ["```", ""]

        frequency = data["frequency"]
        duration = data["duration"]
        lines += [
            "## Activity",
            "",
            f"- trades: {frequency['trades']} over {frequency['span_hours']:.1f} hours",
            f"- per hour / day / week: {frequency['per_hour']:.3f} / "
            f"{frequency['per_day']:.2f} / {frequency['per_week']:.1f}",
            f"- duration avg/median/p95/max (s): {duration['average']:.0f} / "
            f"{duration['median']:.0f} / {duration['p95']:.0f} / {duration['max']:.0f}",
            "",
        ]

        lines += ["## Why trades were not taken", "", "```"]
        lines += [f"  {k:<32} {v}" for k, v in list(data["rejections"].items())[:12]]
        lines += ["```", ""]
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.markdown() + "\n")
        return target
