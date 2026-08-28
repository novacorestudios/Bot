"""Running a backtest reproducibly, across all three execution scenarios.

Two jobs, both of which exist to stop a result being quietly shaped:

1. **Reproducibility (§38).** Every run records the git commit, a hash of the
   configuration, a hash of the datasets, the seed and the code version, under a
   run ID. A number in a report with no record of its inputs cannot be
   reproduced, audited or compared against a later run, so it is not evidence.

2. **All three scenarios or none (§41).** `run_scenarios` executes the same
   signals under BASE, CONSERVATIVE and STRESS and returns all three. Reporting
   only the flattering one is the single easiest way to fabricate an edge, so
   the API does not offer a way to produce one scenario in isolation for a
   report.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess  # nosec B404 - reads the local git commit, no user input
from dataclasses import dataclass, field
from typing import Any

from tradebot.backtesting.engine import BacktestData, BacktestEngine, BacktestResult
from tradebot.backtesting.execution import Scenario, scenarios
from tradebot.core.config import TunableConfig
from tradebot.core.logging import get_logger
from tradebot.data.manifest import DatasetManifest, dataset_fingerprint, utc_now_iso

log = get_logger(__name__)

CODE_VERSION = "v3"


def git_commit() -> str:
    """The commit this run was produced from, or 'unknown' outside a checkout.

    ``git`` is resolved to an absolute path first rather than left to a PATH
    lookup, so which binary runs does not depend on the caller's environment.
    Fixed argv, no shell, no user input.
    """
    executable = shutil.which("git")
    if executable is None:
        return "unknown"
    try:
        # argv is a resolved absolute path plus two literals; there is no
        # untrusted input for B603 to be about, and no shell.
        result = subprocess.run(  # noqa: S603  # nosec B603
            [executable, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def config_hash(config: TunableConfig) -> str:
    """A digest of every tunable that could change a result.

    Hashes the whole validated config rather than a chosen subset, because the
    parameter someone forgot to include is exactly the one that will differ
    between two runs that are supposed to match.
    """
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(slots=True)
class RunContext:
    """Everything needed to reproduce a run (§38)."""

    run_id: str
    git_commit: str
    config_hash: str
    dataset_fingerprint: str
    seed: int
    code_version: str = CODE_VERSION
    started_at: str = field(default_factory=utc_now_iso)
    symbols: list[str] = field(default_factory=list)
    intervals: list[str] = field(default_factory=list)
    start_ms: int = 0
    end_ms: int = 0
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "dataset_fingerprint": self.dataset_fingerprint,
            "seed": self.seed,
            "code_version": self.code_version,
            "started_at": self.started_at,
            "symbols": list(self.symbols),
            "intervals": list(self.intervals),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "notes": self.notes,
        }

    def lines(self) -> list[str]:
        return [
            f"  Run ID              {self.run_id}",
            f"  Git commit          {self.git_commit[:12]}",
            f"  Config hash         {self.config_hash[:16]}",
            f"  Dataset fingerprint {self.dataset_fingerprint[:16] or 'NONE'}",
            f"  Seed                {self.seed}",
            f"  Code version        {self.code_version}",
            f"  Started             {self.started_at}",
        ]


def build_context(
    config: TunableConfig,
    data: dict[str, BacktestData],
    seed: int,
    manifests: list[DatasetManifest] | None = None,
    notes: str = "",
) -> RunContext:
    fingerprint = dataset_fingerprint(manifests or [])
    if not fingerprint and data:
        # No manifests (fixtures, or a legacy dataset): fingerprint the bars we
        # were actually handed, so the run is still self-describing.
        digest = hashlib.sha256()
        for symbol in sorted(data):
            entry = data[symbol]
            for interval in sorted(entry.candles):
                bars = entry.candles[interval]
                digest.update(f"{symbol}:{interval}:{len(bars)}".encode())
                if bars:
                    digest.update(f"{bars[0].open_time}:{bars[-1].close_time}".encode())
        fingerprint = digest.hexdigest()

    config_digest = config_hash(config)
    run_id = hashlib.sha256(
        f"{config_digest}:{fingerprint}:{seed}:{utc_now_iso()}".encode()
    ).hexdigest()[:16]

    intervals = sorted({tf for entry in data.values() for tf in entry.candles})
    starts = [b[0].open_time for e in data.values() for b in e.candles.values() if b]
    ends = [b[-1].close_time for e in data.values() for b in e.candles.values() if b]

    return RunContext(
        run_id=run_id,
        git_commit=git_commit(),
        config_hash=config_digest,
        dataset_fingerprint=fingerprint,
        seed=seed,
        symbols=sorted(data),
        intervals=intervals,
        start_ms=min(starts) if starts else 0,
        end_ms=max(ends) if ends else 0,
        notes=notes,
    )


@dataclass(slots=True)
class ScenarioResults:
    """The same signals under all three execution assumptions."""

    context: RunContext
    results: dict[Scenario, BacktestResult] = field(default_factory=dict)
    engines: dict[Scenario, BacktestEngine] = field(default_factory=dict)

    def comparison(self) -> list[dict[str, Any]]:
        rows = []
        for scenario in (Scenario.BASE, Scenario.CONSERVATIVE, Scenario.STRESS):
            result = self.results.get(scenario)
            if result is None:
                continue
            metrics = result.metrics
            engine = self.engines.get(scenario)
            rows.append(
                {
                    "scenario": scenario.value,
                    "trades": metrics.total_trades,
                    "net_pnl": round(metrics.net_profit, 4),
                    "return_pct": round(metrics.total_return * 100, 3),
                    "win_rate": round(metrics.win_rate, 4),
                    "profit_factor": round(metrics.profit_factor, 3),
                    "expectancy": round(metrics.expectancy, 5),
                    "max_drawdown": round(metrics.max_drawdown, 4),
                    "sharpe": round(metrics.sharpe_ratio, 3),
                    "total_costs": round(metrics.total_costs, 4),
                    "liquidations": result.liquidations,
                    "rejected_orders": engine.simulator.rejections
                    if engine and engine.simulator
                    else 0,
                }
            )
        return rows

    @property
    def survives_stress(self) -> bool:
        """Did the edge survive the worst assumptions?

        The question that matters. A strategy profitable under BASE and
        destroyed under CONSERVATIVE has an edge thinner than the error bars on
        the cost model, which is the same as having no edge you can rely on.
        """
        stress = self.results.get(Scenario.STRESS)
        return stress is not None and stress.metrics.net_profit > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.as_dict(),
            "scenarios": self.comparison(),
            "survives_stress": self.survives_stress,
        }


def run_scenarios(
    config: TunableConfig,
    data: dict[str, BacktestData],
    start_ms: int | None = None,
    end_ms: int | None = None,
    seed: int = 0,
    manifests: list[DatasetManifest] | None = None,
    initial_capital: float | None = None,
    notes: str = "",
) -> ScenarioResults:
    """Run BASE, CONSERVATIVE and STRESS on identical data and signals.

    The same seed is used for all three, so the random events (rejections,
    partial fills) that *do* differ between scenarios differ because the
    scenario says they should, not because the RNG wandered.
    """
    context = build_context(config, data, seed, manifests, notes)
    assumptions = scenarios(
        config.backtest.spread_bps,
        config.backtest.slippage_bps,
        config.backtest.taker_fee,
        config.backtest.maker_fee,
    )

    out = ScenarioResults(context=context)
    for scenario in (Scenario.BASE, Scenario.CONSERVATIVE, Scenario.STRESS):
        engine = BacktestEngine(config, initial_capital)
        result = engine.run(data, start_ms, end_ms, assumptions[scenario], seed=seed)
        out.results[scenario] = result
        out.engines[scenario] = engine
        log.info(
            "scenario_complete",
            run_id=context.run_id,
            scenario=scenario.value,
            trades=result.metrics.total_trades,
            net_pnl=round(result.metrics.net_profit, 4),
        )
    return out
