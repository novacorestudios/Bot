"""Walk-forward analysis.

A single backtest over one period proves almost nothing. Parameters chosen by
looking at that period are fitted to it, and the result is a description of the
past rather than a prediction. Walk-forward is the standard defence: repeatedly
fit on a training window and evaluate on the *next*, unseen window.

    |---- train ----|- val -|- test -|
             |---- train ----|- val -|- test -|
                      |---- train ----|- val -|- test -|

Only the **test** windows count. They are the only data the parameters have
never seen, so their combined result is the closest thing to an honest estimate
of forward performance.

What makes a walk-forward result trustworthy is not a high average — it is
**consistency**. A strategy that made all its money in one fold and lost in five
is a strategy that worked once. The efficiency ratio (out-of-sample return
divided by in-sample return) measures how much of the fitted performance
survived contact with unseen data; well below 1 means the fit was to noise.

This module runs the folds and reports the distribution. It does NOT tune
parameters automatically — automated search over a fixed dataset is precisely
how overfitting happens, and the brief forbids adjusting live parameters without
validation. Parameter variants must be supplied explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tradebot.backtesting.engine import BacktestData, BacktestEngine, BacktestResult
from tradebot.core.config import TunableConfig, WalkForwardConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div

log = get_logger(__name__)

DAY_MS = 86_400_000


@dataclass(slots=True)
class Fold:
    """One train/validate/test cycle."""

    index: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def as_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


@dataclass(slots=True)
class FoldResult:
    fold: Fold
    train: BacktestResult | None
    test: BacktestResult | None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def test_return(self) -> float:
        return self.test.metrics.total_return if self.test else 0.0

    @property
    def train_return(self) -> float:
        return self.train.metrics.total_return if self.train else 0.0

    @property
    def test_trades(self) -> int:
        return self.test.metrics.total_trades if self.test else 0

    @property
    def efficiency(self) -> float:
        """Out-of-sample return as a fraction of in-sample return.

        Near 1.0 means the fit generalised. Well below 1.0 (or negative) means
        the in-sample result was largely fitted noise.
        """
        if self.train_return <= 0:
            return 0.0
        return self.test_return / self.train_return


@dataclass(slots=True)
class WalkForwardReport:
    folds: list[FoldResult]
    total_test_return: float
    mean_test_return: float
    median_test_return: float
    std_test_return: float
    positive_folds: int
    negative_folds: int
    consistency: float
    mean_efficiency: float
    worst_fold_return: float
    best_fold_return: float
    total_test_trades: int
    insufficient_folds: int
    verdict: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "WALK-FORWARD ANALYSIS",
            "=" * 60,
            f"Folds                  {len(self.folds)}",
            f"Positive / negative    {self.positive_folds} / {self.negative_folds}",
            f"Consistency            {self.consistency:.0%} of folds positive",
            f"Mean test return       {self.mean_test_return * 100:+.2f}%",
            f"Median test return     {self.median_test_return * 100:+.2f}%",
            f"Std dev of returns     {self.std_test_return * 100:.2f}%",
            f"Best / worst fold      {self.best_fold_return * 100:+.2f}% / "
            f"{self.worst_fold_return * 100:+.2f}%",
            f"Mean efficiency        {self.mean_efficiency:.2f} (out-of-sample / in-sample)",
            f"Total test trades      {self.total_test_trades}",
            "",
            f"VERDICT: {self.verdict}",
        ]
        if self.warnings:
            lines += ["", "WARNINGS:"] + [f"  ! {w}" for w in self.warnings]
        lines += [
            "",
            "Per fold:",
            f"  {'#':>3} {'train':>10} {'test':>10} {'trades':>7} {'eff':>7}",
        ]
        for result in self.folds:
            lines.append(
                f"  {result.fold.index:>3} "
                f"{result.train_return * 100:>9.2f}% "
                f"{result.test_return * 100:>9.2f}% "
                f"{result.test_trades:>7} "
                f"{result.efficiency:>7.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class WalkForwardAnalyzer:
    """Runs walk-forward folds over historical data."""

    def __init__(self, config: TunableConfig, walk_config: WalkForwardConfig | None = None) -> None:
        self.config = config
        self.walk = walk_config or config.walk_forward

    # ------------------------------------------------------------------ #
    def build_folds(self, start_ms: int, end_ms: int) -> list[Fold]:
        """Split the period into rolling train/validation/test windows."""
        train_ms = self.walk.train_days * DAY_MS
        validation_ms = self.walk.validation_days * DAY_MS
        test_ms = self.walk.test_days * DAY_MS
        step_ms = self.walk.step_days * DAY_MS
        window_ms = train_ms + validation_ms + test_ms

        folds: list[Fold] = []
        cursor = start_ms
        index = 0
        while cursor + window_ms <= end_ms:
            train_end = cursor + train_ms
            validation_end = train_end + validation_ms
            folds.append(
                Fold(
                    index=index,
                    train_start=cursor,
                    train_end=train_end,
                    validation_start=train_end,
                    validation_end=validation_end,
                    test_start=validation_end,
                    test_end=validation_end + test_ms,
                )
            )
            cursor += step_ms
            index += 1
        return folds

    # ------------------------------------------------------------------ #
    def run(
        self, data: dict[str, BacktestData], start_ms: int | None = None, end_ms: int | None = None
    ) -> WalkForwardReport:
        """Run every fold and summarise the out-of-sample distribution."""
        bounds = _data_bounds(data, self.config.timeframes.primary)
        start = start_ms if start_ms is not None else bounds[0]
        end = end_ms if end_ms is not None else bounds[1]

        folds = self.build_folds(start, end)
        if not folds:
            span_days = (end - start) / DAY_MS
            needed = self.walk.train_days + self.walk.validation_days + self.walk.test_days
            return _empty_report(
                f"only {span_days:.1f} days of data; a single fold needs "
                f"{needed} days. Download more history before drawing any "
                f"conclusion."
            )

        log.info(
            "walkforward_starting",
            folds=len(folds),
            train_days=self.walk.train_days,
            test_days=self.walk.test_days,
        )

        results: list[FoldResult] = []
        for fold in folds:
            train = BacktestEngine(self.config).run(data, fold.train_start, fold.train_end)
            # The parameters are NOT refitted here. Automated search over the
            # training window is exactly how overfitting is manufactured; the
            # training run exists to measure in-sample performance so that
            # efficiency (out-of-sample / in-sample) is meaningful.
            test = BacktestEngine(self.config).run(data, fold.test_start, fold.test_end)
            results.append(FoldResult(fold=fold, train=train, test=test))
            log.info(
                "walkforward_fold_complete",
                fold=fold.index,
                train_return=round(train.metrics.total_return, 4),
                test_return=round(test.metrics.total_return, 4),
                test_trades=test.metrics.total_trades,
            )

        return self._summarise(results)

    # ------------------------------------------------------------------ #
    def _summarise(self, results: list[FoldResult]) -> WalkForwardReport:
        returns = np.array([r.test_return for r in results], dtype=np.float64)
        positive = int((returns > 0).sum())
        negative = int((returns < 0).sum())
        consistency = safe_div(positive, len(results), 0.0)

        efficiencies = [r.efficiency for r in results if r.train_return > 0]
        mean_efficiency = float(np.mean(efficiencies)) if efficiencies else 0.0

        total_trades = sum(r.test_trades for r in results)
        insufficient = sum(1 for r in results if r.test_trades < self.walk.min_trades_per_fold)

        warnings: list[str] = []
        if insufficient:
            warnings.append(
                f"{insufficient} of {len(results)} folds had fewer than "
                f"{self.walk.min_trades_per_fold} trades; those folds carry no "
                f"statistical weight"
            )
        if total_trades == 0:
            warnings.append(
                "no trades in any test window. Either the filters are too "
                "strict for this data, or the data is unsuitable. Check the "
                "rejection counts in the individual fold reports."
            )
        if len(results) < 4:
            warnings.append(
                f"only {len(results)} folds; walk-forward needs considerably "
                f"more to distinguish skill from luck"
            )
        if mean_efficiency < 0.4 and efficiencies:
            warnings.append(
                f"efficiency {mean_efficiency:.2f}: out-of-sample performance is "
                f"far below in-sample, which is the signature of overfitting"
            )

        verdict = self._verdict(results, consistency, total_trades, mean_efficiency)

        return WalkForwardReport(
            folds=results,
            total_test_return=float(returns.sum()),
            mean_test_return=float(returns.mean()) if returns.size else 0.0,
            median_test_return=float(np.median(returns)) if returns.size else 0.0,
            std_test_return=float(returns.std()) if returns.size > 1 else 0.0,
            positive_folds=positive,
            negative_folds=negative,
            consistency=consistency,
            mean_efficiency=mean_efficiency,
            worst_fold_return=float(returns.min()) if returns.size else 0.0,
            best_fold_return=float(returns.max()) if returns.size else 0.0,
            total_test_trades=total_trades,
            insufficient_folds=insufficient,
            verdict=verdict,
            warnings=warnings,
        )

    @staticmethod
    def _verdict(
        results: list[FoldResult], consistency: float, total_trades: int, efficiency: float
    ) -> str:
        if total_trades == 0:
            return "INCONCLUSIVE — no trades were taken in any test window"
        if len(results) < 4:
            return "INCONCLUSIVE — too few folds to judge"
        if total_trades < 100:
            return (
                f"INCONCLUSIVE — {total_trades} out-of-sample trades is too few; "
                f"aim for several hundred"
            )
        returns = [r.test_return for r in results]
        if consistency >= 0.6 and sum(returns) > 0 and efficiency >= 0.4:
            return (
                "PASSES the walk-forward criteria. This is necessary, not "
                "sufficient — paper trading is still required."
            )
        if sum(returns) <= 0:
            return "FAILS — negative aggregate out-of-sample return"
        return (
            f"FAILS — only {consistency:.0%} of folds positive; the result "
            f"depends on a minority of periods"
        )


def _data_bounds(data: dict[str, BacktestData], timeframe: str) -> tuple[int, int]:
    starts, ends = [], []
    for entry in data.values():
        candles = entry.primary(timeframe)
        if candles:
            starts.append(candles[0].open_time)
            ends.append(candles[-1].close_time)
    if not starts:
        return 0, 0
    return min(starts), max(ends)


def _empty_report(reason: str) -> WalkForwardReport:
    return WalkForwardReport(
        folds=[],
        total_test_return=0.0,
        mean_test_return=0.0,
        median_test_return=0.0,
        std_test_return=0.0,
        positive_folds=0,
        negative_folds=0,
        consistency=0.0,
        mean_efficiency=0.0,
        worst_fold_return=0.0,
        best_fold_return=0.0,
        total_test_trades=0,
        insufficient_folds=0,
        verdict=f"INCONCLUSIVE — {reason}",
        warnings=[reason],
    )
