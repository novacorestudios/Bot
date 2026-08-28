"""Data validation: the stage between download and backtest.

A backtest cannot tell you that its input was wrong. It will happily compute a
Sharpe ratio from a series with a six-hour hole in it, and the hole will read as
a price jump the strategies interpret as a real move. The result looks like a
number and is not one.

So validation is a **separate, explicit stage** with its own artefact, rather
than a side effect of loading. Its output is a report an operator reads before
believing anything downstream.

The distinction that runs through this module is between problems that make data
**unusable** and problems that make it **imperfect**:

* *Unusable* — out of order, duplicated, non-positive prices, `high < low`,
  inconsistent OHLC. These corrupt indicators silently and cannot be repaired
  without inventing data, so they fail the dataset.
* *Imperfect* — gaps. Exchanges have outages; Binance has had multi-hour ones.
  Requiring perfection would reject all real history. Gaps are measured,
  reported, and left for the operator to judge against their tolerance.

Nothing here repairs data by interpolation. A synthesised bar is a fabricated
price, and a backtest that trades on fabricated prices is worse than no backtest
because it carries the same authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import Candle, Timeframe

log = get_logger(__name__)


class QualityStatus(StrEnum):
    OK = "OK"  # complete enough to trust
    DEGRADED = "DEGRADED"  # usable, with documented gaps
    UNUSABLE = "UNUSABLE"  # do not backtest on this


@dataclass(slots=True)
class Gap:
    """A run of missing bars."""

    after_ms: int
    before_ms: int
    missing_bars: int

    @property
    def duration_sec(self) -> float:
        return (self.before_ms - self.after_ms) / 1000.0


@dataclass(slots=True)
class ValidationReport:
    """Everything found in one symbol/interval series."""

    symbol: str
    interval: str
    rows: int = 0
    start_ms: int = 0
    end_ms: int = 0

    duplicates: int = 0
    out_of_order: int = 0
    invalid_timestamps: int = 0
    non_positive_prices: int = 0
    inconsistent_ohlc: int = 0
    negative_volume: int = 0
    zero_volume_bars: int = 0

    expected_bars: int = 0
    missing_bars: int = 0
    gaps: list[Gap] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def largest_gap_bars(self) -> int:
        return max((g.missing_bars for g in self.gaps), default=0)

    @property
    def coverage(self) -> float:
        """Fraction of the expected bars that are actually present."""
        return safe_div(self.rows, self.expected_bars, 1.0 if self.rows else 0.0)

    @property
    def status(self) -> QualityStatus:
        if self.errors:
            return QualityStatus.UNUSABLE
        if self.warnings or self.missing_bars:
            return QualityStatus.DEGRADED
        return QualityStatus.OK

    @property
    def usable(self) -> bool:
        return self.status is not QualityStatus.UNUSABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "rows": self.rows,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "invalid_timestamps": self.invalid_timestamps,
            "non_positive_prices": self.non_positive_prices,
            "inconsistent_ohlc": self.inconsistent_ohlc,
            "negative_volume": self.negative_volume,
            "zero_volume_bars": self.zero_volume_bars,
            "expected_bars": self.expected_bars,
            "missing_bars": self.missing_bars,
            "gaps": len(self.gaps),
            "largest_gap_bars": self.largest_gap_bars,
            "coverage": round(self.coverage, 6),
            "status": self.status.value,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def row(self) -> dict[str, Any]:
        """One line for the quality table the brief asks for in §5."""
        return {
            "SYMBOL": self.symbol,
            "INTERVAL": self.interval,
            "DATA_START": self.start_ms,
            "DATA_END": self.end_ms,
            "ROWS": self.rows,
            "MISSING": self.missing_bars,
            "DUPLICATES": self.duplicates,
            "GAPS": len(self.gaps),
            "COVERAGE": round(self.coverage, 4),
            "QUALITY_STATUS": self.status.value,
        }


def interval_ms(interval: str) -> int:
    """Milliseconds per bar, from the project's own timeframe table."""
    return Timeframe(interval).seconds * 1000


def validate_candles(
    symbol: str,
    interval: str,
    candles: list[Candle],
    max_gap_bars: int = 10,
    max_missing_fraction: float = 0.02,
) -> ValidationReport:
    """Check one series. Returns findings; never raises, never repairs.

    ``max_gap_bars`` and ``max_missing_fraction`` decide when a gap stops being
    an exchange hiccup and starts being a hole that would distort the result.
    Both are the caller's judgement, not this module's.
    """
    report = ValidationReport(symbol=symbol, interval=interval, rows=len(candles))
    if not candles:
        report.errors.append("no rows")
        return report

    step = interval_ms(interval)
    report.start_ms = candles[0].open_time
    report.end_ms = candles[-1].close_time

    seen: set[int] = set()
    previous_open = -1

    for candle in candles:
        # -- timestamps ---------------------------------------------------- #
        if candle.open_time <= 0 or candle.close_time <= candle.open_time:
            report.invalid_timestamps += 1
        if candle.open_time in seen:
            report.duplicates += 1
        seen.add(candle.open_time)
        if candle.open_time < previous_open:
            report.out_of_order += 1
        previous_open = max(previous_open, candle.open_time)

        # -- prices --------------------------------------------------------- #
        if min(candle.open, candle.high, candle.low, candle.close) <= 0:
            report.non_positive_prices += 1
        elif not (
            candle.high >= max(candle.open, candle.close)
            and candle.low <= min(candle.open, candle.close)
            and candle.high >= candle.low
        ):
            # An OHLC bar whose high is not the highest is not a bar; it is a
            # transcription error, and indicators built on it are meaningless.
            report.inconsistent_ohlc += 1

        # -- volume --------------------------------------------------------- #
        if candle.volume < 0:
            report.negative_volume += 1
        elif candle.volume == 0:
            report.zero_volume_bars += 1

    # -- continuity ---------------------------------------------------------- #
    span = report.end_ms - report.start_ms
    report.expected_bars = max(1, round(span / step)) if step > 0 else len(candles)

    ordered = sorted(seen)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        delta = later - earlier
        if delta <= step:
            continue
        missing = round(delta / step) - 1
        if missing > 0:
            report.gaps.append(Gap(after_ms=earlier, before_ms=later, missing_bars=missing))
            report.missing_bars += missing

    # -- verdicts ------------------------------------------------------------ #
    if report.duplicates:
        report.errors.append(f"{report.duplicates} duplicate bars")
    if report.out_of_order:
        report.errors.append(f"{report.out_of_order} out-of-order bars")
    if report.invalid_timestamps:
        report.errors.append(f"{report.invalid_timestamps} invalid timestamps")
    if report.non_positive_prices:
        report.errors.append(f"{report.non_positive_prices} bars with a non-positive price")
    if report.inconsistent_ohlc:
        report.errors.append(f"{report.inconsistent_ohlc} bars where OHLC is inconsistent")
    if report.negative_volume:
        report.errors.append(f"{report.negative_volume} bars with negative volume")

    if report.largest_gap_bars > max_gap_bars:
        report.warnings.append(
            f"largest gap is {report.largest_gap_bars} bars (tolerance {max_gap_bars})"
        )
    missing_fraction = safe_div(report.missing_bars, report.expected_bars, 0.0)
    if missing_fraction > max_missing_fraction:
        report.warnings.append(
            f"{missing_fraction:.2%} of expected bars are missing "
            f"(tolerance {max_missing_fraction:.2%})"
        )
    # A dead symbol produces bars with no trading in them, which score as
    # tradable and are not.
    zero_fraction = safe_div(report.zero_volume_bars, report.rows, 0.0)
    if zero_fraction > 0.10:
        report.warnings.append(f"{zero_fraction:.1%} of bars have zero volume")

    return report


def quality_table(reports: list[ValidationReport]) -> list[dict[str, Any]]:
    """The §5 report, worst first — the rows an operator needs to look at."""
    order = {QualityStatus.UNUSABLE: 0, QualityStatus.DEGRADED: 1, QualityStatus.OK: 2}
    return [
        r.row() for r in sorted(reports, key=lambda r: (order[r.status], -r.missing_bars, r.symbol))
    ]


def summarise(reports: list[ValidationReport]) -> dict[str, Any]:
    return {
        "datasets": len(reports),
        "usable": sum(1 for r in reports if r.usable),
        "unusable": sum(1 for r in reports if not r.usable),
        "degraded": sum(1 for r in reports if r.status is QualityStatus.DEGRADED),
        "total_rows": sum(r.rows for r in reports),
        "total_missing_bars": sum(r.missing_bars for r in reports),
        "total_gaps": sum(len(r.gaps) for r in reports),
        "symbols": sorted({r.symbol for r in reports}),
    }
