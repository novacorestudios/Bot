"""Whether a backtest's inputs are good enough to believe its output.

A backtest cannot tell you its input was wrong. It will compute a Sharpe ratio
from a series with a six-hour hole, placeholder exchange filters and no funding
data, and the number will look exactly like a real one.

So trust is decided **before** the run, explicitly, and travels with the result.
Three states:

* ``TRUSTED`` — every input requirement met.
* ``UNTRUSTED`` — the run may proceed and the numbers may be inspected, but
  they are not evidence. Every report says so, at the top.
* ``REFUSED`` — the run does not happen.

The distinction that matters is between *missing* and *damaged*:

* **Damaged** data (out-of-order bars, impossible OHLC) is refused outright.
  There is no honest way to use it.
* **Missing** inputs — exchange filters, funding history — downgrade the run to
  UNTRUSTED rather than blocking it, because inspecting a rough run is a
  legitimate thing to want. What is *not* legitimate is that happening
  silently, which is what `strict=False` used to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tradebot.core.logging import get_logger

log = get_logger(__name__)


class TrustLevel(StrEnum):
    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"
    REFUSED = "REFUSED"


@dataclass(slots=True)
class TrustReport:
    """Why a run is or is not believable."""

    level: TrustLevel = TrustLevel.TRUSTED
    blockers: list[str] = field(default_factory=list)
    downgrades: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)

    @property
    def is_trusted(self) -> bool:
        return self.level is TrustLevel.TRUSTED

    @property
    def may_run(self) -> bool:
        return self.level is not TrustLevel.REFUSED

    def block(self, reason: str) -> None:
        self.blockers.append(reason)
        self.level = TrustLevel.REFUSED

    def downgrade(self, reason: str) -> None:
        self.downgrades.append(reason)
        if self.level is TrustLevel.TRUSTED:
            self.level = TrustLevel.UNTRUSTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "blockers": list(self.blockers),
            "downgrades": list(self.downgrades),
            "overrides": list(self.overrides),
        }

    def lines(self) -> list[str]:
        out = [f"  Data trust          {self.level.value}"]
        out += [f"    BLOCKER   {reason}" for reason in self.blockers]
        out += [f"    DOWNGRADE {reason}" for reason in self.downgrades]
        out += [f"    OVERRIDE  {reason}" for reason in self.overrides]
        return out

    def banner(self) -> str:
        """The line that goes at the top of any report built from this run."""
        if self.level is TrustLevel.TRUSTED:
            return ""
        if self.level is TrustLevel.REFUSED:
            return "**REFUSED** — the inputs were not fit to run on."
        return "**UNTRUSTED** — these numbers are not evidence. " + "; ".join(
            self.downgrades + self.overrides
        )


def evaluate_trust(
    data: dict[str, Any],
    quality: list[Any],
    required_timeframes: list[str],
    funding_enabled: bool,
    have_exchange_info: bool,
    allow_degraded: bool = False,
) -> TrustReport:
    """Decide whether this dataset may be believed.

    ``allow_degraded`` is the explicit override the brief asks for: it lets a
    run proceed on gappy data, and forces the result to UNTRUSTED so nobody can
    quote it as a result.
    """
    report = TrustReport()

    if not data:
        report.block("no symbols loaded")
        return report

    # -- damaged data: refused ------------------------------------------- #
    unusable = [q for q in quality if getattr(q, "status", None) and not q.usable]
    if unusable:
        names = ", ".join(f"{q.symbol}/{q.interval}" for q in unusable[:5])
        message = f"{len(unusable)} dataset(s) failed validation ({names})"
        if allow_degraded:
            report.overrides.append(f"{message} — allowed by --allow-degraded")
            report.downgrade(message)
        else:
            report.block(message)

    # -- missing timeframes: refused, because the run is meaningless ------ #
    missing_tf: dict[str, list[str]] = {}
    for symbol, entry in data.items():
        present = {tf for tf, bars in entry.candles.items() if bars}
        absent = sorted(set(required_timeframes) - present)
        if absent:
            missing_tf[symbol] = absent
    if missing_tf:
        sample = ", ".join(f"{s}:{','.join(tf)}" for s, tf in sorted(missing_tf.items())[:3])
        report.block(
            f"{len(missing_tf)} symbol(s) are missing timeframes the strategies "
            f"read ({sample}) — a low trade count would mean NO DATA, not NO EDGE"
        )

    # -- degraded data: usable, but the result is not evidence ------------ #
    degraded = [
        q
        for q in quality
        if getattr(q, "status", None) and q.usable and str(q.status) == "DEGRADED"
    ]
    if degraded:
        total_missing = sum(getattr(q, "missing_bars", 0) for q in degraded)
        report.downgrade(f"{len(degraded)} dataset(s) have gaps ({total_missing} bars missing)")

    # -- missing metadata: downgrade -------------------------------------- #
    if not have_exchange_info:
        report.downgrade(
            "no exchangeInfo — tick size, step size and MINIMUM NOTIONAL are "
            "placeholders, so the run may take positions the exchange rejects"
        )

    if funding_enabled:
        without = sorted(s for s, entry in data.items() if not entry.funding_rates)
        if without:
            report.downgrade(
                f"{len(without)} symbol(s) have no funding history "
                f"({', '.join(without[:5])}) — held positions are under-costed"
            )

    log.info(
        "data_trust_evaluated",
        level=report.level.value,
        blockers=len(report.blockers),
        downgrades=len(report.downgrades),
    )
    return report
