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

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from tradebot.core.logging import get_logger
from tradebot.data.validation import QualityStatus

log = get_logger(__name__)


@runtime_checkable
class DatasetQuality(Protocol):
    """What the trust gate needs to know about one symbol/interval series.

    This is the **one** contract, and it is a Protocol on purpose. Two classes
    implement it — :class:`tradebot.backtesting.data.DataQuality` (load time)
    and :class:`tradebot.data.validation.ValidationReport` (acquisition time) —
    and the gate treats them identically.

    Before V3.2 there was no contract. The gate read ``status``, ``interval``
    and ``missing_bars`` off whatever it was handed, through
    ``getattr(q, "status", None)``. The loader's ``DataQuality`` had none of
    those attributes, so every guard evaluated to ``False``, and a series with
    impossible OHLC and a 500-bar hole was reported ``TRUSTED``. Typing the
    parameter turns that class of mistake into a mypy error instead of a
    confident wrong number.
    """

    @property
    def symbol(self) -> str: ...

    @property
    def interval(self) -> str: ...

    @property
    def missing_bars(self) -> int: ...

    @property
    def status(self) -> QualityStatus: ...

    @property
    def usable(self) -> bool: ...


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
    placeholder_filter_symbols: list[str] = field(default_factory=list)

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
            "exchange_filter_fidelity": {
                "trusted": not self.placeholder_filter_symbols,
                "placeholder_symbols": list(self.placeholder_filter_symbols),
            },
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
    quality: Sequence[DatasetQuality],
    required_timeframes: list[str],
    funding_enabled: bool,
    have_exchange_info: bool,
    allow_degraded: bool = False,
) -> TrustReport:
    """Decide whether this dataset may be believed.

    The decision table, in full:

    ==========================================  ===================  ==========
    condition                                   without the flag     with it
    ==========================================  ===================  ==========
    no symbols loaded                           REFUSED              REFUSED
    UNUSABLE series (structural corruption)     REFUSED              REFUSED
    a required timeframe is absent              REFUSED              REFUSED
    DEGRADED series (gaps, duplicates)          REFUSED              UNTRUSTED
    no exchangeInfo                             UNTRUSTED            UNTRUSTED
    funding enabled, no funding history         UNTRUSTED            UNTRUSTED
    everything present and clean                TRUSTED              TRUSTED
    ==========================================  ===================  ==========

    Two properties hold by construction, and both are tested:

    * ``--allow-degraded`` cannot rescue structurally corrupt data. Before V3.2
      it could, which meant one flag turned a refusal into a running backtest
      over bars with impossible OHLC.
    * **No input condition produces TRUSTED except a clean one.** An override
      can only move REFUSED to UNTRUSTED; nothing moves anything to TRUSTED.
    """
    report = TrustReport()

    if not data:
        report.block("no symbols loaded")
        return report

    # -- damaged data: refused, and no flag overrides this ---------------- #
    unusable = [q for q in quality if q.status is QualityStatus.UNUSABLE]
    if unusable:
        names = ", ".join(f"{q.symbol}/{q.interval}" for q in unusable[:5])
        report.block(
            f"{len(unusable)} dataset(s) are structurally corrupt ({names}) — "
            f"there is no honest way to backtest on them"
        )

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

    # -- degraded data: runs only when the operator says so --------------- #
    degraded = [q for q in quality if q.status is QualityStatus.DEGRADED]
    if degraded:
        total_missing = sum(q.missing_bars for q in degraded)
        names = ", ".join(f"{q.symbol}/{q.interval}" for q in degraded[:5])
        message = (
            f"{len(degraded)} dataset(s) are DEGRADED ({names}); "
            f"{total_missing} bar(s) missing in total"
        )
        if allow_degraded:
            report.overrides.append(f"{message} — accepted via --allow-degraded")
            report.downgrade(message)
        else:
            report.block(f"{message}. Re-run with --allow-degraded to proceed as UNTRUSTED")

    # -- missing metadata: downgrade -------------------------------------- #
    placeholder_symbols = sorted(
        symbol
        for symbol, entry in data.items()
        if getattr(entry, "exchange_filter_provenance", "GENUINE_EXCHANGE_INFO") == "PLACEHOLDER"
    )
    # Backwards compatibility for direct callers that only know whether an
    # exchangeInfo snapshot exists. Loaded datasets carry per-symbol provenance
    # and are therefore authoritative when the snapshot has partial coverage.
    if not have_exchange_info and not placeholder_symbols:
        placeholder_symbols = sorted(data)

    if placeholder_symbols:
        report.placeholder_filter_symbols = placeholder_symbols
        report.downgrade(
            f"{len(placeholder_symbols)} symbol(s) use placeholder exchange filters "
            f"({', '.join(placeholder_symbols)}) — tick size, step size and "
            "MINIMUM NOTIONAL are not genuine exchangeInfo metadata, so the run "
            "may take positions the exchange rejects"
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
