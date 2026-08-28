"""Observational recording of what each gate saw, for diagnosis only.

The rejection counters answer *how many* candidates each gate removed. They
cannot answer *how close* those candidates were, and the two questions have
different answers. `INSUFFICIENT_CONSENSUS` is emitted from two separate
branches — "fewer than N strategies agreed" and "N agreed but scored below the
floor" — and a single counter cannot tell them apart, which is precisely the
distinction between "the strategies rarely agree" and "the consensus bar is
where the candidates die".

So this module records the **scalar each gate thresholded**, per candidate.
Every sensitivity question then becomes an offline re-thresholding of recorded
values rather than another engine run.

**This is observational and must stay so.** A recorder is `None` everywhere by
default; the call sites are `if self.recorder is not None: ...record(...)` and
record nothing back into the decision. No value here is read by any gate, and
attaching a recorder cannot change a decision, a fill, or a PnL. That property
is enforced by `tests/integration/test_diagnostics_are_observational.py`, which
runs the same backtest with and without a recorder and compares the results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateRecord:
    """One evaluated candidate, and whatever each gate had computed about it.

    Fields are `None` when the value did not exist at the point the candidate
    was rejected — not zero. A candidate rejected for breadth never had a
    consensus score computed, and recording 0.0 there would invent a data point
    sitting at the bottom of the distribution.
    """

    symbol: str
    stage: str
    reason: str | None
    agreeing: int | None = None
    consensus: float | None = None
    expected_net: float | None = None
    stop_distance: float | None = None
    raw_quantity: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateRecorder:
    """Collects `CandidateRecord`s. Deliberately dumb: append, never decide."""

    records: list[CandidateRecord] = field(default_factory=list)

    def record(
        self,
        symbol: str,
        stage: str,
        reason: str | None,
        *,
        agreeing: int | None = None,
        consensus: float | None = None,
        expected_net: float | None = None,
        stop_distance: float | None = None,
        raw_quantity: float | None = None,
    ) -> None:
        self.records.append(
            CandidateRecord(
                symbol=symbol,
                stage=stage,
                reason=reason,
                agreeing=agreeing,
                consensus=consensus,
                expected_net=expected_net,
                stop_distance=stop_distance,
                raw_quantity=raw_quantity,
            )
        )

    # -- read-side helpers, used by the diagnostic report ----------------- #
    def by_reason(self) -> dict[str, int]:
        return dict(Counter(r.reason or "ACCEPTED" for r in self.records))

    def values(self, attribute: str, reason: str | None = None) -> list[float]:
        """Every non-None value of `attribute`, optionally for one reason."""
        out: list[float] = []
        for record in self.records:
            if reason is not None and record.reason != reason:
                continue
            value = getattr(record, attribute)
            if value is not None:
                out.append(float(value))
        return out
