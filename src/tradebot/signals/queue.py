"""The opportunity queue.

The V1 engine evaluated candidates in scan-rank order and traded the first one
that passed risk. With four position slots and twenty-five candidates, that
means the best opportunity of the cycle loses to a merely adequate one that
happened to sit higher in the *market* ranking — a ranking of how tradable a
symbol is, not of how good this particular trade is (AUDIT_REPORT.md M-7).

The queue fixes the ordering: score everything first, then spend the remaining
slots best-first.

Two properties matter as much as the ordering:

* **Opportunities expire.** A signal computed on a 5-minute bar is not still
  valid ten minutes later, and a queue that does not expire is a queue that
  eventually acts on a stale thesis. Every entry carries the time it was
  created and is dropped once it ages past its time-to-live.
* **The queue never trades.** It holds and orders candidates; the risk engine
  still decides on each one, and it can refuse every entry in the queue. An
  empty result is a legitimate, common outcome — no opportunities means no
  trades, never a forced fill.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.logging import get_logger
from tradebot.core.types import Direction
from tradebot.signals.pipeline import Opportunity

log = get_logger(__name__)


@dataclass(slots=True)
class QueuedOpportunity:
    """One opportunity, with the bookkeeping the queue needs."""

    opportunity: Opportunity
    queued_ms: int
    ttl_sec: float
    audit: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    last_rejection: str = ""

    @property
    def symbol(self) -> str:
        return self.opportunity.symbol

    @property
    def strategy(self) -> str:
        return self.opportunity.strategy

    @property
    def direction(self) -> Direction:
        return self.opportunity.direction

    @property
    def score(self) -> float:
        return self.opportunity.opportunity_score.total

    @property
    def expected_net_edge(self) -> float:
        return self.opportunity.expected_net_edge

    def age_sec(self, now_ms: int) -> float:
        return max(0.0, (now_ms - self.queued_ms) / 1000.0)

    def is_expired(self, now_ms: int) -> bool:
        return self.age_sec(now_ms) > self.ttl_sec

    def as_dict(self, now_ms: int) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction.value,
            "score": round(self.score, 2),
            "expected_net_edge": round(self.expected_net_edge, 6),
            "age_sec": round(self.age_sec(now_ms), 1),
            "attempts": self.attempts,
            "last_rejection": self.last_rejection,
        }


def rank_key(entry: QueuedOpportunity) -> tuple[float, float, float]:
    """Best first.

    Score leads because it already folds in market quality, consensus, cost and
    correlation. Expected net edge breaks ties, because between two equally
    scored trades the one that keeps more after costs is strictly better.
    Confidence breaks the remaining ties.
    """
    return (
        -entry.score,
        -entry.expected_net_edge,
        -entry.opportunity.signal.confidence,
    )


class OpportunityQueue:
    """Holds this cycle's opportunities and hands them out best-first.

    It is rebuilt every signal cycle rather than carried across cycles: a
    candidate that is still good will be re-queued by the next evaluation, and
    one that is not should not linger.
    """

    def __init__(self, ttl_sec: float = 60.0, max_size: int = 50, clock: Clock | None = None):
        self.ttl_sec = ttl_sec
        self.max_size = max_size
        self.clock = clock or SystemClock()

        self._entries: dict[str, QueuedOpportunity] = {}
        self.queued = 0
        self.expired = 0
        self.replaced = 0
        self.dropped_full = 0

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._entries

    def __iter__(self) -> Iterator[QueuedOpportunity]:
        return iter(self.ranked())

    @property
    def is_empty(self) -> bool:
        return not self._entries

    # ------------------------------------------------------------------ #
    def add(self, opportunity: Opportunity, audit: dict[str, Any] | None = None) -> bool:
        """Queue an opportunity. Returns False when it was not accepted.

        One entry per symbol: two strategies firing on the same symbol are two
        views of one trade, not two trades, so the better-scoring one wins.
        """
        now = self.clock.now_ms()
        self._expire(now)

        symbol = opportunity.symbol
        entry = QueuedOpportunity(
            opportunity=opportunity,
            queued_ms=now,
            ttl_sec=self.ttl_sec,
            audit=dict(audit or {}),
        )

        existing = self._entries.get(symbol)
        if existing is not None:
            if rank_key(entry) >= rank_key(existing):
                return False
            self.replaced += 1
            self._entries[symbol] = entry
            return True

        if len(self._entries) >= self.max_size:
            # Full: displace the worst entry, but only if this one beats it.
            worst = max(self._entries.values(), key=rank_key)
            if rank_key(entry) >= rank_key(worst):
                self.dropped_full += 1
                return False
            del self._entries[worst.symbol]
            self.dropped_full += 1

        self._entries[symbol] = entry
        self.queued += 1
        return True

    def ranked(self) -> list[QueuedOpportunity]:
        """Live entries, best first."""
        self._expire(self.clock.now_ms())
        return sorted(self._entries.values(), key=rank_key)

    def best(self) -> QueuedOpportunity | None:
        entries = self.ranked()
        return entries[0] if entries else None

    def take(self, limit: int) -> list[QueuedOpportunity]:
        """Remove and return up to ``limit`` best entries.

        ``limit`` is the number of position slots actually free. Zero free slots
        returns nothing — the queue is emptied by the next cycle's rebuild, not
        by forcing trades through.
        """
        if limit <= 0:
            return []
        taken = self.ranked()[:limit]
        for entry in taken:
            self._entries.pop(entry.symbol, None)
        return taken

    def discard(self, symbol: str, reason: str = "") -> None:
        """Drop one symbol, e.g. after the risk engine refused it outright."""
        entry = self._entries.pop(symbol, None)
        if entry is not None and reason:
            log.debug("opportunity_discarded", symbol=symbol, reason=reason)

    def record_attempt(self, symbol: str, rejection: str) -> None:
        """Note that this entry was offered to risk and refused."""
        entry = self._entries.get(symbol)
        if entry is not None:
            entry.attempts += 1
            entry.last_rejection = rejection

    def clear(self) -> None:
        self._entries.clear()

    # ------------------------------------------------------------------ #
    def _expire(self, now_ms: int) -> None:
        stale = [s for s, e in self._entries.items() if e.is_expired(now_ms)]
        for symbol in stale:
            del self._entries[symbol]
        if stale:
            self.expired += len(stale)
            log.debug("opportunities_expired", count=len(stale), symbols=stale[:5])

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._entries),
            "queued_total": self.queued,
            "expired": self.expired,
            "replaced": self.replaced,
            "dropped_full": self.dropped_full,
            "ttl_sec": self.ttl_sec,
        }

    def report(self, limit: int = 20) -> list[dict[str, Any]]:
        now = self.clock.now_ms()
        return [entry.as_dict(now) for entry in self.ranked()[:limit]]
