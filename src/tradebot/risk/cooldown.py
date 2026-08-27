"""Cooldown.

After a position closes, the same symbol is not immediately re-entered. The
reason is not superstition: the conditions that produced the exit are usually
still present a minute later, so an immediate re-entry tends to be the same
trade again — and if the first one lost, the second is likely to lose for the
same reason while paying a second round trip in fees.

The cooldown is dynamic:

* longer after a **loss** than after a win
* multiplied for each **consecutive** loss on that symbol, so a symbol that
  keeps failing is progressively stepped away from
* scaled by **volatility**, because a volatile market invalidates a thesis
  faster and re-forms one faster too
* tracked **per strategy** as well as per symbol when configured, so one
  strategy failing on a symbol does not silence the others that might be right

Cooldowns never block exits, and never block a position already open.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from tradebot.core.config import CooldownConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp

log = get_logger(__name__)


@dataclass(slots=True)
class CooldownEntry:
    until: float
    reason: str
    consecutive_losses: int = 0
    applied_seconds: float = 0.0


@dataclass(slots=True)
class CooldownStatus:
    active: bool
    remaining_sec: float = 0.0
    reason: str = ""
    key: str = ""


class CooldownManager:
    """Per-symbol and per-strategy re-entry delays."""

    def __init__(self, config: CooldownConfig, clock=None) -> None:
        self.config = config
        self._clock = clock
        self._entries: dict[str, CooldownEntry] = {}
        self._consecutive: dict[str, int] = {}

    def _now(self) -> float:
        return self._clock.now() if self._clock is not None else time.time()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(symbol: str, strategy: str | None = None) -> str:
        return f"{symbol}:{strategy}" if strategy else symbol

    def duration_for(
        self, won: bool, symbol: str, strategy: str | None = None, volatility: float = 0.0
    ) -> float:
        """How long the cooldown should last, in seconds."""
        cfg = self.config
        base = cfg.after_win_seconds if won else cfg.after_loss_seconds

        if not won:
            streak = self._consecutive.get(self._key(symbol, strategy), 0)
            if streak > 0:
                base *= cfg.consecutive_loss_multiplier**streak

        # Higher volatility shortens the cooldown: theses form and invalidate
        # faster, so a fixed wait is disproportionately long.
        if volatility > 0:
            reference = 0.005
            scale = clamp(reference / volatility, 0.5, 2.0)
            base *= scale

        return clamp(base, 0.0, float(cfg.max_seconds))

    def register_close(
        self,
        symbol: str,
        won: bool,
        strategy: str | None = None,
        volatility: float = 0.0,
        reason: str = "",
    ) -> float:
        """Start a cooldown after a position closes. Returns its duration."""
        cfg = self.config
        now = self._now()

        symbol_key = self._key(symbol)
        strategy_key = self._key(symbol, strategy) if (strategy and cfg.per_strategy) else None

        # Duration is computed BEFORE the streak is updated, so the first loss
        # gets the base cooldown and only the SECOND consecutive loss is
        # multiplied. Incrementing first would double the very first cooldown.
        duration = self.duration_for(won, symbol, strategy, volatility)

        for key in (symbol_key, strategy_key):
            if key is None:
                continue
            self._consecutive[key] = 0 if won else self._consecutive.get(key, 0) + 1
        detail = reason or ("win" if won else "loss")

        for key in (symbol_key, strategy_key):
            if key is None:
                continue
            self._entries[key] = CooldownEntry(
                until=now + duration,
                reason=f"{detail}; {duration:.0f}s cooldown",
                consecutive_losses=self._consecutive.get(key, 0),
                applied_seconds=duration,
            )

        log.info(
            "cooldown_started",
            symbol=symbol,
            strategy=strategy,
            won=won,
            seconds=round(duration, 1),
            consecutive_losses=self._consecutive.get(symbol_key, 0),
        )
        return duration

    def register_rejection(
        self, symbol: str, seconds: float, reason: str = "order rejected"
    ) -> None:
        """Short cooldown after an order rejection, to avoid a retry storm."""
        now = self._now()
        self._entries[self._key(symbol)] = CooldownEntry(
            until=now + seconds, reason=reason, applied_seconds=seconds
        )

    # ------------------------------------------------------------------ #
    def check(self, symbol: str, strategy: str | None = None) -> CooldownStatus:
        """Is this symbol (optionally for this strategy) still cooling down?"""
        now = self._now()
        for key in (self._key(symbol), self._key(symbol, strategy) if strategy else None):
            if key is None:
                continue
            entry = self._entries.get(key)
            if entry is None:
                continue
            if now >= entry.until:
                del self._entries[key]
                continue
            return CooldownStatus(
                active=True, remaining_sec=entry.until - now, reason=entry.reason, key=key
            )
        return CooldownStatus(active=False)

    def is_active(self, symbol: str, strategy: str | None = None) -> bool:
        return self.check(symbol, strategy).active

    def clear(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._entries.clear()
            self._consecutive.clear()
            return
        for key in list(self._entries):
            if key == symbol or key.startswith(f"{symbol}:"):
                del self._entries[key]

    def active_cooldowns(self) -> dict[str, float]:
        """Remaining seconds per key — surfaced on the dashboard."""
        now = self._now()
        return {
            key: round(entry.until - now, 1)
            for key, entry in self._entries.items()
            if entry.until > now
        }
