"""Capital preservation modes.

A 75 USDT account cannot absorb a bad day the way a large one can. A 10%
drawdown is 7.50 USDT — small in absolute terms, but it is 15 trades' worth of
risk at 0.5%, and recovering it requires an 11% gain. The arithmetic of
drawdown is the reason this module exists: losses compound against you faster
than gains compound for you, so the response to a losing streak must be to risk
less, not to trade the same size and hope.

Four modes, each a strict tightening of the one before:

| Mode | Trigger | Risk | Positions | New entries |
|---|---|---|---|---|
| ``NORMAL`` | default | 100% | full | yes |
| ``CAUTIOUS`` | mild drawdown or a losing streak | 60% | −1 | yes |
| ``DEFENSIVE`` | serious drawdown | 35% | 1 | only exceptional scores |
| ``HALTED`` | the daily loss limit | 0% | 0 | no |

Two design decisions worth stating plainly:

* **Exits are never gated.** No mode, ``HALTED`` included, may prevent closing
  a position. Preservation restricts what we take on, never what we can get out
  of — a mode that blocked exits would turn a bad day into a catastrophic one.
* **De-escalation is deliberately slower than escalation, and requires
  evidence.** Tightening happens the moment a threshold is crossed; loosening
  requires both recovery below the *lower* threshold (a hysteresis band, so a
  drawdown oscillating around a line does not flip the mode every cycle) and a
  minimum dwell time. Without both, one lucky trade would restore full size in
  exactly the conditions that caused the drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.logging import get_logger

log = get_logger(__name__)


class PreservationMode(StrEnum):
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"
    HALTED = "HALTED"

    @property
    def rank(self) -> int:
        return _RANK[self]

    @property
    def allows_entries(self) -> bool:
        return self is not PreservationMode.HALTED


_RANK = {
    PreservationMode.NORMAL: 0,
    PreservationMode.CAUTIOUS: 1,
    PreservationMode.DEFENSIVE: 2,
    PreservationMode.HALTED: 3,
}


@dataclass(frozen=True, slots=True)
class ModeLimits:
    """What a mode permits."""

    risk_multiplier: float
    max_positions_delta: int  # subtracted from the configured maximum
    max_positions_cap: int  # absolute cap; 0 means "no additional cap"
    min_opportunity_score: float  # 0 means "use the configured minimum"

    def positions_allowed(self, configured: int) -> int:
        allowed = configured + self.max_positions_delta
        if self.max_positions_cap > 0:
            allowed = min(allowed, self.max_positions_cap)
        return max(0, allowed)


LIMITS: dict[PreservationMode, ModeLimits] = {
    PreservationMode.NORMAL: ModeLimits(1.0, 0, 0, 0.0),
    PreservationMode.CAUTIOUS: ModeLimits(0.6, -1, 0, 0.0),
    # One position at a time, and only for opportunities that are genuinely
    # exceptional rather than merely acceptable.
    PreservationMode.DEFENSIVE: ModeLimits(0.35, -2, 1, 85.0),
    PreservationMode.HALTED: ModeLimits(0.0, -99, 0, 101.0),
}


@dataclass(slots=True)
class PreservationState:
    """The current mode and why it is what it is."""

    mode: PreservationMode
    reason: str
    since: float
    drawdown: float = 0.0
    daily_loss: float = 0.0
    consecutive_losses: int = 0
    capital_reserve: float = 0.0

    @property
    def limits(self) -> ModeLimits:
        return LIMITS[self.mode]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "risk_multiplier": self.limits.risk_multiplier,
            "drawdown": round(self.drawdown, 4),
            "daily_loss": round(self.daily_loss, 4),
            "consecutive_losses": self.consecutive_losses,
            "capital_reserve": round(self.capital_reserve, 4),
            "entries_allowed": self.mode.allows_entries,
        }


class CapitalPreservation:
    """Decides which preservation mode the account is in.

    It only ever *reports* a mode and a risk multiplier; the risk engine applies
    them. Keeping the decision and its application apart means the thresholds
    can be tested against a table of scenarios without constructing an engine.
    """

    def __init__(
        self,
        config: Any,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or SystemClock()

        self.state = PreservationState(
            mode=PreservationMode.NORMAL,
            reason="normal operation",
            since=self.clock.now(),
        )
        self.transitions = 0
        self.time_in_mode: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        drawdown: float,
        daily_loss: float,
        consecutive_losses: int,
        equity: float = 0.0,
    ) -> PreservationState:
        """Recompute the mode from the account's current condition.

        ``drawdown`` and ``daily_loss`` are positive fractions of equity: 0.03
        means "3% down".
        """
        target, reason = self._target_mode(drawdown, daily_loss, consecutive_losses)
        current = self.state.mode

        # Tightening is immediate; loosening must earn it.
        escalating = target.rank > current.rank
        if target is not current and (escalating or self._may_relax(drawdown)):
            self._switch(target, reason, drawdown, daily_loss, consecutive_losses)

        self.state.drawdown = drawdown
        self.state.daily_loss = daily_loss
        self.state.consecutive_losses = consecutive_losses
        self.state.capital_reserve = self.reserve_for(equity)
        return self.state

    # ------------------------------------------------------------------ #
    def _target_mode(
        self, drawdown: float, daily_loss: float, consecutive_losses: int
    ) -> tuple[PreservationMode, str]:
        cfg = self.config

        if daily_loss >= cfg.halt_daily_loss:
            return (
                PreservationMode.HALTED,
                f"daily loss {daily_loss * 100:.2f}% reached the "
                f"{cfg.halt_daily_loss * 100:.2f}% limit",
            )
        if drawdown >= cfg.halt_drawdown:
            return (
                PreservationMode.HALTED,
                f"drawdown {drawdown * 100:.2f}% reached the {cfg.halt_drawdown * 100:.2f}% limit",
            )
        if drawdown >= cfg.defensive_drawdown:
            return (
                PreservationMode.DEFENSIVE,
                f"drawdown {drawdown * 100:.2f}% at or beyond the defensive threshold",
            )
        if consecutive_losses >= cfg.defensive_consecutive_losses:
            return (
                PreservationMode.DEFENSIVE,
                f"{consecutive_losses} consecutive losses",
            )
        if drawdown >= cfg.cautious_drawdown:
            return (
                PreservationMode.CAUTIOUS,
                f"drawdown {drawdown * 100:.2f}% at or beyond the cautious threshold",
            )
        if consecutive_losses >= cfg.cautious_consecutive_losses:
            return (
                PreservationMode.CAUTIOUS,
                f"{consecutive_losses} consecutive losses",
            )
        return PreservationMode.NORMAL, "normal operation"

    def _may_relax(self, drawdown: float) -> bool:
        """Loosening needs both a dwell time and a recovery below the band.

        The hysteresis is the important part: without it a drawdown sitting on
        a threshold flips the mode on every evaluation, and the account gets
        full size back in exactly the conditions that shrank it.

        ``drawdown`` is this cycle's value, not the stored one: reading the
        stored value would judge the decision on the previous cycle's number
        and relax exactly one evaluation late.
        """
        held = self.clock.now() - self.state.since
        if held < self.config.min_mode_duration_sec:
            return False

        band = self.config.recovery_hysteresis
        if self.state.mode is PreservationMode.HALTED:
            # Only a manual reset or a new trading day leaves HALTED. A
            # recovering drawdown is not evidence that the cause is gone.
            return False
        if self.state.mode is PreservationMode.DEFENSIVE:
            return drawdown <= self.config.defensive_drawdown - band
        if self.state.mode is PreservationMode.CAUTIOUS:
            return drawdown <= self.config.cautious_drawdown - band
        return True

    def _switch(
        self,
        target: PreservationMode,
        reason: str,
        drawdown: float,
        daily_loss: float,
        consecutive_losses: int,
    ) -> None:
        now = self.clock.now()
        previous = self.state.mode
        self.time_in_mode[previous.value] = self.time_in_mode.get(previous.value, 0.0) + (
            now - self.state.since
        )
        self.transitions += 1
        self.state = PreservationState(
            mode=target,
            reason=reason,
            since=now,
            drawdown=drawdown,
            daily_loss=daily_loss,
            consecutive_losses=consecutive_losses,
        )
        log_fn = log.critical if target is PreservationMode.HALTED else log.warning
        log_fn(
            "preservation_mode_changed",
            previous=previous.value,
            mode=target.value,
            reason=reason,
            risk_multiplier=LIMITS[target].risk_multiplier,
            drawdown=round(drawdown, 4),
        )

    # ------------------------------------------------------------------ #
    # What the risk engine asks
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> PreservationMode:
        return self.state.mode

    @property
    def risk_multiplier(self) -> float:
        return self.state.limits.risk_multiplier

    @property
    def entries_allowed(self) -> bool:
        return self.state.mode.allows_entries

    def max_positions(self, configured: int) -> int:
        return self.state.limits.positions_allowed(configured)

    def min_opportunity_score(self, configured: float) -> float:
        floor = self.state.limits.min_opportunity_score
        return max(configured, floor) if floor > 0 else configured

    def reserve_for(self, equity: float) -> float:
        """Capital held back from deployment.

        A reserve is not idle money: it is what pays the funding, fees and
        adverse margin moves on positions that are already open. An account
        that deploys every cent has no buffer between a normal adverse move and
        a liquidation.
        """
        return max(0.0, equity * self.config.capital_reserve_fraction)

    def deployable(self, equity: float) -> float:
        return max(0.0, equity - self.reserve_for(equity))

    # ------------------------------------------------------------------ #
    def reset(self, reason: str = "manual reset") -> None:
        """Return to NORMAL. Used by the daily reset and by an operator."""
        if self.state.mode is PreservationMode.NORMAL:
            return
        self._switch(PreservationMode.NORMAL, reason, 0.0, 0.0, 0)

    def stats(self) -> dict[str, Any]:
        return {
            **self.state.as_dict(),
            "seconds_in_mode": round(self.clock.now() - self.state.since, 1),
            "transitions": self.transitions,
            "time_in_mode": {k: round(v, 1) for k, v in self.time_in_mode.items()},
        }
