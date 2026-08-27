"""Kill switches.

Automatic circuit breakers. Every one of them blocks **new entries** only —
none of them ever blocks an exit. That asymmetry is deliberate: the situations
that trip a kill switch are exactly the situations where being unable to close a
position would be catastrophic.

The switches, and why each exists:

| Switch | Catches |
|---|---|
| daily loss | a bad day compounding into a disaster |
| hourly loss | a fast bleed that a daily limit would notice too late |
| drawdown | slow erosion from the equity peak |
| consecutive losses | a strategy that has stopped working, before the loss limits notice |
| API errors | a broken connection producing garbage decisions |
| rejected orders | a systematic sizing or filter bug burning the order rate limit |
| slippage | execution quality collapsing, which invalidates every edge estimate |
| reconciliation | local state disagreeing with the exchange — the most dangerous state to keep trading in |
| stale data | acting on prices that are no longer true |
| connection | no reliable view of the account |

A tripped switch re-arms either after ``auto_rearm_seconds`` or on the next
trading day, depending on the switch. Loss-based switches deliberately do NOT
auto-re-arm within the day: if the day's budget is gone, it is gone.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from tradebot.core.config import KillSwitchConfig, RiskConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import RiskEvent, RiskEventType

log = get_logger(__name__)


class SwitchName(StrEnum):
    DAILY_LOSS = "DAILY_LOSS"
    HOURLY_LOSS = "HOURLY_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    API_ERRORS = "API_ERRORS"
    REJECTED_ORDERS = "REJECTED_ORDERS"
    SLIPPAGE = "SLIPPAGE"
    RECONCILIATION = "RECONCILIATION"
    STALE_DATA = "STALE_DATA"
    CONNECTION = "CONNECTION"
    ABNORMAL_MARKET = "ABNORMAL_MARKET"
    MANUAL = "MANUAL"


#: Switches that must NOT re-arm automatically inside the same trading day.
_DAY_BOUND = {SwitchName.DAILY_LOSS, SwitchName.MAX_DRAWDOWN}


@dataclass(slots=True)
class TrippedSwitch:
    name: SwitchName
    reason: str
    tripped_at: float
    rearm_at: float | None
    data: dict[str, float] = field(default_factory=dict)


class KillSwitchManager:
    """Evaluates circuit breakers and reports whether entries are permitted."""

    def __init__(self, config: KillSwitchConfig, risk_config: RiskConfig, clock=None) -> None:
        self.config = config
        self.risk_config = risk_config
        self._clock = clock
        self.tripped: dict[SwitchName, TrippedSwitch] = {}

        # Rolling event windows.
        self._api_errors: deque[float] = deque()
        self._rejections: deque[float] = deque()
        self._slippages: deque[tuple[float, float]] = deque(maxlen=50)

        self.consecutive_losses = 0
        self.reconciliation_mismatches = 0
        self.peak_equity = 0.0
        self.day_start_equity = 0.0
        self.hour_start_equity = 0.0
        self._day_index = -1
        self._hour_index = -1

        self.events: list[RiskEvent] = []

    def _now(self) -> float:
        return self._clock.now() if self._clock is not None else time.time()

    # ------------------------------------------------------------------ #
    # State updates
    # ------------------------------------------------------------------ #
    def update_equity(self, equity: float) -> None:
        """Record equity, rolling the day/hour baselines when they change."""
        now = self._now()
        reset_hour = self.risk_config.day_reset_hour_utc
        day_index = int((now - reset_hour * 3600) // 86_400)
        hour_index = int(now // 3600)

        if day_index != self._day_index:
            self._day_index = day_index
            self.day_start_equity = equity
            self.consecutive_losses = 0
            # A new trading day clears the day-bound switches.
            for name in list(self.tripped):
                if name in _DAY_BOUND:
                    self._clear(name, "new trading day")

        if hour_index != self._hour_index:
            self._hour_index = hour_index
            self.hour_start_equity = equity
            self._clear(SwitchName.HOURLY_LOSS, "new hour")

        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.day_start_equity <= 0:
            self.day_start_equity = equity
        if self.hour_start_equity <= 0:
            self.hour_start_equity = equity

    def record_api_error(self) -> None:
        self._api_errors.append(self._now())

    def record_order_rejection(self) -> None:
        self._rejections.append(self._now())

    def record_slippage(self, slippage: float) -> None:
        self._slippages.append((self._now(), abs(slippage)))

    def record_trade_result(self, won: bool) -> None:
        self.consecutive_losses = 0 if won else self.consecutive_losses + 1

    def record_reconciliation_mismatch(self) -> None:
        self.reconciliation_mismatches += 1

    def clear_reconciliation_mismatches(self) -> None:
        self.reconciliation_mismatches = 0
        self._clear(SwitchName.RECONCILIATION, "reconciliation clean")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(
        self, equity: float, data_age_sec: float = 0.0, connected: bool = True
    ) -> list[TrippedSwitch]:
        """Check every switch. Returns those newly tripped."""
        self.update_equity(equity)
        cfg = self.config
        risk = self.risk_config
        now = self._now()
        newly: list[TrippedSwitch] = []

        # -- loss limits ----------------------------------------------------- #
        if self.day_start_equity > 0:
            daily_loss = safe_div(self.day_start_equity - equity, self.day_start_equity, 0.0)
            if daily_loss >= risk.max_daily_loss:
                newly += self._trip(
                    SwitchName.DAILY_LOSS,
                    f"daily loss {daily_loss * 100:.2f}% reached the "
                    f"{risk.max_daily_loss * 100:.2f}% limit",
                    rearm=None,
                    daily_loss=daily_loss,
                )

        if self.hour_start_equity > 0:
            hourly_loss = safe_div(self.hour_start_equity - equity, self.hour_start_equity, 0.0)
            if hourly_loss >= risk.max_hourly_loss:
                newly += self._trip(
                    SwitchName.HOURLY_LOSS,
                    f"hourly loss {hourly_loss * 100:.2f}% reached the "
                    f"{risk.max_hourly_loss * 100:.2f}% limit",
                    rearm=None,
                    hourly_loss=hourly_loss,
                )

        if self.peak_equity > 0:
            drawdown = safe_div(self.peak_equity - equity, self.peak_equity, 0.0)
            if drawdown >= risk.max_drawdown:
                newly += self._trip(
                    SwitchName.MAX_DRAWDOWN,
                    f"drawdown {drawdown * 100:.2f}% from the peak of "
                    f"{self.peak_equity:.2f} reached the "
                    f"{risk.max_drawdown * 100:.2f}% limit",
                    rearm=None,
                    drawdown=drawdown,
                    peak=self.peak_equity,
                )

        if self.consecutive_losses >= risk.max_consecutive_losses:
            newly += self._trip(
                SwitchName.CONSECUTIVE_LOSSES,
                f"{self.consecutive_losses} consecutive losses reached the "
                f"limit of {risk.max_consecutive_losses}",
                rearm=now + cfg.auto_rearm_seconds if cfg.auto_rearm_seconds else None,
                consecutive_losses=float(self.consecutive_losses),
            )

        # -- operational ------------------------------------------------------ #
        self._prune(self._api_errors, now, 300.0)
        if len(self._api_errors) >= cfg.max_api_errors_per_5min:
            newly += self._trip(
                SwitchName.API_ERRORS,
                f"{len(self._api_errors)} API errors in 5 minutes "
                f"(limit {cfg.max_api_errors_per_5min}); decisions may be based "
                f"on stale or wrong data",
                rearm=now + cfg.auto_rearm_seconds if cfg.auto_rearm_seconds else None,
                errors=float(len(self._api_errors)),
            )

        self._prune(self._rejections, now, 3600.0)
        if len(self._rejections) >= cfg.max_rejected_orders_per_hour:
            newly += self._trip(
                SwitchName.REJECTED_ORDERS,
                f"{len(self._rejections)} orders rejected in an hour "
                f"(limit {cfg.max_rejected_orders_per_hour}); likely a sizing "
                f"or filter bug",
                rearm=now + cfg.auto_rearm_seconds if cfg.auto_rearm_seconds else None,
                rejections=float(len(self._rejections)),
            )

        recent_slippage = [
            value for timestamp, value in self._slippages if now - timestamp <= 900.0
        ]
        if len(recent_slippage) >= 3:
            average = sum(recent_slippage) / len(recent_slippage)
            if average >= cfg.max_slippage:
                newly += self._trip(
                    SwitchName.SLIPPAGE,
                    f"average slippage {average * 100:.3f}% over the last "
                    f"{len(recent_slippage)} fills exceeds "
                    f"{cfg.max_slippage * 100:.3f}%; every edge estimate is "
                    f"now optimistic",
                    rearm=now + cfg.auto_rearm_seconds if cfg.auto_rearm_seconds else None,
                    average_slippage=average,
                )

        if self.reconciliation_mismatches >= cfg.max_reconciliation_mismatches:
            newly += self._trip(
                SwitchName.RECONCILIATION,
                f"{self.reconciliation_mismatches} reconciliation mismatches; "
                f"local state disagrees with the exchange",
                rearm=None,
                mismatches=float(self.reconciliation_mismatches),
            )

        # -- connectivity ------------------------------------------------------ #
        if data_age_sec > cfg.ws_stale_seconds:
            newly += self._trip(
                SwitchName.STALE_DATA,
                f"market data is {data_age_sec:.0f}s old (limit {cfg.ws_stale_seconds:.0f}s)",
                rearm=None,
                data_age=data_age_sec,
            )
        else:
            self._clear(SwitchName.STALE_DATA, "market data is fresh again")

        if not connected:
            newly += self._trip(SwitchName.CONNECTION, "exchange connection lost", rearm=None)
        else:
            self._clear(SwitchName.CONNECTION, "exchange connection restored")

        self._expire(now)
        return newly

    # ------------------------------------------------------------------ #
    def trip_manually(self, reason: str) -> None:
        self._trip(SwitchName.MANUAL, reason, rearm=None)

    def trip_abnormal_market(self, symbol: str, detail: str) -> None:
        self._trip(
            SwitchName.ABNORMAL_MARKET,
            f"{symbol}: {detail}",
            rearm=self._now() + self.config.auto_rearm_seconds
            if self.config.auto_rearm_seconds
            else None,
        )

    def reset(self, name: SwitchName | None = None) -> None:
        """Operator override. Clears one switch or all of them."""
        if name is None:
            for switch in list(self.tripped):
                self._clear(switch, "manual reset")
        else:
            self._clear(name, "manual reset")

    # ------------------------------------------------------------------ #
    @property
    def entries_allowed(self) -> bool:
        """False when ANY switch is tripped. Exits are never blocked."""
        return not self.tripped

    @property
    def active(self) -> tuple[SwitchName, ...]:
        return tuple(self.tripped)

    def blocking_reason(self) -> str:
        if not self.tripped:
            return ""
        return "; ".join(f"{s.name.value}: {s.reason}" for s in self.tripped.values())

    def status(self) -> dict[str, object]:
        return {
            "entries_allowed": self.entries_allowed,
            "tripped": [
                {
                    "switch": switch.name.value,
                    "reason": switch.reason,
                    "tripped_at": switch.tripped_at,
                    "rearm_at": switch.rearm_at,
                }
                for switch in self.tripped.values()
            ],
            "consecutive_losses": self.consecutive_losses,
            "reconciliation_mismatches": self.reconciliation_mismatches,
            "peak_equity": round(self.peak_equity, 4),
            "day_start_equity": round(self.day_start_equity, 4),
            "api_errors_5min": len(self._api_errors),
            "rejections_1h": len(self._rejections),
        }

    # ------------------------------------------------------------------ #
    def _trip(
        self, name: SwitchName, reason: str, rearm: float | None, **data: float
    ) -> list[TrippedSwitch]:
        if name in self.tripped:
            return []
        switch = TrippedSwitch(
            name=name, reason=reason, tripped_at=self._now(), rearm_at=rearm, data=data
        )
        self.tripped[name] = switch
        log.critical(
            "kill_switch_tripped", switch=name.value, reason=reason, rearm_at=rearm, **data
        )
        self.events.append(
            RiskEvent(
                event_type=RiskEventType.KILL_SWITCH_TRIGGERED,
                severity="CRITICAL",
                message=f"{name.value}: {reason}",
                timestamp=int(self._now() * 1000),
                data=dict(data),
            )
        )
        return [switch]

    def _clear(self, name: SwitchName, why: str) -> None:
        if self.tripped.pop(name, None) is not None:
            log.warning("kill_switch_cleared", switch=name.value, reason=why)
            self.events.append(
                RiskEvent(
                    event_type=RiskEventType.KILL_SWITCH_CLEARED,
                    severity="WARNING",
                    message=f"{name.value} cleared: {why}",
                    timestamp=int(self._now() * 1000),
                )
            )

    def _expire(self, now: float) -> None:
        for name, switch in list(self.tripped.items()):
            if switch.rearm_at is not None and now >= switch.rearm_at:
                self._clear(name, "auto re-arm interval elapsed")

    @staticmethod
    def _prune(window: deque[float], now: float, span: float) -> None:
        cutoff = now - span
        while window and window[0] < cutoff:
            window.popleft()
