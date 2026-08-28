"""Health monitoring and safe mode.

Tracks whether each component is actually working, and degrades the system
deliberately when one is not. "The process is running" is not health: a bot that
is up but disconnected from Binance, or acting on ten-minute-old prices, is
worse than one that has stopped, because it still looks fine.

Safe mode is the middle ground between running and stopped:

* **new entries are disabled**
* **existing positions are still managed and can still be closed**

That asymmetry is the whole design. Every degraded state we can detect is one
where opening a position is unjustifiable and closing one may be urgent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tradebot.core.logging import get_logger

log = get_logger(__name__)


class ComponentState(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class Component:
    """One monitored subsystem."""

    name: str
    critical: bool  # does its failure force safe mode?
    timeout_sec: float
    last_heartbeat: float = 0.0
    state: ComponentState = ComponentState.UNKNOWN
    detail: str = ""
    failures: int = 0

    def beat(self, detail: str = "") -> None:
        self.last_heartbeat = time.time()
        self.state = ComponentState.HEALTHY
        self.detail = detail

    def fail(self, detail: str) -> None:
        self.state = ComponentState.FAILED
        self.detail = detail
        self.failures += 1

    def degrade(self, detail: str) -> None:
        self.state = ComponentState.DEGRADED
        self.detail = detail

    @property
    def age_sec(self) -> float:
        if self.last_heartbeat == 0.0:
            return float("inf")
        return time.time() - self.last_heartbeat

    def evaluate(self) -> ComponentState:
        """A component that has gone quiet is not healthy, whatever it last said."""
        if self.state is ComponentState.FAILED:
            return self.state
        if self.last_heartbeat == 0.0:
            return ComponentState.UNKNOWN
        if self.age_sec > self.timeout_sec:
            self.state = ComponentState.FAILED
            self.detail = f"no heartbeat for {self.age_sec:.0f}s"
            return self.state
        return self.state

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.evaluate().value,
            "critical": self.critical,
            "detail": self.detail,
            "age_sec": round(self.age_sec, 1) if self.last_heartbeat else None,
            "failures": self.failures,
        }


@dataclass(slots=True)
class HealthReport:
    healthy: bool
    safe_mode: bool
    safe_mode_reason: str
    components: list[dict[str, Any]]
    uptime_sec: float
    memory_mb: float
    cpu_pct: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "safe_mode": self.safe_mode,
            "safe_mode_reason": self.safe_mode_reason,
            "uptime_sec": round(self.uptime_sec, 1),
            "memory_mb": round(self.memory_mb, 1),
            "cpu_pct": round(self.cpu_pct, 1),
            "components": self.components,
            "warnings": self.warnings,
        }


class HealthMonitor:
    """Watches components and decides when to enter or leave safe mode."""

    def __init__(self, config: Any, on_safe_mode=None, on_recovered=None) -> None:
        self.config = config
        self.started_at = time.time()
        self.safe_mode = False
        self.safe_mode_reason = ""
        self._on_safe_mode = on_safe_mode
        self._on_recovered = on_recovered

        timeout = config.component_timeout_sec
        self.components: dict[str, Component] = {
            # Critical: without these, opening a position is unjustifiable.
            "market_data": Component("market_data", True, timeout),
            "exchange_rest": Component("exchange_rest", True, timeout),
            "risk_engine": Component("risk_engine", True, timeout),
            "execution": Component("execution", True, timeout),
            # Non-critical: their loss degrades observability, not safety.
            "user_stream": Component("user_stream", False, timeout * 2),
            # Critical by default: the audit trail is what makes a trade
            # reconstructable. Deployments that genuinely do not need it can
            # set health.database_critical to false.
            "database": Component(
                "database", bool(getattr(config, "database_critical", True)), timeout * 3
            ),
            "telegram": Component("telegram", False, timeout * 10),
            "dashboard": Component("dashboard", False, timeout * 10),
        }

    # ------------------------------------------------------------------ #
    def beat(self, name: str, detail: str = "") -> None:
        component = self.components.get(name)
        if component is not None:
            component.beat(detail)

    def fail(self, name: str, detail: str) -> None:
        component = self.components.get(name)
        if component is not None:
            component.fail(detail)
            log.error("component_failed", component=name, detail=detail)

    def degrade(self, name: str, detail: str) -> None:
        component = self.components.get(name)
        if component is not None:
            component.degrade(detail)

    # ------------------------------------------------------------------ #
    def check(self) -> HealthReport:
        """Evaluate every component and update safe mode."""
        warnings: list[str] = []
        failed_critical: list[str] = []

        for component in self.components.values():
            state = component.evaluate()
            if state is ComponentState.FAILED:
                if component.critical:
                    failed_critical.append(f"{component.name} ({component.detail})")
                else:
                    warnings.append(f"{component.name} has failed: {component.detail}")
            elif state is ComponentState.DEGRADED:
                warnings.append(f"{component.name} is degraded: {component.detail}")

        memory_mb, cpu_pct = _resource_usage()
        if memory_mb > self.config.memory_warn_mb:
            warnings.append(
                f"memory {memory_mb:.0f} MB exceeds the "
                f"{self.config.memory_warn_mb:.0f} MB warning level"
            )

        if failed_critical and not self.safe_mode:
            self._enter_safe_mode("; ".join(failed_critical))
        elif not failed_critical and self.safe_mode:
            self._leave_safe_mode()

        return HealthReport(
            healthy=not failed_critical and not self.safe_mode,
            safe_mode=self.safe_mode,
            safe_mode_reason=self.safe_mode_reason,
            components=[c.as_dict() for c in self.components.values()],
            uptime_sec=time.time() - self.started_at,
            memory_mb=memory_mb,
            cpu_pct=cpu_pct,
            warnings=warnings,
        )

    def _enter_safe_mode(self, reason: str) -> None:
        self.safe_mode = True
        self.safe_mode_reason = reason
        log.critical(
            "safe_mode_entered",
            reason=reason,
            message="new entries disabled; open positions are still "
            "managed and can still be closed",
        )
        if self._on_safe_mode is not None:
            self._on_safe_mode(reason)

    def _leave_safe_mode(self) -> None:
        previous = self.safe_mode_reason
        self.safe_mode = False
        self.safe_mode_reason = ""
        log.info("safe_mode_exited", previous_reason=previous)
        if self._on_recovered is not None:
            self._on_recovered()

    def force_safe_mode(self, reason: str) -> None:
        if not self.safe_mode:
            self._enter_safe_mode(reason)


def _resource_usage() -> tuple[float, float]:
    """Memory (MB) and CPU (%) without requiring psutil.

    Falls back to ``/proc`` and then to zeros — resource reporting must never be
    the thing that takes the bot down.
    """
    memory_mb = 0.0
    cpu_pct = 0.0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is kilobytes on Linux, bytes on macOS.
        import sys

        divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
        memory_mb = usage.ru_maxrss / divisor
        cpu_seconds = usage.ru_utime + usage.ru_stime
        cpu_pct = (
            min(100.0, cpu_seconds / max(1.0, time.process_time()) * 100.0)
            if time.process_time() > 0
            else 0.0
        )
    except (ImportError, OSError, ValueError):
        pass
    return memory_mb, cpu_pct
