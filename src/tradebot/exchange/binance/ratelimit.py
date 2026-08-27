"""Weight-aware rate limiting for the Binance USDⓈ-M Futures API.

Binance enforces three independent budgets, and exceeding any of them earns a
``429`` and then an IP ban (``418``):

* request **weight** per minute (default 2400 for futures)
* **order** count per 10 seconds (default 300)
* **order** count per minute (default 1200)

Two mechanisms work together here:

1. A local prediction: every request reserves its weight before being sent, so
   bursts are shaped rather than discovered after the fact.
2. Server truth: the ``X-MBX-USED-WEIGHT-1M`` and ``X-MBX-ORDER-COUNT-*``
   response headers are authoritative and override the local count, because
   other processes may share the same IP and the local view can drift.

The limiter deliberately reserves at a fraction of the true limit
(``safety_factor``) so that a burst of retries cannot push us over the line.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class _Window:
    """A sliding-window counter."""

    limit: int
    window_sec: float
    events: deque[tuple[float, int]] = field(default_factory=deque)
    server_value: int | None = None
    server_at: float = 0.0

    def prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def used(self, now: float) -> int:
        self.prune(now)
        local = sum(count for _, count in self.events)
        # Trust the server's number while it is fresh; it accounts for other
        # clients on the same IP that we cannot see.
        if self.server_value is not None and now - self.server_at < self.window_sec:
            return max(local, self.server_value)
        return local

    def add(self, now: float, amount: int) -> None:
        self.events.append((now, amount))

    def set_server(self, now: float, value: int) -> None:
        self.server_value = value
        self.server_at = now

    def wait_time(self, now: float, amount: int, effective_limit: int) -> float:
        """Seconds to wait before `amount` more units fit inside the window."""
        self.prune(now)
        if self.used(now) + amount <= effective_limit:
            return 0.0
        if not self.events:
            return self.window_sec
        # Wait until the oldest event ages out.
        return max(0.0, self.events[0][0] + self.window_sec - now)


class RateLimiter:
    """Async, weight-aware limiter shared by every REST call."""

    def __init__(
        self,
        weight_limit: int = 2400,
        order_limit_10s: int = 300,
        order_limit_1m: int = 1200,
        safety_factor: float = 0.85,
        clock: Clock | None = None,
    ) -> None:
        self.safety_factor = safety_factor
        self._weight = _Window(weight_limit, 60.0)
        self._orders_10s = _Window(order_limit_10s, 10.0)
        self._orders_1m = _Window(order_limit_1m, 60.0)
        self._lock = asyncio.Lock()
        self._banned_until = 0.0
        # Reading the time and sleeping must go through the SAME source, or a
        # virtual clock advances while the limiter waits on the real one.
        self._clock: Clock = clock or SystemClock()
        self.waits = 0
        self.total_wait_sec = 0.0

    def _now(self) -> float:
        return self._clock.now()

    def _effective(self, window: _Window) -> int:
        return max(1, int(window.limit * self.safety_factor))

    def configure_from_exchange_info(self, rate_limits: list[dict]) -> None:
        """Adopt the limits Binance reports in ``exchangeInfo``."""
        for entry in rate_limits:
            kind = entry.get("rateLimitType")
            interval = entry.get("interval")
            num = int(entry.get("intervalNum", 1))
            limit = int(entry.get("limit", 0))
            if limit <= 0:
                continue
            if kind == "REQUEST_WEIGHT" and interval == "MINUTE":
                self._weight = _Window(limit, 60.0 * num)
            elif kind == "ORDERS" and interval == "SECOND":
                self._orders_10s = _Window(limit, float(num))
            elif kind == "ORDERS" and interval == "MINUTE":
                self._orders_1m = _Window(limit, 60.0 * num)
        log.info(
            "rate_limits_configured",
            weight_per_min=self._weight.limit,
            orders_per_10s=self._orders_10s.limit,
            orders_per_min=self._orders_1m.limit,
            safety_factor=self.safety_factor,
        )

    async def acquire(self, weight: int = 1, is_order: bool = False) -> None:
        """Block until this request fits inside every budget."""
        while True:
            async with self._lock:
                now = self._now()

                if now < self._banned_until:
                    delay = self._banned_until - now
                else:
                    delay = self._weight.wait_time(now, weight, self._effective(self._weight))
                    if is_order:
                        delay = max(
                            delay,
                            self._orders_10s.wait_time(now, 1, self._effective(self._orders_10s)),
                            self._orders_1m.wait_time(now, 1, self._effective(self._orders_1m)),
                        )

                if delay <= 0:
                    self._weight.add(now, weight)
                    if is_order:
                        self._orders_10s.add(now, 1)
                        self._orders_1m.add(now, 1)
                    return

            self.waits += 1
            self.total_wait_sec += delay
            log.debug("rate_limit_wait", seconds=round(delay, 3), weight=weight, is_order=is_order)
            await self._clock.sleep(min(delay, 60.0))

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Adopt the server's authoritative usage counters."""
        now = self._now()
        for key, value in headers.items():
            lowered = key.lower()
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if lowered.startswith("x-mbx-used-weight-1m"):
                self._weight.set_server(now, parsed)
            elif lowered.startswith("x-mbx-order-count-10s"):
                self._orders_10s.set_server(now, parsed)
            elif lowered.startswith("x-mbx-order-count-1m"):
                self._orders_1m.set_server(now, parsed)

    def register_ban(self, retry_after_sec: float) -> None:
        """Called on 418/429: refuse to send anything until the ban expires."""
        self._banned_until = self._now() + retry_after_sec
        log.warning("rate_limit_ban_registered", retry_after_sec=retry_after_sec)

    @property
    def is_banned(self) -> bool:
        return self._now() < self._banned_until

    def usage(self) -> dict[str, float]:
        """Current usage as fractions of the limits — surfaced on the dashboard."""
        now = self._now()
        return {
            "weight_used": self._weight.used(now),
            "weight_limit": self._weight.limit,
            "weight_pct": self._weight.used(now) / max(1, self._weight.limit),
            "orders_10s": self._orders_10s.used(now),
            "orders_1m": self._orders_1m.used(now),
            "banned": float(self.is_banned),
            "waits": self.waits,
            "total_wait_sec": round(self.total_wait_sec, 2),
        }
