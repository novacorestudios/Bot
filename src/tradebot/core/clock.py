"""Time abstraction.

Nothing in the engine calls ``time.time()`` directly. In LIVE/PAPER the clock is
the wall clock corrected by the measured offset against Binance's server time;
in BACKTEST it is a virtual clock driven by bar timestamps. This is what makes
the same engine code testable and replayable.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...
    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Wall clock, optionally corrected by an exchange time offset."""

    def __init__(self, offset_ms: int = 0) -> None:
        self._offset_ms = offset_ms

    @property
    def offset_ms(self) -> int:
        return self._offset_ms

    def set_offset(self, offset_ms: int) -> None:
        """Set (server_time - local_time) in milliseconds."""
        self._offset_ms = offset_ms

    def now_ms(self) -> int:
        return int(time.time() * 1000) + self._offset_ms

    def now(self) -> float:
        return time.time() + self._offset_ms / 1000.0

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class VirtualClock:
    """Deterministic clock for backtests and tests.

    ``sleep`` advances the virtual time instantly rather than waiting, so tests
    involving timeouts and cooldowns run at full speed.
    """

    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = start_ms

    def now_ms(self) -> int:
        return self._now_ms

    def now(self) -> float:
        return self._now_ms / 1000.0

    def set(self, now_ms: int) -> None:
        self._now_ms = now_ms

    def advance(self, seconds: float) -> None:
        self._now_ms += int(seconds * 1000)

    def advance_ms(self, ms: int) -> None:
        self._now_ms += ms

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        await asyncio.sleep(0)  # yield to the loop without real delay


def ms_to_iso(ms: int) -> str:
    """UTC ISO-8601 string from epoch milliseconds. For logs and reports."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()


def format_duration(seconds: float) -> str:
    """Human-friendly duration, e.g. '8m 21s' — used in Telegram messages."""
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
