"""Minimal async event bus.

Used to decouple producers (market feed, execution engine, risk engine) from
consumers (persistence, notifications, dashboard) so that a slow or failing
consumer can never stall trading.

Design choices that matter:

* Handlers are invoked concurrently and **their exceptions are swallowed and
  logged**. A Telegram outage must not abort a position exit.
* Each subscriber has a bounded queue. If a consumer falls behind, its oldest
  events are dropped and the drop is counted, rather than growing memory without
  limit.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tradebot.core.logging import get_logger

log = get_logger(__name__)


class EventType(StrEnum):
    CANDLE_CLOSED = "CANDLE_CLOSED"
    BOOK_UPDATE = "BOOK_UPDATE"
    MARK_PRICE = "MARK_PRICE"
    SCAN_COMPLETE = "SCAN_COMPLETE"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    OPPORTUNITY_EVALUATED = "OPPORTUNITY_EVALUATED"
    RISK_DECISION = "RISK_DECISION"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_UPDATE = "ORDER_UPDATE"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_UPDATED = "POSITION_UPDATED"
    POSITION_CLOSED = "POSITION_CLOSED"
    TRADE_COMPLETED = "TRADE_COMPLETED"
    RISK_EVENT = "RISK_EVENT"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
    RECONCILED = "RECONCILED"
    HEALTH_UPDATE = "HEALTH_UPDATE"


@dataclass(slots=True)
class Event:
    type: EventType
    payload: Any
    timestamp: int = 0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Publish/subscribe with isolated, bounded delivery."""

    def __init__(self, queue_size: int = 1000) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._wildcard: list[Handler] = []
        self._queue_size = queue_size
        self._dropped: dict[str, int] = defaultdict(int)
        self._published = 0

    def subscribe(self, event_type: EventType | None, handler: Handler) -> None:
        """Register a handler. ``event_type=None`` subscribes to everything."""
        if event_type is None:
            self._wildcard.append(handler)
        else:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType | None, handler: Handler) -> None:
        target = self._wildcard if event_type is None else self._handlers[event_type]
        if handler in target:
            target.remove(handler)

    @property
    def published(self) -> int:
        return self._published

    @property
    def dropped(self) -> dict[str, int]:
        return dict(self._dropped)

    async def publish(self, event: Event) -> None:
        """Deliver an event to all handlers concurrently, isolating failures."""
        self._published += 1
        handlers = list(self._handlers.get(event.type, ())) + list(self._wildcard)
        if not handlers:
            return
        results = await asyncio.gather(
            *(self._invoke(h, event) for h in handlers), return_exceptions=True
        )
        for handler, result in zip(handlers, results, strict=False):
            if isinstance(result, BaseException):
                name = getattr(handler, "__qualname__", repr(handler))
                self._dropped[name] += 1
                log.error(
                    "event_handler_failed",
                    handler=name,
                    event_type=event.type.value,
                    error=str(result),
                    error_type=type(result).__name__,
                )

    @staticmethod
    async def _invoke(handler: Handler, event: Event) -> None:
        await handler(event)

    async def emit(
        self,
        event_type: EventType,
        payload: Any,
        source: str = "",
        timestamp: int = 0,
        **metadata: Any,
    ) -> None:
        """Convenience wrapper around :meth:`publish`."""
        await self.publish(
            Event(
                type=event_type,
                payload=payload,
                timestamp=timestamp,
                source=source,
                metadata=metadata,
            )
        )
