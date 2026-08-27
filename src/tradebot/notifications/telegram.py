"""Telegram notifications.

Three properties matter:

**It can never break trading.** Every send is wrapped; a Telegram outage, a bad
token or a rate limit is logged and counted, never raised into the trading loop.
The alternative — an exit that fails because a notification failed — is not
survivable.

**It never leaks secrets.** The bot token is in the URL, so the URL is never
logged. Message bodies are built from typed domain objects, not from formatted
log lines that might contain credentials.

**It is rate limited.** Telegram allows roughly 30 messages a second and about
20 a minute to one chat. A bot taking many trades an hour, plus risk alerts,
can exceed that; the queue coalesces and drops low-priority messages rather than
getting the account throttled at the moment a CRITICAL alert matters.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from tradebot.core.clock import format_duration
from tradebot.core.logging import get_logger
from tradebot.core.types import Direction, Position, RiskEvent, Trade

log = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class Priority(IntEnum):
    """Higher priority survives when the queue must shed load."""

    LOW = 0  # scan summaries
    NORMAL = 1  # trade opened/closed
    HIGH = 2  # risk alerts
    CRITICAL = 3  # kill switches, connection loss — never dropped


@dataclass(slots=True)
class QueuedMessage:
    text: str
    priority: Priority
    queued_at: float


class TelegramNotifier:
    """Async Telegram client with rate limiting and failure isolation."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        max_per_minute: int = 18,
        queue_size: int = 200,
        session: Any = None,
    ) -> None:
        self._token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token and chat_id)
        self.max_per_minute = max_per_minute

        self._queue: deque[QueuedMessage] = deque(maxlen=queue_size)
        self._sent_times: deque[float] = deque()
        self._session = session
        self._owns_session = session is None
        self._worker: asyncio.Task | None = None
        self._running = False

        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if not self.enabled:
            log.info("telegram_disabled", reason="no bot token or chat id configured")
            return
        self._running = True
        self._worker = asyncio.create_task(self._drain(), name="telegram")

    async def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        if self._session is not None and self._owns_session:
            with contextlib.suppress(Exception):
                await self._session.aclose()
        self._session = None

    # ------------------------------------------------------------------ #
    def send(self, text: str, priority: Priority = Priority.NORMAL) -> None:
        """Queue a message. Never blocks, never raises."""
        if not self.enabled:
            return
        if len(self._queue) == self._queue.maxlen:
            # Shed the lowest-priority message rather than the newest, so a
            # CRITICAL alert is never lost to a backlog of scan summaries.
            lowest = min(self._queue, key=lambda m: (m.priority, -m.queued_at))
            if lowest.priority < priority:
                self._queue.remove(lowest)
            else:
                self.dropped += 1
                return
        self._queue.append(QueuedMessage(text, priority, time.time()))

    async def _drain(self) -> None:
        while self._running:
            await asyncio.sleep(0.5)
            if not self._queue:
                continue
            if not self._can_send():
                continue
            # Highest priority first, then oldest.
            message = max(self._queue, key=lambda m: (m.priority, -m.queued_at))
            self._queue.remove(message)
            await self._deliver(message.text)

    def _can_send(self) -> bool:
        now = time.time()
        while self._sent_times and now - self._sent_times[0] > 60.0:
            self._sent_times.popleft()
        return len(self._sent_times) < self.max_per_minute

    async def _deliver(self, text: str) -> bool:
        """Send one message. Failure is logged and counted, never raised."""
        try:
            import httpx

            if self._session is None:
                self._session = httpx.AsyncClient(timeout=10.0)
                self._owns_session = True

            response = await self._session.post(
                f"{TELEGRAM_API}/bot{self._token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4096],
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if response.status_code == 200:
                self.sent += 1
                self._sent_times.append(time.time())
                return True

            self.failed += 1
            # The URL contains the bot token, so only the status is logged.
            self.last_error = f"HTTP {response.status_code}"
            log.warning("telegram_send_failed", status=response.status_code)
            return False
        except Exception as exc:  # noqa: BLE001 - notifications never break trading
            self.failed += 1
            self.last_error = type(exc).__name__
            log.warning("telegram_send_error", error_type=type(exc).__name__)
            return False

    # ------------------------------------------------------------------ #
    # Message builders
    # ------------------------------------------------------------------ #
    def notify_trade_opened(
        self, position: Position, risk_pct: float, expected_edge: float, score: float
    ) -> None:
        emoji = "🟢" if position.direction is Direction.LONG else "🔴"
        self.send(
            f"{emoji} <b>NEW {position.direction.value}</b>\n\n"
            f"Symbol: <code>{position.symbol}</code>\n"
            f"Strategy: {position.strategy}\n"
            f"Score: {score:.0f}\n"
            f"Entry: <code>{position.entry_price:.8g}</code>\n"
            f"SL: <code>{position.stop_loss:.8g}</code>\n"
            f"TP: <code>{position.take_profit:.8g}</code>\n"
            f"Qty: {position.quantity:.8g} ({position.leverage}x)\n"
            f"Risk: {risk_pct * 100:.2f}%\n"
            f"Expected Net Edge: {expected_edge * 100:.3f}%",
            Priority.NORMAL,
        )

    def notify_trade_closed(self, trade: Trade) -> None:
        emoji = "🔵" if trade.net_pnl >= 0 else "🟠"
        sign = "+" if trade.net_pnl >= 0 else ""
        self.send(
            f"{emoji} <b>CLOSED</b>\n\n"
            f"Symbol: <code>{trade.symbol}</code>\n"
            f"Strategy: {trade.strategy}\n"
            f"Duration: {format_duration(trade.duration_sec)}\n"
            f"Exit: {trade.exit_reason.value}\n\n"
            f"Gross PnL: {sign}{trade.gross_pnl:.4f}\n"
            f"Fees: -{trade.fees:.4f}\n"
            f"Funding: {-trade.funding:+.4f}\n"
            f"<b>Net PnL: {sign}{trade.net_pnl:.4f}</b> "
            f"({trade.r_multiple:+.2f}R)",
            Priority.NORMAL,
        )

    def notify_risk_alert(self, title: str, detail: str, drawdown: float | None = None) -> None:
        body = f"⚠️ <b>RISK ALERT</b>\n\n{title}\n{detail}"
        if drawdown is not None:
            body += f"\n\nDrawdown: {drawdown * 100:.2f}%"
        self.send(body, Priority.HIGH)

    def notify_kill_switch(self, switch: str, reason: str, equity: float | None = None) -> None:
        body = (
            f"🚨 <b>TRADING SUSPENDED</b>\n\n"
            f"Switch: <code>{switch}</code>\n"
            f"Reason: {reason}\n\n"
            f"New entries are disabled. Open positions are still managed "
            f"and can still be closed."
        )
        if equity is not None:
            body += f"\n\nEquity: {equity:.2f}"
        self.send(body, Priority.CRITICAL)

    def notify_system_alert(self, message: str, detail: str = "") -> None:
        self.send(
            f"🚨 <b>SYSTEM ALERT</b>\n\n{message}" + (f"\n\n{detail}" if detail else ""),
            Priority.CRITICAL,
        )

    def notify_risk_event(self, event: RiskEvent) -> None:
        priority = {
            "CRITICAL": Priority.CRITICAL,
            "ERROR": Priority.HIGH,
            "WARNING": Priority.HIGH,
            "INFO": Priority.LOW,
        }.get(event.severity, Priority.NORMAL)
        icon = "🚨" if event.severity == "CRITICAL" else "⚠️"
        self.send(
            f"{icon} <b>{event.event_type.value}</b>\n\n{event.message}"
            + (f"\n\nSymbol: <code>{event.symbol}</code>" if event.symbol else ""),
            priority,
        )

    def notify_startup(
        self, mode: str, equity: float, strategies: list[str], testnet: bool
    ) -> None:
        warning = "" if testnet else "\n\n<b>⚠️ LIVE — REAL MONEY</b>"
        self.send(
            f"▶️ <b>ENGINE STARTED</b>\n\n"
            f"Mode: <code>{mode}</code>\n"
            f"Testnet: {testnet}\n"
            f"Equity: {equity:.2f}\n"
            f"Strategies: {len(strategies)}{warning}",
            Priority.HIGH,
        )

    def notify_shutdown(self, reason: str, equity: float, trades_today: int) -> None:
        self.send(
            f"⏹️ <b>ENGINE STOPPED</b>\n\n"
            f"Reason: {reason}\n"
            f"Equity: {equity:.2f}\n"
            f"Trades today: {trades_today}",
            Priority.CRITICAL,
        )

    def notify_daily_summary(self, metrics: dict[str, Any]) -> None:
        self.send(
            f"📊 <b>DAILY SUMMARY</b>\n\n"
            f"Trades: {metrics.get('trades', 0)}\n"
            f"Win rate: {metrics.get('win_rate', 0) * 100:.1f}%\n"
            f"Net PnL: {metrics.get('net_pnl', 0):+.4f}\n"
            f"Fees: -{metrics.get('fees', 0):.4f}\n"
            f"Equity: {metrics.get('equity', 0):.2f}\n"
            f"Drawdown: {metrics.get('drawdown', 0) * 100:.2f}%",
            Priority.NORMAL,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sent": self.sent,
            "failed": self.failed,
            "dropped": self.dropped,
            "queued": len(self._queue),
            "last_error": self.last_error,
        }
