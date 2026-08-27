"""Persistence layer.

Two rules shape this module:

**1. A database problem must never stop or delay a trade.** Writes are buffered
and flushed off the hot path, and every write is wrapped so a failure is logged
and counted rather than raised into the trading loop. Losing an audit row is
regrettable; failing to exit a position because the disk is full is not
survivable.

**2. Reads are for humans and analysis, not for decisions.** The engine's
authoritative state comes from the exchange and from memory. Nothing in the
trading path blocks on a query.

The buffer is bounded. If the database is unavailable for a long time the oldest
buffered rows are dropped and the drop is counted, rather than growing memory
until the process dies — a bot that OOMs with open positions is far worse than
one with a gap in its logs.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from tradebot.core.logging import get_logger
from tradebot.core.types import (
    Candidate,
    Order,
    Position,
    RiskEvent,
    Signal,
    Trade,
    TradingMode,
)
from tradebot.database.models import (
    PRUNABLE,
    Base,
    DecisionRecord,
    EquityRecord,
    MarketSnapshotRecord,
    OrderRecord,
    PositionRecord,
    RiskEventRecord,
    SignalRecord,
    StrategyMetricRecord,
    SystemEventRecord,
    TradeRecord,
)

log = get_logger(__name__)


@dataclass(slots=True)
class DatabaseHealth:
    available: bool = True
    writes: int = 0
    failures: int = 0
    dropped: int = 0
    buffered: int = 0
    last_error: str | None = None
    last_flush_ms: int = 0


class Repository:
    """Buffered, failure-isolated persistence."""

    def __init__(
        self,
        database_url: str,
        mode: TradingMode = TradingMode.PAPER,
        buffer_size: int = 5000,
        flush_interval_sec: float = 2.0,
    ) -> None:
        self.database_url = database_url
        self.mode = mode
        self.flush_interval_sec = flush_interval_sec

        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._buffer: deque[Base] = deque(maxlen=buffer_size)
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._lock = asyncio.Lock()

        self.health = DatabaseHealth()

    # ------------------------------------------------------------------ #
    async def connect(self) -> None:
        """Open the engine and create tables if they do not exist."""
        if "sqlite" in self.database_url:
            from pathlib import Path

            path = self.database_url.split("///")[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(self.database_url, echo=False, pool_pre_ping=True)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop(), name="db:flush")
        self.health.available = True
        log.info("database_connected", url=_redact_url(self.database_url))

    async def close(self) -> None:
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None
        await self.flush()
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    # ------------------------------------------------------------------ #
    def _enqueue(self, record: Base) -> None:
        """Buffer a row. Never raises, never blocks."""
        if len(self._buffer) == self._buffer.maxlen:
            self.health.dropped += 1
            if self.health.dropped % 100 == 1:
                log.warning(
                    "database_buffer_full",
                    dropped=self.health.dropped,
                    message="dropping the oldest audit rows; trading continues unaffected",
                )
        self._buffer.append(record)
        self.health.buffered = len(self._buffer)

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.flush_interval_sec)
            with contextlib.suppress(Exception):
                await self.flush()

    async def flush(self) -> int:
        """Write buffered rows. Failures are logged, never raised."""
        if self._session_factory is None or not self._buffer:
            return 0

        async with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()

        if not batch:
            return 0

        try:
            async with self._session_factory() as session:
                session.add_all(batch)
                await session.commit()
        except Exception as exc:  # noqa: BLE001 - persistence must not break trading
            self.health.failures += 1
            self.health.available = False
            self.health.last_error = str(exc)[:200]
            log.error(
                "database_write_failed",
                error=str(exc)[:200],
                rows=len(batch),
                message="trading continues; these audit rows are lost",
            )
            return 0

        self.health.writes += len(batch)
        self.health.available = True
        self.health.buffered = len(self._buffer)
        return len(batch)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def record_trade(self, trade: Trade, realised_edge: float = 0.0) -> None:
        self._enqueue(
            TradeRecord(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                strategy=trade.strategy,
                direction=trade.direction.value,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                leverage=trade.leverage,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                opened_at=trade.opened_at,
                closed_at=trade.closed_at,
                duration_sec=trade.duration_sec,
                gross_pnl=trade.gross_pnl,
                fees=trade.fees,
                funding=trade.funding,
                slippage_cost=trade.slippage_cost,
                net_pnl=trade.net_pnl,
                r_multiple=trade.r_multiple,
                market_regime=trade.regime.value,
                signal_score=trade.opportunity_score,
                consensus_score=trade.consensus_score,
                expected_net_edge=trade.expected_net_edge,
                realised_edge=realised_edge,
                entry_notional=trade.entry_notional,
                initial_risk=trade.initial_risk,
                exit_reason=trade.exit_reason.value,
                reason_codes=list(trade.reason_codes),
                extra=_jsonable(trade.metadata),
                mode=self.mode.value,
            )
        )

    def record_decision(
        self,
        symbol: str,
        accepted: bool,
        stage: str,
        detail: str,
        timestamp: int,
        rejection_reason: str | None = None,
        context: dict[str, Any] | None = None,
        trade_id: str | None = None,
    ) -> None:
        """The audit row. Written for rejections as well as acceptances."""
        ctx = context or {}
        self._enqueue(
            DecisionRecord(
                symbol=symbol,
                timestamp=timestamp,
                accepted=accepted,
                stage=stage,
                rejection_reason=rejection_reason,
                detail=detail[:2000],
                market_regime=str(ctx.get("regime", "")),
                direction=str(ctx.get("direction", "WAIT")),
                strategies=list(ctx.get("agreeing_strategies", [])),
                consensus_score=float(ctx.get("consensus_score", 0.0) or 0.0),
                opportunity_score=float(ctx.get("opportunity_score", 0.0) or 0.0),
                expected_net_edge=float(ctx.get("expected_net_edge", 0.0) or 0.0),
                win_probability=float(ctx.get("win_probability", 0.0) or 0.0),
                entry_price=float(ctx.get("entry", 0.0) or 0.0),
                stop_loss=float(ctx.get("stop_loss", 0.0) or 0.0),
                take_profit=float(ctx.get("take_profit", 0.0) or 0.0),
                quantity=float(ctx.get("quantity", 0.0) or 0.0),
                leverage=int(ctx.get("leverage", 0) or 0),
                risk_amount=float(ctx.get("risk_amount", 0.0) or 0.0),
                context=_jsonable(ctx),
                trade_id=trade_id,
            )
        )

    def record_signal(self, signal: Signal, regime: str = "") -> None:
        self._enqueue(
            SignalRecord(
                symbol=signal.symbol,
                strategy=signal.strategy,
                direction=signal.direction.value,
                confidence=signal.confidence,
                timeframe=signal.timeframe,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                volatility=signal.volatility,
                risk_score=signal.risk_score,
                reason_codes=list(signal.reason_codes),
                market_regime=regime,
                timestamp=signal.signal_timestamp,
            )
        )

    def record_order(self, order: Order) -> None:
        self._enqueue(
            OrderRecord(
                client_order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                status=order.status.value,
                filled_quantity=order.filled_quantity,
                average_price=order.average_price,
                commission=order.total_commission,
                reduce_only=order.reduce_only,
                intent_id=order.intent_id,
                error=order.error,
                created_at=order.created_at,
                updated_at=order.updated_at,
            )
        )

    def record_position(self, position: Position, is_open: bool = True) -> None:
        self._enqueue(
            PositionRecord(
                position_id=position.position_id,
                symbol=position.symbol,
                direction=position.direction.value,
                quantity=position.quantity,
                entry_price=position.entry_price,
                leverage=position.leverage,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                strategy=position.strategy,
                market_regime=position.regime.value,
                opened_at=position.opened_at,
                entry_order_id=position.entry_order_id,
                stop_order_id=position.stop_order_id,
                take_profit_order_id=position.take_profit_order_id,
                adopted=position.adopted,
                is_open=is_open,
                extra=_jsonable(position.metadata),
            )
        )

    def record_risk_event(self, event: RiskEvent) -> None:
        self._enqueue(
            RiskEventRecord(
                event_type=event.event_type.value,
                severity=event.severity,
                message=event.message[:2000],
                symbol=event.symbol,
                timestamp=event.timestamp,
                data=_jsonable(event.data),
            )
        )

    def record_system_event(
        self,
        event_type: str,
        message: str,
        timestamp: int,
        severity: str = "INFO",
        data: dict[str, Any] | None = None,
    ) -> None:
        self._enqueue(
            SystemEventRecord(
                event_type=event_type,
                message=message[:2000],
                severity=severity,
                timestamp=timestamp,
                data=_jsonable(data or {}),
            )
        )

    def record_equity(self, timestamp: int, equity: float, balance: float, **fields: float) -> None:
        self._enqueue(
            EquityRecord(
                timestamp=timestamp,
                equity=equity,
                balance=balance,
                unrealized_pnl=fields.get("unrealized_pnl", 0.0),
                realized_pnl_today=fields.get("realized_pnl_today", 0.0),
                open_positions=int(fields.get("open_positions", 0)),
                total_exposure=fields.get("total_exposure", 0.0),
                open_risk=fields.get("open_risk", 0.0),
                margin_used=fields.get("margin_used", 0.0),
                drawdown=fields.get("drawdown", 0.0),
            )
        )

    def record_scan(self, candidates: tuple[Candidate, ...], timestamp: int) -> None:
        for candidate in candidates:
            score = candidate.market_score
            self._enqueue(
                MarketSnapshotRecord(
                    timestamp=timestamp,
                    symbol=candidate.symbol,
                    rank=candidate.rank,
                    market_score=score.total,
                    market_regime=candidate.regime.value,
                    volatility=score.volatility,
                    liquidity_usd=score.liquidity_usd,
                    spread_bps=score.spread_bps,
                    funding_rate=score.funding_rate,
                    components=_jsonable(score.components),
                )
            )

    def record_strategy_metrics(
        self, strategy: str, timestamp: int, metrics: dict[str, Any]
    ) -> None:
        self._enqueue(
            StrategyMetricRecord(
                strategy=strategy,
                timestamp=timestamp,
                total_trades=int(metrics.get("trades", 0)),
                winning_trades=int(metrics.get("wins", 0)),
                losing_trades=int(metrics.get("trades", 0)) - int(metrics.get("wins", 0)),
                win_rate=float(metrics.get("win_rate", 0.0)),
                profit_factor=float(metrics.get("profit_factor") or 0.0),
                expectancy=float(metrics.get("expectancy", 0.0)),
                expectancy_r=float(metrics.get("expectancy_r", 0.0)),
                max_drawdown=float(metrics.get("max_drawdown_r", 0.0)),
                total_fees=float(metrics.get("fees", 0.0)),
                net_pnl=float(metrics.get("net_pnl", 0.0)),
                allocation_weight=float(metrics.get("allocation_weight", 1.0)),
                is_suspended=bool(metrics.get("suspended", False)),
            )
        )

    # ------------------------------------------------------------------ #
    # Reads — for the dashboard and analysis only
    # ------------------------------------------------------------------ #
    async def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        from sqlalchemy import select

        if self._session_factory is None:
            return []
        try:
            async with self._session_factory() as session:
                rows = await session.execute(
                    select(TradeRecord).order_by(TradeRecord.closed_at.desc()).limit(limit)
                )
                return [_row_to_dict(r) for r in rows.scalars().all()]
        except Exception as exc:  # noqa: BLE001
            log.warning("database_read_failed", error=str(exc)[:200])
            return []

    async def recent_decisions(
        self, limit: int = 100, accepted: bool | None = None
    ) -> list[dict[str, Any]]:
        from sqlalchemy import select

        if self._session_factory is None:
            return []
        try:
            async with self._session_factory() as session:
                query = (
                    select(DecisionRecord).order_by(DecisionRecord.timestamp.desc()).limit(limit)
                )
                if accepted is not None:
                    query = query.where(DecisionRecord.accepted == accepted)
                rows = await session.execute(query)
                return [_row_to_dict(r) for r in rows.scalars().all()]
        except Exception as exc:  # noqa: BLE001
            log.warning("database_read_failed", error=str(exc)[:200])
            return []

    async def equity_history(self, limit: int = 500) -> list[dict[str, Any]]:
        from sqlalchemy import select

        if self._session_factory is None:
            return []
        try:
            async with self._session_factory() as session:
                rows = await session.execute(
                    select(EquityRecord).order_by(EquityRecord.timestamp.desc()).limit(limit)
                )
                return [_row_to_dict(r) for r in reversed(rows.scalars().all())]
        except Exception as exc:  # noqa: BLE001
            log.warning("database_read_failed", error=str(exc)[:200])
            return []

    async def open_positions(self) -> list[dict[str, Any]]:
        from sqlalchemy import select

        if self._session_factory is None:
            return []
        try:
            async with self._session_factory() as session:
                rows = await session.execute(
                    select(PositionRecord).where(PositionRecord.is_open.is_(True))
                )
                return [_row_to_dict(r) for r in rows.scalars().all()]
        except Exception as exc:  # noqa: BLE001
            log.warning("database_read_failed", error=str(exc)[:200])
            return []

    # ------------------------------------------------------------------ #
    async def prune(self, retention: dict[str, int], now_ms: int) -> dict[str, int]:
        """Delete rows older than their retention window. 0 days = keep forever."""
        from sqlalchemy import delete

        if self._session_factory is None:
            return {}

        removed: dict[str, int] = {}
        try:
            async with self._session_factory() as session:
                for model, key in PRUNABLE.items():
                    days = int(retention.get(key, 0) or 0)
                    if days <= 0:
                        continue
                    cutoff = now_ms - days * 86_400_000
                    result = await session.execute(delete(model).where(model.timestamp < cutoff))
                    removed[model.__tablename__] = int(result.rowcount or 0)
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("database_prune_failed", error=str(exc)[:200])
            return {}

        if any(removed.values()):
            log.info("database_pruned", **removed)
        return removed

    def stats(self) -> dict[str, Any]:
        return {
            "available": self.health.available,
            "writes": self.health.writes,
            "failures": self.health.failures,
            "dropped": self.health.dropped,
            "buffered": len(self._buffer),
            "last_error": self.health.last_error,
        }


# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    """Coerce values into something the JSON column can store.

    Enums, tuples and sets appear throughout the domain types; letting one reach
    the driver raises mid-flush and loses the whole batch.
    """
    from enum import Enum

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {column.name: _jsonable(getattr(row, column.name)) for column in row.__table__.columns}


def _redact_url(url: str) -> str:
    """A Postgres URL can contain a password. Never log it."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _credentials, _, host = rest.partition("@")
    return f"{scheme}://<redacted>@{host}"
