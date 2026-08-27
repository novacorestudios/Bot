"""Database schema.

SQLite by default — a single file is entirely adequate for one bot on one VPS,
and it survives a crash without a server process to also recover. Postgres works
by changing `DATABASE_URL`.

The schema exists to answer one question months later: **"why did the bot do
that?"** So every decision is stored, not just every trade — including the
decisions that ended in a rejection, which are the majority and are usually the
more informative ones.

Retention is bounded: `market_snapshots` and `signals` accumulate fast enough to
fill a small VPS disk in weeks, so both are pruned by age. `trades` and
`risk_events` are kept indefinitely — they are the record that matters.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _now() -> datetime:
    return datetime.now(tz=UTC)


class Base(DeclarativeBase):
    """Declarative base with a JSON type that works on SQLite and Postgres."""

    type_annotation_map = {dict: JSON, list: JSON}


class TradeRecord(Base):
    """One completed round trip. The permanent record of what happened."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    strategy: Mapped[str] = mapped_column(String(48), index=True)
    direction: Mapped[str] = mapped_column(String(8))

    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    leverage: Mapped[int] = mapped_column(Integer)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)

    opened_at: Mapped[int] = mapped_column(Integer, index=True)
    closed_at: Mapped[int] = mapped_column(Integer, index=True)
    duration_sec: Mapped[float] = mapped_column(Float)

    # Costs are stored separately from PnL on purpose: "net_pnl was positive" and
    # "the strategy had an edge" are different claims, and only the breakdown
    # distinguishes them.
    gross_pnl: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    funding: Mapped[float] = mapped_column(Float)
    slippage_cost: Mapped[float] = mapped_column(Float)
    net_pnl: Mapped[float] = mapped_column(Float, index=True)
    r_multiple: Mapped[float] = mapped_column(Float, default=0.0)

    market_regime: Mapped[str] = mapped_column(String(24), index=True)
    signal_score: Mapped[float] = mapped_column(Float, default=0.0)
    consensus_score: Mapped[float] = mapped_column(Float, default=0.0)
    expected_net_edge: Mapped[float] = mapped_column(Float, default=0.0)
    realised_edge: Mapped[float] = mapped_column(Float, default=0.0)
    entry_notional: Mapped[float] = mapped_column(Float, default=0.0)
    initial_risk: Mapped[float] = mapped_column(Float, default=0.0)

    exit_reason: Mapped[str] = mapped_column(String(32), index=True)
    reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    mode: Mapped[str] = mapped_column(String(12), index=True, default="PAPER")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_trades_strategy_closed", "strategy", "closed_at"),
        Index("ix_trades_symbol_closed", "symbol", "closed_at"),
    )


class PositionRecord(Base):
    """Live position state, so a crash does not lose what we believe we hold.

    The exchange remains the source of truth; this exists so that after a
    restart the reconciler can tell an ADOPTED position (unknown thesis, close
    it) from one this bot deliberately opened (known thesis, manage it).
    """

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    position_id: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    leverage: Mapped[int] = mapped_column(Integer)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    strategy: Mapped[str] = mapped_column(String(48))
    market_regime: Mapped[str] = mapped_column(String(24))
    opened_at: Mapped[int] = mapped_column(Integer)
    entry_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stop_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    take_profit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    adopted: Mapped[bool] = mapped_column(Boolean, default=False)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class OrderRecord(Base):
    """Every order submitted, with its status transitions."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    side: Mapped[str] = mapped_column(String(8))
    order_type: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    filled_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    average_price: Mapped[float] = mapped_column(Float, default=0.0)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    reduce_only: Mapped[bool] = mapped_column(Boolean, default=False)
    intent_id: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[int] = mapped_column(Integer)

    fills: Mapped[list[FillRecord]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class FillRecord(Base):
    """An individual execution, with the commission actually charged."""

    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True)
    fill_id: Mapped[str] = mapped_column(String(48), unique=True)
    order_pk: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    commission_asset: Mapped[str] = mapped_column(String(12), default="USDT")
    is_maker: Mapped[bool] = mapped_column(Boolean, default=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[int] = mapped_column(Integer, index=True)

    order: Mapped[OrderRecord] = relationship(back_populates="fills")


class DecisionRecord(Base):
    """The audit log.

    One row per evaluated opportunity, **accepted or rejected**. This is what
    makes "why did the bot enter?" and "why didn't it?" answerable months later,
    which the brief requires and which is impossible to reconstruct after the
    fact from prices alone.
    """

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    timestamp: Mapped[int] = mapped_column(Integer, index=True)
    accepted: Mapped[bool] = mapped_column(Boolean, index=True)
    stage: Mapped[str] = mapped_column(String(32), index=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    detail: Mapped[str] = mapped_column(Text, default="")

    market_regime: Mapped[str] = mapped_column(String(24), default="")
    direction: Mapped[str] = mapped_column(String(8), default="WAIT")
    strategies: Mapped[list] = mapped_column(JSON, default=list)
    consensus_score: Mapped[float] = mapped_column(Float, default=0.0)
    opportunity_score: Mapped[float] = mapped_column(Float, default=0.0)
    expected_net_edge: Mapped[float] = mapped_column(Float, default=0.0)
    win_probability: Mapped[float] = mapped_column(Float, default=0.0)

    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss: Mapped[float] = mapped_column(Float, default=0.0)
    take_profit: Mapped[float] = mapped_column(Float, default=0.0)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    leverage: Mapped[int] = mapped_column(Integer, default=0)
    risk_amount: Mapped[float] = mapped_column(Float, default=0.0)

    # The full context: indicators, costs, correlation, portfolio state.
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    trade_id: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)

    __table_args__ = (
        Index("ix_decisions_symbol_time", "symbol", "timestamp"),
        Index("ix_decisions_reason_time", "rejection_reason", "timestamp"),
    )


class SignalRecord(Base):
    """Individual strategy signals, including WAIT.

    'Considered and declined' is different from 'never ran', and only storing
    both makes strategy behaviour reconstructable.
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    strategy: Mapped[str] = mapped_column(String(48), index=True)
    direction: Mapped[str] = mapped_column(String(8), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    timeframe: Mapped[str] = mapped_column(String(8))
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss: Mapped[float] = mapped_column(Float, default=0.0)
    take_profit: Mapped[float] = mapped_column(Float, default=0.0)
    volatility: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=50.0)
    reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    market_regime: Mapped[str] = mapped_column(String(24), default="")
    timestamp: Mapped[int] = mapped_column(Integer, index=True)


class MarketSnapshotRecord(Base):
    """A compact per-scan record of the ranked candidates.

    Deliberately compact: storing full order books or every indicator would fill
    a small VPS disk within weeks for no analytical benefit.
    """

    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[int] = mapped_column(Integer, index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    market_score: Mapped[float] = mapped_column(Float)
    market_regime: Mapped[str] = mapped_column(String(24))
    volatility: Mapped[float] = mapped_column(Float)
    liquidity_usd: Mapped[float] = mapped_column(Float)
    spread_bps: Mapped[float] = mapped_column(Float)
    funding_rate: Mapped[float] = mapped_column(Float)
    components: Mapped[dict] = mapped_column(JSON, default=dict)


class StrategyMetricRecord(Base):
    """Rolling per-strategy performance, sampled periodically."""

    __tablename__ = "strategy_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy: Mapped[str] = mapped_column(String(48), index=True)
    timestamp: Mapped[int] = mapped_column(Integer, index=True)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    average_win: Mapped[float] = mapped_column(Float, default=0.0)
    average_loss: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    expectancy: Mapped[float] = mapped_column(Float, default=0.0)
    expectancy_r: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    sortino: Mapped[float] = mapped_column(Float, default=0.0)
    average_duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    total_fees: Mapped[float] = mapped_column(Float, default=0.0)
    total_slippage: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    allocation_weight: Mapped[float] = mapped_column(Float, default=1.0)
    is_suspended: Mapped[bool] = mapped_column(Boolean, default=False)


class RiskEventRecord(Base):
    """Every risk event. Kept indefinitely — this is the incident log."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(12), index=True)
    message: Mapped[str] = mapped_column(Text)
    symbol: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)
    timestamp: Mapped[int] = mapped_column(Integer, index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


class EquityRecord(Base):
    """Periodic equity samples, for drawdown and risk-adjusted metrics."""

    __tablename__ = "equity_curve"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[int] = mapped_column(Integer, index=True, unique=True)
    equity: Mapped[float] = mapped_column(Float)
    balance: Mapped[float] = mapped_column(Float)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl_today: Mapped[float] = mapped_column(Float, default=0.0)
    open_positions: Mapped[int] = mapped_column(Integer, default=0)
    total_exposure: Mapped[float] = mapped_column(Float, default=0.0)
    open_risk: Mapped[float] = mapped_column(Float, default=0.0)
    margin_used: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown: Mapped[float] = mapped_column(Float, default=0.0)


class SystemEventRecord(Base):
    """Connectivity, safe mode, restarts — the operational history."""

    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(12), default="INFO")
    timestamp: Mapped[int] = mapped_column(Integer, index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)


#: Tables pruned by age, with the config key that controls each.
PRUNABLE: dict[type[Base], str] = {
    MarketSnapshotRecord: "market_snapshot_retention_days",
    SignalRecord: "signal_retention_days",
    DecisionRecord: "decision_retention_days",
}
