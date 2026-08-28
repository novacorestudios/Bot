"""Core domain types shared by every layer.

These are deliberately plain dataclasses/enums with no I/O and no dependency on
configuration, so they can be constructed freely in tests.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class TradingMode(StrEnum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class Direction(StrEnum):
    """Signal direction. WAIT means the strategy has no opinion."""

    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"

    @property
    def sign(self) -> int:
        """+1 for LONG, -1 for SHORT, 0 for WAIT."""
        return {Direction.LONG: 1, Direction.SHORT: -1, Direction.WAIT: 0}[self]

    def opposite(self) -> Direction:
        if self is Direction.LONG:
            return Direction.SHORT
        if self is Direction.SHORT:
            return Direction.LONG
        return Direction.WAIT


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @classmethod
    def for_entry(cls, direction: Direction) -> OrderSide:
        if direction is Direction.LONG:
            return cls.BUY
        if direction is Direction.SHORT:
            return cls.SELL
        raise ValueError("cannot derive an order side from WAIT")

    @classmethod
    def for_exit(cls, direction: Direction) -> OrderSide:
        return cls.for_entry(direction.opposite())


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    STOP = "STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTX = "GTX"  # post-only


class OrderStatus(StrEnum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"  # submitted locally, no exchange ack yet
    UNKNOWN = "UNKNOWN"  # submission result indeterminate (timeout)

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    @property
    def is_open(self) -> bool:
        return self in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING}


class MarketRegime(StrEnum):
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    PANIC = "PANIC"

    @property
    def blocks_entries(self) -> bool:
        return self is MarketRegime.PANIC


class ExitReason(StrEnum):
    TAKE_PROFIT = "TAKE_PROFIT"
    PARTIAL_TAKE_PROFIT = "PARTIAL_TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_LIMIT = "TIME_LIMIT"
    SIGNAL_FLIP = "SIGNAL_FLIP"
    REGIME_CHANGE = "REGIME_CHANGE"
    NEGATIVE_EDGE = "NEGATIVE_EDGE"
    RISK_EVENT = "RISK_EVENT"
    KILL_SWITCH = "KILL_SWITCH"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"
    ADOPTED_POSITION = "ADOPTED_POSITION"
    LIQUIDATION = "LIQUIDATION"


class RejectionReason(StrEnum):
    """Why an opportunity did not become a trade. Every rejection is audited."""

    NO_SIGNAL = "NO_SIGNAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    INSUFFICIENT_CONSENSUS = "INSUFFICIENT_CONSENSUS"
    LOW_OPPORTUNITY_SCORE = "LOW_OPPORTUNITY_SCORE"
    NEGATIVE_EXPECTED_EDGE = "NEGATIVE_EXPECTED_EDGE"
    REGIME_BLOCKED = "REGIME_BLOCKED"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    MAX_POSITIONS = "MAX_POSITIONS"
    ALREADY_IN_POSITION = "ALREADY_IN_POSITION"
    INTENT_IN_FLIGHT = "INTENT_IN_FLIGHT"
    RISK_BUDGET_EXCEEDED = "RISK_BUDGET_EXCEEDED"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    DIRECTION_EXPOSURE_LIMIT = "DIRECTION_EXPOSURE_LIMIT"
    SYMBOL_EXPOSURE_LIMIT = "SYMBOL_EXPOSURE_LIMIT"
    MARGIN_LIMIT = "MARGIN_LIMIT"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    SIZE_BELOW_MINIMUM = "SIZE_BELOW_MINIMUM"
    NOTIONAL_BELOW_MINIMUM = "NOTIONAL_BELOW_MINIMUM"
    NOTIONAL_ABOVE_MAXIMUM = "NOTIONAL_ABOVE_MAXIMUM"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    LIQUIDATION_TOO_CLOSE = "LIQUIDATION_TOO_CLOSE"
    INVALID_STOP = "INVALID_STOP"
    INVALID_TARGET = "INVALID_TARGET"
    STALE_DATA = "STALE_DATA"
    ENTRIES_BLOCKED = "ENTRIES_BLOCKED"
    RECONCILING = "RECONCILING"
    EXCHANGE_FILTER = "EXCHANGE_FILTER"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    SLIPPAGE_TOO_HIGH = "SLIPPAGE_TOO_HIGH"


class RiskEventType(StrEnum):
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
    KILL_SWITCH_CLEARED = "KILL_SWITCH_CLEARED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    HOURLY_LOSS_LIMIT = "HOURLY_LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    API_ERROR_BURST = "API_ERROR_BURST"
    ORDER_REJECTED = "ORDER_REJECTED"
    EXCESSIVE_SLIPPAGE = "EXCESSIVE_SLIPPAGE"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    UNEXPECTED_POSITION = "UNEXPECTED_POSITION"
    ORPHAN_ORDER = "ORPHAN_ORDER"
    ABNORMAL_MARKET = "ABNORMAL_MARKET"
    CONNECTION_LOST = "CONNECTION_LOST"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"
    SAFE_MODE_ENTERED = "SAFE_MODE_ENTERED"
    SAFE_MODE_EXITED = "SAFE_MODE_EXITED"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    STRATEGY_REENABLED = "STRATEGY_REENABLED"
    MISSING_STOP_LOSS = "MISSING_STOP_LOSS"
    POSITION_ADOPTED = "POSITION_ADOPTED"


class Timeframe(StrEnum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }[self.value]

    @property
    def milliseconds(self) -> int:
        return self.seconds * 1000


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV bar. `open_time`/`close_time` are epoch milliseconds."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_volume: float = 0.0
    closed: bool = True

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_fraction(self) -> float:
        r = self.range
        return self.body / r if r > 0 else 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True, slots=True)
class BookTicker:
    """Best bid/ask snapshot."""

    symbol: str
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    timestamp: int

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.spread / m) * 10_000.0 if m > 0 else float("inf")

    @property
    def imbalance(self) -> float:
        """(bid - ask) / (bid + ask) in [-1, 1]. Positive = buy pressure."""
        total = self.bid_qty + self.ask_qty
        return (self.bid_qty - self.ask_qty) / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class MarkPriceInfo:
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time: int
    timestamp: int


@dataclass(frozen=True, slots=True)
class DepthLevel:
    price: float
    qty: float


@dataclass(frozen=True, slots=True)
class OrderBook:
    symbol: str
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    timestamp: int

    def notional_within(self, fraction: float) -> tuple[float, float]:
        """Quote-notional resting within `fraction` of the mid, (bid, ask)."""
        if not self.bids or not self.asks:
            return 0.0, 0.0
        mid = (self.bids[0].price + self.asks[0].price) / 2.0
        lo, hi = mid * (1 - fraction), mid * (1 + fraction)
        bid_n = sum(lvl.price * lvl.qty for lvl in self.bids if lvl.price >= lo)
        ask_n = sum(lvl.price * lvl.qty for lvl in self.asks if lvl.price <= hi)
        return bid_n, ask_n

    @property
    def imbalance(self) -> float:
        bid_n, ask_n = self.notional_within(0.002)
        total = bid_n + ask_n
        return (bid_n - ask_n) / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Ticker24h:
    symbol: str
    last_price: float
    price_change_pct: float
    high: float
    low: float
    volume: float
    quote_volume: float
    trades: int
    timestamp: int


# --------------------------------------------------------------------------- #
# Symbol metadata / exchange filters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LeverageBracket:
    bracket: int
    initial_leverage: int
    notional_cap: float
    notional_floor: float
    maint_margin_ratio: float
    cum: float


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    """Everything needed to construct a valid order for one symbol."""

    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    contract_type: str
    price_precision: int
    quantity_precision: int
    tick_size: float
    step_size: float
    min_qty: float
    max_qty: float
    min_notional: float
    market_min_qty: float = 0.0
    market_max_qty: float = 0.0
    multiplier_up: float = 5.0
    multiplier_down: float = 0.2
    max_leverage: int = 20
    brackets: tuple[LeverageBracket, ...] = ()
    onboard_date: int = 0

    @property
    def is_tradable(self) -> bool:
        return self.status == "TRADING" and self.contract_type == "PERPETUAL"

    def bracket_for_notional(self, notional: float) -> LeverageBracket | None:
        for b in self.brackets:
            if b.notional_floor <= notional <= b.notional_cap:
                return b
        return self.brackets[-1] if self.brackets else None


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Signal:
    """Structured output of a single strategy. Strategies return data only."""

    symbol: str
    strategy: str
    direction: Direction
    confidence: float  # 0..100
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    signal_timestamp: int
    expected_duration_sec: int = 0
    volatility: float = 0.0  # ATR / price
    risk_score: float = 50.0  # 0..100, higher = riskier
    reason_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction is not Direction.WAIT

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def target_distance(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def risk_reward(self) -> float:
        d = self.stop_distance
        return self.target_distance / d if d > 0 else 0.0

    def validate(self) -> list[str]:
        """Return a list of structural problems; empty means well-formed."""
        problems: list[str] = []
        if not 0.0 <= self.confidence <= 100.0:
            problems.append("confidence out of range")
        if self.direction is Direction.WAIT:
            return problems
        if self.entry_price <= 0:
            problems.append("entry_price must be positive")
        if self.stop_loss <= 0:
            problems.append("stop_loss must be positive")
        if self.take_profit <= 0:
            problems.append("take_profit must be positive")
        if self.direction is Direction.LONG:
            if self.stop_loss >= self.entry_price:
                problems.append("LONG stop must sit below entry")
            if self.take_profit <= self.entry_price:
                problems.append("LONG target must sit above entry")
        if self.direction is Direction.SHORT:
            if self.stop_loss <= self.entry_price:
                problems.append("SHORT stop must sit above entry")
            if self.take_profit >= self.entry_price:
                problems.append("SHORT target must sit below entry")
        return problems


@dataclass(frozen=True, slots=True)
class AggregatedSignal:
    """Consensus across strategies for one symbol."""

    symbol: str
    direction: Direction
    consensus_score: float  # 0..100
    confidence: float  # 0..100, weighted
    entry_price: float
    stop_loss: float
    take_profit: float
    contributing: tuple[Signal, ...]
    opposing: tuple[Signal, ...]
    conflict_ratio: float
    regime: MarketRegime
    timestamp: int
    reason_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def risk_reward(self) -> float:
        d = self.stop_distance
        return abs(self.take_profit - self.entry_price) / d if d > 0 else 0.0

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(s.strategy for s in self.contributing)

    # ------------------------------------------------------------------ #
    # Attribution. ONE definition, used by everything.
    # ------------------------------------------------------------------ #
    @property
    def primary_strategy(self) -> str:
        """The strategy this signal — and any trade from it — belongs to.

        Highest confidence wins; ties break on the strategy name so the answer
        cannot depend on tuple ordering. That determinism is the point: the edge
        calculator previously took ``contributing[0]`` while the opportunity
        took the highest-confidence one, so the *same trade* could be attributed
        to two different strategies. Edge statistics, risk allocation and trade
        ownership then referred to different things, and every per-strategy
        report was quietly wrong.
        """
        if not self.contributing:
            return "unknown"
        return min(self.contributing, key=lambda s: (-s.confidence, s.strategy)).strategy

    @property
    def contributing_strategies(self) -> tuple[str, ...]:
        """Every strategy that agreed, primary first."""
        ordered = sorted(self.contributing, key=lambda s: (-s.confidence, s.strategy))
        return tuple(s.strategy for s in ordered)

    @property
    def contribution_weights(self) -> dict[str, float]:
        """Each contributor's share of the total agreeing confidence.

        Lets a report separate "this strategy owned the trade" from "this
        strategy supported it", which a single name cannot express.
        """
        total = sum(s.confidence for s in self.contributing)
        if total <= 0:
            return {}
        return {
            s.strategy: round(s.confidence / total, 6)
            for s in sorted(self.contributing, key=lambda s: (-s.confidence, s.strategy))
        }


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """All costs of a round trip, expressed as fractions of notional."""

    entry_fee: float
    exit_fee: float
    spread_cost: float
    slippage: float
    funding: float

    @property
    def total(self) -> float:
        return self.entry_fee + self.exit_fee + self.spread_cost + self.slippage + self.funding

    def as_dict(self) -> dict[str, float]:
        return {
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "spread_cost": self.spread_cost,
            "slippage": self.slippage,
            "funding": self.funding,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class EdgeEstimate:
    """Expected value of a candidate trade, per unit of notional."""

    win_probability: float
    gross_win: float  # fraction of notional if target is reached
    gross_loss: float  # positive fraction of notional if stop is hit
    costs: CostEstimate
    expected_gross: float
    expected_net: float

    @property
    def is_positive(self) -> bool:
        return self.expected_net > 0.0


@dataclass(frozen=True, slots=True)
class OpportunityScore:
    total: float  # 0..100
    components: dict[str, float]
    penalties: dict[str, float]

    @property
    def grade(self) -> str:
        if self.total >= 90:
            return "EXCEPTIONAL"
        if self.total >= 80:
            return "STRONG"
        if self.total >= 70:
            return "MODERATE"
        return "REJECT"


@dataclass(frozen=True, slots=True)
class MarketScore:
    """Scanner output for one symbol."""

    symbol: str
    total: float
    components: dict[str, float]
    penalties: dict[str, float]
    volatility: float
    liquidity_usd: float
    spread_bps: float
    funding_rate: float
    timestamp: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total": self.total,
            "components": dict(self.components),
            "penalties": dict(self.penalties),
            "volatility": self.volatility,
            "liquidity_usd": self.liquidity_usd,
            "spread_bps": self.spread_bps,
            "funding_rate": self.funding_rate,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """A ranked market plus, when available, its best current opportunity."""

    rank: int
    market_score: MarketScore
    regime: MarketRegime
    best_strategy: str | None = None
    direction: Direction = Direction.WAIT
    confidence: float = 0.0
    expected_net_edge: float | None = None
    opportunity_score: float | None = None
    risk_level: str = "UNKNOWN"

    @property
    def symbol(self) -> str:
        return self.market_score.symbol


# --------------------------------------------------------------------------- #
# Orders, fills, positions, trades
# --------------------------------------------------------------------------- #
def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}{raw}" if prefix else raw


@dataclass(slots=True)
class OrderIntent:
    """The risk engine's approved instruction to the execution engine.

    Only the risk engine constructs this. Its `intent_id` seeds the deterministic
    client order id, which is what makes retries idempotent.
    """

    intent_id: str
    symbol: str
    direction: Direction
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None
    stop_loss: float
    take_profit: float
    leverage: int
    notional: float
    risk_amount: float
    strategy: str
    regime: MarketRegime
    opportunity_score: float
    expected_net_edge: float
    reduce_only: bool = False
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def client_order_id(self) -> str:
        return f"tb_{self.intent_id}"


@dataclass(slots=True)
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    commission_asset: str
    timestamp: int
    is_maker: bool = False
    realized_pnl: float = 0.0


@dataclass(slots=True)
class Order:
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: str | None = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    reduce_only: bool = False
    close_position: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    fills: list[Fill] = field(default_factory=list)
    intent_id: str | None = None
    error: str | None = None

    @property
    def remaining(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)


@dataclass(slots=True)
class Position:
    """An open position. `quantity` is always positive; `direction` carries sign."""

    position_id: str
    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    leverage: int
    stop_loss: float
    take_profit: float
    strategy: str
    regime: MarketRegime
    #: When the position was FILLED — not when the signal fired. Duration, the
    #: maximum-hold cap and funding eligibility all measure from this. The two
    #: fields below record the rest of the timeline; in live trading they are
    #: set by the execution engine, in the backtest by the fill simulator.
    opened_at: int
    #: When the decision was made. Every bar behind it had already closed.
    signal_at: int = 0
    #: When the order was submitted. Between signal_at and opened_at.
    order_at: int = 0
    entry_notional: float = 0.0
    entry_fee: float = 0.0
    funding_paid: float = 0.0
    entry_slippage: float = 0.0
    initial_stop: float = 0.0
    initial_risk: float = 0.0
    trailing_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    partial_taken: bool = False
    stop_order_id: str | None = None
    take_profit_order_id: str | None = None
    entry_order_id: str | None = None
    opportunity_score: float = 0.0
    expected_net_edge: float = 0.0
    adopted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity * self.direction.sign

    def unrealized_pnl_pct(self, price: float) -> float:
        if self.entry_notional <= 0:
            return 0.0
        return self.unrealized_pnl(price) / self.entry_notional

    def r_multiple(self, price: float) -> float:
        """Profit measured in units of the position's initial risk."""
        if self.initial_risk <= 0:
            return 0.0
        return self.unrealized_pnl(price) / self.initial_risk

    def notional(self, price: float) -> float:
        return price * self.quantity

    def margin(self, price: float) -> float:
        return self.notional(price) / max(1, self.leverage)

    def duration_sec(self, now_ms: int) -> float:
        return max(0.0, (now_ms - self.opened_at) / 1000.0)

    def update_extremes(self, price: float) -> None:
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)

    def is_stop_hit(self, low: float, high: float) -> bool:
        if self.direction is Direction.LONG:
            return low <= self.stop_loss
        return high >= self.stop_loss

    def is_target_hit(self, low: float, high: float) -> bool:
        if self.direction is Direction.LONG:
            return high >= self.take_profit
        return low <= self.take_profit


@dataclass(slots=True)
class Trade:
    """A completed round trip. This is what performance analysis consumes."""

    trade_id: str
    symbol: str
    strategy: str
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    stop_loss: float
    take_profit: float
    #: The FILL timestamp, matching Position.opened_at. `duration_sec` is
    #: measured from here, so it is time exposed to the market — not time
    #: since a signal that had not yet been acted on.
    opened_at: int
    closed_at: int
    #: PnL as actually filled: (exit_fill - entry_fill) x qty x sign. Execution
    #: costs are already inside these prices, so subtracting them again from
    #: this number double-counts. Use `reference_gross_pnl` for that.
    gross_pnl: float
    fees: float
    funding: float
    slippage_cost: float
    net_pnl: float

    exit_reason: ExitReason
    regime: MarketRegime

    # -- the timing ledger (V3.2) --------------------------------------- #
    #: When the decision was made. `opened_at - signal_at` is the delay between
    #: deciding and being in the market — zero would mean the fill was
    #: instantaneous, which no real venue offers.
    signal_at: int = 0
    #: When the order was submitted.
    order_at: int = 0

    # -- the cost ledger (V3.1) ----------------------------------------- #
    # One coherent accounting, so a report cannot double-count. The identity
    # `cost_identity_error()` checks is:
    #
    #   reference_gross_pnl - execution_costs - fees - funding == net_pnl
    #
    # where execution_costs = spread + slippage + latency, both legs.
    #: PnL had both legs filled at their reference prices — no spread, no
    #: slippage, no latency. The pure price move the strategy was right about.
    reference_gross_pnl: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    spread_cost: float = 0.0
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    latency_cost: float = 0.0
    opportunity_score: float = 0.0
    expected_net_edge: float = 0.0
    consensus_score: float = 0.0
    entry_notional: float = 0.0
    initial_risk: float = 0.0
    reason_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_costs(self) -> float:
        """Everything paid to the exchange's microstructure, both legs."""
        return self.spread_cost + self.entry_slippage + self.exit_slippage + self.latency_cost

    @property
    def total_cost(self) -> float:
        """Every cost of the round trip."""
        return self.execution_costs + self.fees + self.funding

    def cost_identity_error(self) -> float:
        """How far the ledger is from balancing. Should be ~0.

        A non-zero value means some cost is counted twice or not at all, and
        every derived figure — expectancy, cost ratio, edge calibration — is
        wrong by that amount.
        """
        if self.reference_gross_pnl == 0.0 and self.execution_costs == 0.0:
            return 0.0  # ledger not populated (legacy trade)
        expected = self.reference_gross_pnl - self.total_cost
        return expected - self.net_pnl

    @property
    def duration_sec(self) -> float:
        """Time held, measured from the FILL. See :attr:`opened_at`."""
        return max(0.0, (self.closed_at - self.opened_at) / 1000.0)

    @property
    def signal_to_fill_sec(self) -> float:
        """How long the trade waited between decision and fill.

        Zero when the timing was never recorded (a trade built before V3.2, or
        by a caller that does not set `signal_at`) — not zero latency.
        """
        if not self.signal_at:
            return 0.0
        return max(0.0, (self.opened_at - self.signal_at) / 1000.0)

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def r_multiple(self) -> float:
        return self.net_pnl / self.initial_risk if self.initial_risk > 0 else 0.0

    @property
    def return_pct(self) -> float:
        return self.net_pnl / self.entry_notional if self.entry_notional > 0 else 0.0


@dataclass(slots=True)
class AccountState:
    """Snapshot of the account as reported by the exchange (or simulated)."""

    total_balance: float
    available_balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    timestamp: int
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def margin_usage(self) -> float:
        return self.margin_used / self.equity if self.equity > 0 else 0.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Outcome of the risk engine for one opportunity. Always audited."""

    approved: bool
    intent: OrderIntent | None = None
    reason: RejectionReason | None = None
    detail: str = ""
    checks: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def reject(cls, reason: RejectionReason, detail: str = "", **checks: Any) -> RiskDecision:
        return cls(approved=False, reason=reason, detail=detail, checks=checks)

    @classmethod
    def approve(cls, intent: OrderIntent, **checks: Any) -> RiskDecision:
        return cls(approved=True, intent=intent, checks=checks)


@dataclass(frozen=True, slots=True)
class RiskEvent:
    event_type: RiskEventType
    severity: str  # INFO | WARNING | ERROR | CRITICAL
    message: str
    timestamp: int
    symbol: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
