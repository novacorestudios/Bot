"""Typed configuration.

Two layers, kept strictly apart:

* ``Secrets`` / ``Settings`` — deployment and credentials, from environment
  variables only. Never written to disk by this package, never logged.
* ``TunableConfig`` — every threshold and parameter, from YAML. Safe to commit,
  safe to diff, safe to show in a bug report.

The LIVE-mode gate lives here because it must be impossible to reach live
trading by editing a YAML file alone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tradebot.core.errors import ConfigError, SafetyError
from tradebot.core.types import TradingMode

# --------------------------------------------------------------------------- #
# Binance endpoints (official documentation)
# --------------------------------------------------------------------------- #
BINANCE_REST_PROD = "https://fapi.binance.com"
BINANCE_REST_TESTNET = "https://testnet.binancefuture.com"
BINANCE_WS_PROD = "wss://fstream.binance.com"
BINANCE_WS_TESTNET = "wss://stream.binancefuture.com"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Tunable sections
# --------------------------------------------------------------------------- #
class AccountConfig(_Model):
    initial_capital: float = Field(200.0, gt=0)
    quote_asset: str = "USDT"


class RiskConfig(_Model):
    risk_per_trade: float = Field(0.005, gt=0, le=0.1)
    min_risk_per_trade: float = Field(0.001, gt=0)
    max_risk_per_trade: float = Field(0.02, gt=0, le=0.1)

    max_concurrent_positions: int = Field(4, ge=1, le=50)
    max_total_risk: float = Field(0.02, gt=0, le=0.5)
    max_symbol_exposure: float = Field(1.0, gt=0)
    max_direction_exposure: float = Field(3.0, gt=0)
    max_total_exposure: float = Field(4.0, gt=0)
    max_margin_usage: float = Field(0.5, gt=0, le=1.0)
    max_margin_per_trade: float = Field(5.0, gt=0)
    max_total_allocated_margin: float = Field(20.0, gt=0)

    max_leverage: int = Field(5, ge=1, le=125)
    min_leverage: int = Field(1, ge=1)
    leverage_volatility_threshold: float = Field(0.01, gt=0)
    min_liquidation_distance_multiple: float = Field(3.0, ge=1.0)

    max_pair_correlation: float = Field(0.75, ge=0, le=1)
    max_portfolio_correlation: float = Field(0.60, ge=0, le=1)
    correlation_lookback_bars: int = Field(240, ge=30)
    correlation_timeframe: str = "5m"
    min_effective_positions_ratio: float = Field(0.55, gt=0, le=1)

    max_daily_loss: float = Field(0.02, gt=0, le=1)
    max_hourly_loss: float = Field(0.01, gt=0, le=1)
    max_drawdown: float = Field(0.10, gt=0, le=1)
    max_consecutive_losses: int = Field(5, ge=1)
    day_reset_hour_utc: int = Field(0, ge=0, le=23)

    def expected_edge_notional(self, equity: float) -> float:
        """Largest notional the pre-trade cost model should price.

        The exposure ceiling can be much larger than the position that the
        absolute margin and leverage controls can actually admit. Costing that
        impossible size overstates market impact and rejects the wrong trade.
        """
        exposure_estimate = equity * self.max_symbol_exposure
        executable_cap = self.max_margin_per_trade * self.max_leverage
        return min(exposure_estimate, executable_cap)

    @model_validator(mode="after")
    def _coherent(self) -> RiskConfig:
        if self.min_risk_per_trade > self.max_risk_per_trade:
            raise ValueError("min_risk_per_trade exceeds max_risk_per_trade")
        if not self.min_risk_per_trade <= self.risk_per_trade <= self.max_risk_per_trade:
            raise ValueError("risk_per_trade outside [min_risk_per_trade, max_risk_per_trade]")
        if self.min_leverage > self.max_leverage:
            raise ValueError("min_leverage exceeds max_leverage")
        if self.max_margin_per_trade > self.max_total_allocated_margin:
            raise ValueError("max_margin_per_trade exceeds max_total_allocated_margin")
        if self.max_hourly_loss > self.max_daily_loss:
            raise ValueError("max_hourly_loss exceeds max_daily_loss")
        if self.max_daily_loss > self.max_drawdown:
            raise ValueError("max_daily_loss exceeds max_drawdown")
        if self.risk_per_trade * self.max_concurrent_positions < self.max_total_risk:
            # Not fatal, but it means max_total_risk can never bind. Warn loudly.
            pass
        return self


class KillSwitchConfig(_Model):
    max_api_errors_per_5min: int = Field(20, ge=1)
    max_rejected_orders_per_hour: int = Field(5, ge=1)
    max_slippage: float = Field(0.0025, gt=0)
    max_reconciliation_mismatches: int = Field(2, ge=1)
    ws_stale_seconds: float = Field(30.0, gt=0)
    abnormal_return_threshold: float = Field(0.05, gt=0)
    abnormal_window_bars: int = Field(5, ge=1)
    auto_rearm_seconds: int = Field(900, ge=0)


class CooldownConfig(_Model):
    base_seconds: int = Field(120, ge=0)
    after_loss_seconds: int = Field(420, ge=0)
    after_win_seconds: int = Field(90, ge=0)
    consecutive_loss_multiplier: float = Field(2.0, ge=1.0)
    max_seconds: int = Field(3600, ge=0)
    per_strategy: bool = True


class ScannerWeights(_Model):
    liquidity: float = 0.16
    volume: float = 0.10
    recent_volume: float = 0.08
    spread: float = 0.12
    volatility: float = 0.12
    momentum: float = 0.08
    trend: float = 0.08
    volume_anomaly: float = 0.06
    breakout_potential: float = 0.06
    mean_reversion_potential: float = 0.04
    funding: float = 0.04
    structure: float = 0.03
    book_imbalance: float = 0.03

    def normalised(self) -> dict[str, float]:
        raw = self.model_dump()
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("scanner weights sum to zero")
        return {k: v / total for k, v in raw.items()}


class ScannerPenalties(_Model):
    estimated_cost: float = 20.0
    risk: float = 15.0


class ScannerConfig(_Model):
    scan_interval_sec: int = Field(300, ge=5)
    signal_interval_sec: int = Field(15, ge=1)
    top_markets: int = Field(25, ge=1, le=200)
    quote_asset: str = "USDT"
    contract_type: str = "PERPETUAL"
    min_24h_quote_volume: float = Field(2e7, ge=0)
    max_spread_bps: float = Field(6.0, gt=0)
    min_price: float = Field(1e-7, gt=0)
    deny_list: tuple[str, ...] = ()
    allow_list: tuple[str, ...] = ()
    scoring_lookback_bars: int = Field(200, ge=50)
    depth_limit: int = Field(20, ge=0)
    weights: ScannerWeights = ScannerWeights()
    penalties: ScannerPenalties = ScannerPenalties()
    volatility_target_low: float = Field(0.0025, gt=0)
    volatility_target_high: float = Field(0.015, gt=0)

    @model_validator(mode="after")
    def _bands(self) -> ScannerConfig:
        if self.volatility_target_low >= self.volatility_target_high:
            raise ValueError("volatility_target_low must be below volatility_target_high")
        return self


class StreamConfig(_Model):
    """Live market data streaming.

    ``stale_after_sec`` is the line between "my view of this symbol is current"
    and "I must not open a position on it". It is deliberately generous
    relative to the stream cadence: a 1m kline stream ticks continuously, so a
    30-second silence means something is genuinely wrong, not merely quiet.
    """

    enabled: bool = True
    include_book: bool = True
    include_mark: bool = True
    lagging_after_sec: float = Field(10.0, gt=0)
    stale_after_sec: float = Field(30.0, gt=0)
    rest_fallback_enabled: bool = True
    user_stream_enabled: bool = True
    keepalive_interval_sec: float = Field(1800.0, gt=0, le=3300)

    @model_validator(mode="after")
    def _ordering(self) -> StreamConfig:
        if self.lagging_after_sec >= self.stale_after_sec:
            raise ValueError("lagging_after_sec must be below stale_after_sec")
        return self


class TimeframeConfig(_Model):
    primary: str = "5m"
    fast: str = "1m"
    entry: str = "3m"
    context: str = "15m"
    higher: str = "1h"
    history_bars: int = Field(500, ge=100)

    def all(self) -> tuple[str, ...]:
        seen: list[str] = []
        for tf in (self.fast, self.entry, self.primary, self.context, self.higher):
            if tf not in seen:
                seen.append(tf)
        return tuple(seen)


class RegimeConfig(_Model):
    adx_period: int = Field(14, ge=2)
    adx_strong_trend: float = Field(28.0, gt=0)
    adx_weak_trend: float = Field(18.0, gt=0)
    atr_period: int = Field(14, ge=2)
    high_volatility_percentile: float = Field(0.80, gt=0, lt=1)
    low_volatility_percentile: float = Field(0.20, gt=0, lt=1)
    regime_lookback: int = Field(200, ge=50)
    bb_period: int = Field(20, ge=2)
    bb_std: float = Field(2.0, gt=0)
    squeeze_bandwidth: float = Field(0.02, gt=0)
    breakout_lookback: int = Field(20, ge=2)
    panic_return_threshold: float = Field(0.04, gt=0)
    panic_window_bars: int = Field(5, ge=1)
    panic_volume_multiple: float = Field(5.0, gt=0)
    strategy_weights: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bands(self) -> RegimeConfig:
        if self.adx_weak_trend >= self.adx_strong_trend:
            raise ValueError("adx_weak_trend must be below adx_strong_trend")
        if self.low_volatility_percentile >= self.high_volatility_percentile:
            raise ValueError("low_volatility_percentile must be below high")
        return self

    def weights_for(self, regime: str) -> dict[str, float]:
        return dict(self.strategy_weights.get(regime, {}))


class AggregatorConfig(_Model):
    min_consensus: float = Field(55.0, ge=0, le=100)
    min_agreeing_strategies: int = Field(2, ge=1)
    max_conflict_ratio: float = Field(0.45, ge=0, le=1)
    min_signal_confidence: float = Field(50.0, ge=0, le=100)


class OpportunityWeights(_Model):
    market_quality: float = 0.18
    consensus: float = 0.26
    momentum: float = 0.08
    volume: float = 0.08
    trend: float = 0.06
    liquidity: float = 0.10
    volatility_fit: float = 0.08
    execution_quality: float = 0.08
    risk_reward: float = 0.08

    def normalised(self) -> dict[str, float]:
        raw = self.model_dump()
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("opportunity weights sum to zero")
        return {k: v / total for k, v in raw.items()}


class OpportunityPenalties(_Model):
    cost: float = 25.0
    correlation: float = 20.0


class OpportunityConfig(_Model):
    min_score: float = Field(70.0, ge=0, le=100)
    exceptional: float = Field(90.0, ge=0, le=100)
    strong: float = Field(80.0, ge=0, le=100)
    moderate: float = Field(70.0, ge=0, le=100)
    weights: OpportunityWeights = OpportunityWeights()
    penalties: OpportunityPenalties = OpportunityPenalties()
    #: How long a queued opportunity stays valid. A signal computed on a 5m bar
    #: is not still valid ten minutes later.
    queue_ttl_sec: float = Field(60.0, gt=0)
    queue_max_size: int = Field(50, ge=1, le=500)


class EdgeConfig(_Model):
    taker_fee: float = Field(0.0004, ge=0)
    maker_fee: float = Field(0.0002, ge=0)
    assume_taker_entry: bool = True
    assume_taker_exit: bool = True
    base_slippage_bps: float = Field(1.0, ge=0)
    impact_coefficient: float = Field(0.5, ge=0)
    spread_cost_fraction: float = Field(0.5, ge=0, le=1)
    funding_interval_hours: int = Field(8, ge=1)
    min_expected_edge: float = Field(0.0008, ge=0)
    win_rate_prior: float = Field(0.45, gt=0, lt=1)
    win_rate_prior_weight: float = Field(40.0, ge=0)
    # Edge is learned for the setup actually being traded, not merely for the
    # strategy name. Sparse contextual cells fall back to the pooled strategy
    # posterior until they contain enough target-before-stop observations.
    contextual_enabled: bool = True
    contextual_min_trades: int = Field(20, ge=1)
    # One-sided Wilson lower bound. 1.0 is deliberately conservative without
    # making a small research account permanently inert.
    confidence_lower_bound_z: float = Field(1.0, ge=0.0, le=3.0)

    # -- bootstrap ---------------------------------------------------------
    # Without history the win probability shrinks to `win_rate_prior`, which at
    # a reward:risk of 2.0 makes EVERY trade negative-edge. No trade passes, so
    # no history accumulates: the system cannot bootstrap.
    #
    # In BACKTEST that circularity has to be broken, because measuring the win
    # rate is the whole point of a backtest. When enabled, a strategy with
    # insufficient evidence is ASSUMED to win at its break-even rate plus
    # `bootstrap_win_rate_margin`, and every trade taken on that assumption is
    # counted and reported separately.
    #
    # It must stay FALSE for live trading. Refusing to risk money on an
    # unproven strategy is correct; live seeds its statistics from validated
    # backtest results instead (EdgeCalculator.seed_from).
    bootstrap_enabled: bool = False
    bootstrap_win_rate_margin: float = Field(0.05, ge=0.0, le=0.5)
    bootstrap_min_trades: int = Field(30, ge=0)


class TradeConfig(_Model):
    max_duration_sec: int = Field(3600, ge=60, le=3600)
    raw_signal_mode: bool = False
    raw_stop_pct: float = Field(0.05, gt=0, lt=1)
    raw_take_profit_min_pct: float = Field(0.0005, gt=0, lt=1)
    raw_take_profit_max_pct: float = Field(0.01, gt=0, lt=1)
    exit_on_negative_edge: bool = True
    exit_on_regime_change: bool = True
    exit_on_signal_flip: bool = True

    @model_validator(mode="after")
    def _raw_exit_band(self) -> TradeConfig:
        if self.raw_take_profit_min_pct > self.raw_take_profit_max_pct:
            raise ValueError("raw_take_profit_min_pct exceeds raw_take_profit_max_pct")
        return self


class StopsConfig(_Model):
    atr_period: int = Field(14, ge=2)
    atr_multiple: float = Field(1.5, gt=0)
    min_stop_pct: float = Field(0.002, gt=0)
    max_stop_pct: float = Field(0.015, gt=0)
    structure_lookback: int = Field(20, ge=2)
    structure_buffer_atr: float = Field(0.25, ge=0)

    @model_validator(mode="after")
    def _bands(self) -> StopsConfig:
        if self.min_stop_pct >= self.max_stop_pct:
            raise ValueError("min_stop_pct must be below max_stop_pct")
        return self


class PartialTakeProfitConfig(_Model):
    enabled: bool = False
    first_target_rr: float = Field(1.0, gt=0)
    first_target_fraction: float = Field(0.5, gt=0, lt=1)
    move_stop_to_breakeven: bool = True


class TargetsConfig(_Model):
    base_rr: float = Field(1.6, gt=0)
    min_rr: float = Field(1.1, gt=0)
    max_rr: float = Field(4.0, gt=0)
    atr_multiple_cap: float = Field(4.0, gt=0)
    partial_take_profit: PartialTakeProfitConfig = PartialTakeProfitConfig()

    @model_validator(mode="after")
    def _bands(self) -> TargetsConfig:
        if not self.min_rr <= self.base_rr <= self.max_rr:
            raise ValueError("base_rr must lie within [min_rr, max_rr]")
        return self


class TrailingStopConfig(_Model):
    enabled: bool = True
    activation_r: float = Field(1.0, gt=0)
    atr_multiple: float = Field(1.8, gt=0)
    never_below_breakeven: bool = True


class AllocationConfig(_Model):
    enabled: bool = True
    min_weight: float = Field(0.4, gt=0)
    max_weight: float = Field(2.0, gt=0)
    lookback_trades: int = Field(100, ge=10)
    min_trades_for_adjustment: int = Field(30, ge=1)

    @model_validator(mode="after")
    def _bands(self) -> AllocationConfig:
        if self.min_weight >= self.max_weight:
            raise ValueError("min_weight must be below max_weight")
        return self


class StrategyKillSwitchConfig(_Model):
    enabled: bool = True
    lookback_trades: int = Field(100, ge=10)
    min_trades: int = Field(40, ge=5)
    min_profit_factor: float = Field(0.85, ge=0)
    max_drawdown: float = Field(0.08, gt=0)
    min_expectancy: float = -0.0005
    disable_seconds: int = Field(21600, ge=60)


class ExecutionConfig(_Model):
    entry_order_type: str = "MARKET"
    limit_offset_bps: float = Field(1.0, ge=0)
    limit_timeout_sec: float = Field(5.0, gt=0)
    stop_order_type: str = "STOP_MARKET"
    take_profit_order_type: str = "TAKE_PROFIT_MARKET"
    max_entry_slippage: float = Field(0.003, gt=0)
    max_retries: int = Field(3, ge=0)
    retry_backoff_sec: float = Field(0.5, gt=0)
    reconcile_interval_sec: int = Field(60, ge=5)
    monitor_interval_sec: float = Field(1.0, gt=0)
    max_min_notional_ratio: float = Field(0.5, gt=0, le=1)
    #: Fills required before the measured slippage bias is fed back into the
    #: cost model. A handful of fills is not evidence of a bias.
    quality_min_samples: int = Field(10, ge=1)
    #: Ceiling on that correction, so one pathological session cannot make the
    #: cost model so pessimistic that nothing ever trades again.
    quality_max_adjustment: float = Field(0.002, gt=0, le=0.05)

    @field_validator("entry_order_type")
    @classmethod
    def _entry_type(cls, v: str) -> str:
        if v not in {"MARKET", "LIMIT"}:
            raise ValueError("entry_order_type must be MARKET or LIMIT")
        return v


class PaperConfig(_Model):
    latency_ms: float = Field(120.0, ge=0)
    latency_jitter_ms: float = Field(80.0, ge=0)
    slippage_bps: float = Field(1.5, ge=0)
    adverse_slippage_probability: float = Field(0.65, ge=0, le=1)
    partial_fill_probability: float = Field(0.10, ge=0, le=1)
    reject_probability: float = Field(0.005, ge=0, le=1)
    apply_funding: bool = True


class BacktestConfig(_Model):
    taker_fee: float = Field(0.0004, ge=0)
    maker_fee: float = Field(0.0002, ge=0)
    slippage_bps: float = Field(1.5, ge=0)
    spread_bps: float = Field(1.0, ge=0)
    apply_funding: bool = True
    fill_model: str = "next_open"
    intrabar: str = "pessimistic"
    warmup_bars: int = Field(250, ge=0)

    @field_validator("fill_model")
    @classmethod
    def _fill(cls, v: str) -> str:
        if v not in {"next_open", "close"}:
            raise ValueError("fill_model must be next_open or close")
        return v

    @field_validator("intrabar")
    @classmethod
    def _intrabar(cls, v: str) -> str:
        if v not in {"pessimistic", "optimistic", "proportional"}:
            raise ValueError("intrabar must be pessimistic, optimistic or proportional")
        return v


class WalkForwardConfig(_Model):
    train_days: int = Field(30, ge=1)
    validation_days: int = Field(7, ge=1)
    test_days: int = Field(7, ge=1)
    step_days: int = Field(7, ge=1)
    min_trades_per_fold: int = Field(20, ge=1)


class MonteCarloConfig(_Model):
    iterations: int = Field(2000, ge=100)
    method: str = "bootstrap"
    drawdown_percentile: float = Field(0.05, gt=0, lt=1)


class DatabaseConfig(_Model):
    market_snapshot_retention_days: int = Field(14, ge=0)
    signal_retention_days: int = Field(30, ge=0)
    decision_retention_days: int = Field(90, ge=0)
    prune_interval_sec: int = Field(3600, ge=60)


class PreservationConfig(_Model):
    """Capital preservation thresholds.

    All values are positive fractions of equity. The thresholds must be ordered
    cautious < defensive < halt, and the hysteresis band must be small enough
    that recovering out of a mode is possible at all.
    """

    enabled: bool = True
    cautious_drawdown: float = Field(0.03, gt=0, le=1)
    defensive_drawdown: float = Field(0.06, gt=0, le=1)
    halt_drawdown: float = Field(0.10, gt=0, le=1)
    halt_daily_loss: float = Field(0.02, gt=0, le=1)
    cautious_consecutive_losses: int = Field(3, ge=1)
    defensive_consecutive_losses: int = Field(4, ge=1)
    #: How far below a threshold the account must recover before the mode
    #: relaxes. Without it, a drawdown sitting on a line flips modes every cycle.
    recovery_hysteresis: float = Field(0.01, ge=0, le=1)
    #: Minimum time in a mode before it may loosen. Escalation ignores it.
    min_mode_duration_sec: float = Field(900.0, ge=0)
    #: Equity held back from deployment, to pay funding, fees and adverse
    #: margin moves on positions already open.
    capital_reserve_fraction: float = Field(0.10, ge=0, lt=1)

    @model_validator(mode="after")
    def _ordered(self) -> PreservationConfig:
        if not self.cautious_drawdown < self.defensive_drawdown < self.halt_drawdown:
            raise ValueError(
                "preservation drawdown thresholds must increase: cautious < defensive < halt"
            )
        if self.cautious_consecutive_losses > self.defensive_consecutive_losses:
            raise ValueError("cautious_consecutive_losses exceeds defensive_consecutive_losses")
        if self.recovery_hysteresis >= self.cautious_drawdown:
            raise ValueError(
                "recovery_hysteresis is at or above cautious_drawdown, so the "
                "engine could never relax out of CAUTIOUS"
            )
        return self


class MatricesConfig(_Model):
    """Strategy x regime and symbol x strategy performance tables.

    Recording is always on — the tables are how an operator answers "is mean
    reversion losing money because it is a bad strategy, or because it keeps
    being run in a trend?".

    Feeding them back into selection is a separate, OFF-by-default decision, and
    deliberately so: suppressing a combination on a handful of trades is
    overfitting against your own history. On a 75 USDT account taking a few
    trades a day, a cell reaches ``min_trades`` after weeks, and until then the
    honest answer is "not enough evidence". Turn ``feedback_enabled`` on once
    the tables have filled and you have looked at them.
    """

    feedback_enabled: bool = False
    strategy_regime_min_trades: int = Field(30, ge=5)
    symbol_strategy_min_trades: int = Field(25, ge=5)
    #: Expectancy in R at or below which a cell is fully suppressed.
    floor_expectancy_r: float = Field(-0.25, lt=0)
    #: How far a cell may be suppressed. 0.0 means "stop that combination".
    min_multiplier: float = Field(0.0, ge=0, le=1)


class HealthConfig(_Model):
    heartbeat_interval_sec: float = Field(5.0, gt=0)
    component_timeout_sec: float = Field(60.0, gt=0)
    memory_warn_mb: float = Field(900.0, gt=0)
    cpu_warn_pct: float = Field(85.0, gt=0, le=100)
    #: A trade with no audit trail cannot be reconstructed, reconciled or
    #: learned from. When the database is down, NEW entries stop; exits and the
    #: management of open positions are never gated on it.
    database_critical: bool = True
    #: Attempts to reopen a failed database connection before giving up on the
    #: current cycle. Reconnection keeps retrying on later cycles regardless.
    database_reconnect_attempts: int = Field(3, ge=0)
    database_reconnect_backoff_sec: float = Field(5.0, gt=0)


class AIConfig(_Model):
    enabled: bool = False
    anomaly_detection: bool = True
    post_trade_analysis: bool = True


class TunableConfig(_Model):
    """The whole YAML file, validated."""

    account: AccountConfig = AccountConfig()
    risk: RiskConfig = RiskConfig()
    kill_switches: KillSwitchConfig = KillSwitchConfig()
    preservation: PreservationConfig = PreservationConfig()
    matrices: MatricesConfig = MatricesConfig()
    cooldown: CooldownConfig = CooldownConfig()
    scanner: ScannerConfig = ScannerConfig()
    timeframes: TimeframeConfig = TimeframeConfig()
    stream: StreamConfig = StreamConfig()
    regime: RegimeConfig = RegimeConfig()
    aggregator: AggregatorConfig = AggregatorConfig()
    opportunity: OpportunityConfig = OpportunityConfig()
    edge: EdgeConfig = EdgeConfig()
    trade: TradeConfig = TradeConfig()
    stops: StopsConfig = StopsConfig()
    targets: TargetsConfig = TargetsConfig()
    trailing_stop: TrailingStopConfig = TrailingStopConfig()
    allocation: AllocationConfig = AllocationConfig()
    strategy_kill_switch: StrategyKillSwitchConfig = StrategyKillSwitchConfig()
    execution: ExecutionConfig = ExecutionConfig()
    paper: PaperConfig = PaperConfig()
    backtest: BacktestConfig = BacktestConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    monte_carlo: MonteCarloConfig = MonteCarloConfig()
    database: DatabaseConfig = DatabaseConfig()
    health: HealthConfig = HealthConfig()
    ai: AIConfig = AIConfig()
    strategies: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cross_section(self) -> TunableConfig:
        # A stop can never be tighter than the round-trip cost, or the trade is
        # mathematically unable to pay for itself.
        round_trip = 2 * self.edge.taker_fee
        if self.stops.min_stop_pct < round_trip:
            raise ValueError(
                f"stops.min_stop_pct ({self.stops.min_stop_pct}) is below the "
                f"round-trip fee ({round_trip}); such a trade cannot be profitable"
            )
        return self


# --------------------------------------------------------------------------- #
# Environment settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Deployment settings and secrets. Environment variables only."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    trading_mode: TradingMode = TradingMode.PAPER
    i_understand_live_trading_risk: str = "NO"

    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True
    binance_rest_url: str = ""
    binance_ws_url: str = ""
    binance_recv_window: int = Field(5000, ge=100, le=60000)

    config_file: str = "config/config.yaml"
    strategies_file: str = "config/strategies.yaml"

    database_url: str = "sqlite+aiosqlite:///data/tradebot.db"

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False

    dashboard_enabled: bool = True
    # Binding to all interfaces is intentional but NOT the effective default:
    # app/runner.py overrides this to 127.0.0.1 whenever dashboard_token is
    # empty, so an unauthenticated dashboard is never reachable off-host.
    dashboard_host: str = "0.0.0.0"  # noqa: S104  # nosec B104
    dashboard_port: int = 8080
    dashboard_token: str = ""

    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/tradebot.log"

    @property
    def rest_url(self) -> str:
        if self.binance_rest_url:
            return self.binance_rest_url.rstrip("/")
        return BINANCE_REST_TESTNET if self.binance_testnet else BINANCE_REST_PROD

    @property
    def ws_url(self) -> str:
        if self.binance_ws_url:
            return self.binance_ws_url.rstrip("/")
        return BINANCE_WS_TESTNET if self.binance_testnet else BINANCE_WS_PROD

    @property
    def has_credentials(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    def redacted(self) -> dict[str, Any]:
        """A dict safe to log: every secret replaced by a presence marker."""
        secret_fields = {
            "binance_api_key",
            "binance_api_secret",
            "telegram_bot_token",
            "dashboard_token",
        }
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if name in secret_fields:
                out[name] = f"<set:{len(str(value))}>" if value else "<unset>"
            else:
                out[name] = value
        return out


# --------------------------------------------------------------------------- #
# Assembly and safety gate
# --------------------------------------------------------------------------- #
class AppConfig:
    """Settings + tunables, with the live-trading gate already enforced."""

    def __init__(
        self, settings: Settings, tunables: TunableConfig, live_flag: bool = False
    ) -> None:
        self.settings = settings
        self.tunables = tunables
        self.live_flag = live_flag
        self.mode = settings.trading_mode

    # Convenience passthroughs — used everywhere, so worth the shorthand.
    @property
    def risk(self) -> RiskConfig:
        return self.tunables.risk

    @property
    def scanner(self) -> ScannerConfig:
        return self.tunables.scanner

    @property
    def edge(self) -> EdgeConfig:
        return self.tunables.edge

    @property
    def execution(self) -> ExecutionConfig:
        return self.tunables.execution

    @property
    def is_live(self) -> bool:
        return self.mode is TradingMode.LIVE


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new mapping.

    Nested mappings merge key by key; anything else replaces wholesale. This is
    what lets an overlay change one threshold without restating every section —
    and, more importantly, without silently emptying the sections it omits. An
    overlay that dropped ``regime.strategy_weights`` would leave no strategy
    enabled in any regime, producing a backtest with zero trades that looked
    like a strategy failure rather than a configuration one.
    """
    out = dict(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError("configuration file not found", path=str(path))
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"configuration file is not valid YAML: {exc}", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration file must contain a mapping", path=str(path))
    return raw


def load_tunables(
    config_file: str | Path, strategies_file: str | Path | None = None
) -> TunableConfig:
    """Read and validate the YAML tunables.

    A file may declare ``extends: <path>`` (relative to its own directory) to
    inherit from another config and override only the keys it names.
    """
    path = Path(config_file)
    raw = _read_yaml(path)

    parent = raw.pop("extends", None)
    if parent:
        resolved = (path.parent / str(parent)).resolve()
        base = _read_yaml(resolved)
        if base.pop("extends", None):
            raise ConfigError("'extends' may only be one level deep", path=str(resolved))
        raw = _deep_merge(base, raw)

    if strategies_file:
        spath = Path(strategies_file)
        if spath.is_file():
            try:
                strat = yaml.safe_load(spath.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(
                    f"strategies file is not valid YAML: {exc}", path=str(spath)
                ) from exc
            if not isinstance(strat, dict):
                raise ConfigError("strategies file must contain a mapping", path=str(spath))
            raw["strategies"] = strat

    try:
        return TunableConfig(**raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"configuration is invalid: {exc}", path=str(path)) from exc


def enforce_live_gate(settings: Settings, live_flag: bool) -> None:
    """Refuse to run LIVE unless every independent confirmation agrees.

    Three independent switches must line up: the mode env var, the explicit
    acknowledgement env var, and a command-line flag. A YAML edit alone, or an
    env var alone, can never reach live trading.
    """
    if settings.trading_mode is not TradingMode.LIVE:
        if live_flag:
            raise SafetyError(
                "--live was passed but TRADING_MODE is not LIVE; refusing to guess",
                trading_mode=settings.trading_mode.value,
            )
        return

    if not live_flag:
        raise SafetyError("LIVE mode requires the explicit --live command-line flag")
    if settings.i_understand_live_trading_risk.strip().upper() != "YES":
        raise SafetyError(
            "LIVE mode requires I_UNDERSTAND_LIVE_TRADING_RISK=YES in the environment"
        )
    if not settings.has_credentials:
        raise SafetyError("LIVE mode requires BINANCE_API_KEY and BINANCE_API_SECRET")
    if settings.binance_testnet:
        raise SafetyError(
            "LIVE mode with BINANCE_TESTNET=true is contradictory; "
            "set BINANCE_TESTNET=false for real trading or use PAPER mode"
        )


def load_config(live_flag: bool = False, env: dict[str, str] | None = None) -> AppConfig:
    """Load settings and tunables, then enforce the live gate."""
    if env is not None:
        settings = Settings(**{k.lower(): v for k, v in env.items()})  # type: ignore[arg-type]
    else:
        settings = Settings()

    tunables = load_tunables(settings.config_file, settings.strategies_file)
    enforce_live_gate(settings, live_flag)

    if settings.trading_mode is TradingMode.LIVE:
        os.environ.setdefault("TRADEBOT_LIVE_CONFIRMED", "1")

    return AppConfig(settings=settings, tunables=tunables, live_flag=live_flag)
