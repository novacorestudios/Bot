"""The risk engine.

**Every trade passes through here, and nothing bypasses it.** This is the only
module in the system that constructs an `OrderIntent`, which is the only object
the execution engine will act on. A strategy, the AI layer, the dashboard and
the scanner can all propose; only this can approve.

The checks run in a deliberate order — cheapest and most categorical first, so
an expensive correlation matrix is never computed for a trade that a kill switch
already forbids:

1. kill switches (account-level circuit breakers)
2. entries-blocked flag (reconciliation in progress, safe mode)
3. duplicate position / in-flight intent for the symbol
4. cooldown
5. strategy suspension
6. stop-loss present and correctly placed
7. position sizing (equity, risk fraction, stop distance, filters, leverage)
8. correlation against existing exposure
9. portfolio limits (count, risk budget, exposure, margin)

Every outcome — approval or rejection — carries a reason code and a `checks`
dictionary, and is written to the audit log. "Why didn't it trade?" is as
important a question as "why did it?", and both must be answerable months later.

The engine never blocks an exit. Not for a kill switch, not for a cooldown, not
for a portfolio limit. Being unable to close a position is strictly worse than
any condition that would justify not opening one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tradebot.core.clock import SystemClock
from tradebot.core.config import TunableConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import (
    Direction,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    RejectionReason,
    RiskDecision,
    SymbolInfo,
    new_id,
)
from tradebot.market.candles import CandleStore
from tradebot.risk.allocation import StrategyAllocator, StrategyKillSwitch
from tradebot.risk.cooldown import CooldownManager
from tradebot.risk.correlation import CorrelationEngine
from tradebot.risk.killswitch import KillSwitchManager
from tradebot.risk.matrices import MatrixSet
from tradebot.risk.portfolio import PortfolioState, PortfolioTracker
from tradebot.risk.preservation import CapitalPreservation
from tradebot.risk.sizing import PositionSizer
from tradebot.signals.pipeline import Opportunity

log = get_logger(__name__)


@dataclass(slots=True)
class RiskContext:
    """Everything the engine needs to judge one opportunity."""

    equity: float
    available_balance: float
    positions: dict[str, Position]
    prices: dict[str, float]
    symbol_info: SymbolInfo
    realized_pnl_today: float = 0.0
    unrealized_pnl: float = 0.0
    data_age_sec: float = 0.0
    connected: bool = True
    entries_blocked: bool = False
    entries_blocked_reason: str = ""
    in_flight: set[str] = field(default_factory=set)
    #: Drawdown from peak equity, as a positive fraction. Drives the capital
    #: preservation mode.
    drawdown: float = 0.0
    now: float = field(default_factory=time.time)


class RiskEngine:
    """The single gate between an opportunity and an order."""

    def __init__(
        self,
        config: TunableConfig,
        candles: CandleStore,
        clock=None,
    ) -> None:
        self.config = config
        self.clock = clock

        self.sizer = PositionSizer(config.risk, config.execution.max_min_notional_ratio)
        self.portfolio = PortfolioTracker(config.risk)
        self.correlation = CorrelationEngine(config.risk, candles)
        self.kill_switches = KillSwitchManager(config.kill_switches, config.risk, clock)
        self.cooldowns = CooldownManager(config.cooldown, clock)
        self.allocator = StrategyAllocator(config.allocation)
        self.preservation = CapitalPreservation(config.preservation, clock or SystemClock())
        self._seen_day_rollovers = 0
        # Evidence about which strategy works in which regime and on which
        # symbol. It may only ever reduce a weight, never raise one.
        self.matrices = MatrixSet(config.matrices)
        self.strategy_kill_switch = StrategyKillSwitch(
            config.strategy_kill_switch, self.allocator, clock
        )

        #: Strategies suspended by the strategy kill switch, name -> expiry.
        self.suspended_strategies: dict[str, float] = {}

        self.approvals = 0
        self.rejections: dict[str, int] = {}
        self.decisions: list[RiskDecision] = []

    def _now(self) -> float:
        return self.clock.now() if self.clock is not None else time.time()

    # ------------------------------------------------------------------ #
    # The gate
    # ------------------------------------------------------------------ #
    def evaluate(self, opportunity: Opportunity, context: RiskContext) -> RiskDecision:
        """Approve an opportunity into an OrderIntent, or reject it with a reason."""
        symbol = opportunity.symbol
        signal = opportunity.signal
        direction = signal.direction

        state = self.portfolio.state(
            equity=context.equity,
            available_balance=context.available_balance,
            positions=context.positions,
            prices=context.prices,
            unrealized_pnl=context.unrealized_pnl,
            realized_pnl_today=context.realized_pnl_today,
        )

        # -- 1. account-level kill switches --------------------------------- #
        self.kill_switches.evaluate(context.equity, context.data_age_sec, context.connected)
        if not self.kill_switches.entries_allowed:
            return self._reject(
                RejectionReason.KILL_SWITCH_ACTIVE,
                self.kill_switches.blocking_reason(),
                switches=[s.value for s in self.kill_switches.active],
            )

        # -- 1b. capital preservation ---------------------------------------- #
        # A HALTED day ends with the day. Nothing else releases HALTED: a
        # drawdown that has partly recovered is not evidence its cause is gone.
        if self.kill_switches.day_rollovers != self._seen_day_rollovers:
            self._seen_day_rollovers = self.kill_switches.day_rollovers
            self.preservation.reset("new trading day")

        # Evaluated on every decision so a drawdown that opened mid-cycle is
        # reflected immediately, not at the next heartbeat.
        preservation = self.preservation.evaluate(
            drawdown=context.drawdown,
            daily_loss=max(0.0, -safe_div(context.realized_pnl_today, context.equity, 0.0)),
            consecutive_losses=self.kill_switches.consecutive_losses,
            equity=context.equity,
        )
        if self.config.preservation.enabled and not preservation.mode.allows_entries:
            return self._reject(
                RejectionReason.KILL_SWITCH_ACTIVE,
                f"capital preservation is HALTED: {preservation.reason}",
                preservation_mode=preservation.mode.value,
            )

        # -- 2. entries blocked (reconciliation, safe mode) ------------------ #
        if context.entries_blocked:
            return self._reject(
                RejectionReason.ENTRIES_BLOCKED,
                context.entries_blocked_reason or "entries are administratively blocked",
            )

        # -- 3. duplicates and races ----------------------------------------- #
        if symbol in context.positions:
            return self._reject(
                RejectionReason.ALREADY_IN_POSITION,
                f"a position in {symbol} is already open",
            )
        if symbol in context.in_flight:
            return self._reject(
                RejectionReason.INTENT_IN_FLIGHT,
                f"an order for {symbol} is already in flight",
            )

        # -- 3b. position count -------------------------------------------- #
        # Checked here, before sizing and before the correlation matrix: it is
        # categorical and cheap, and reporting it directly is far more useful
        # than the correlation rejection a full book would otherwise produce.
        max_positions = self.config.risk.max_concurrent_positions
        if self.config.preservation.enabled:
            max_positions = self.preservation.max_positions(max_positions)
        if state.position_count >= max_positions:
            return self._reject(
                RejectionReason.MAX_POSITIONS,
                f"{state.position_count} positions already open (limit {max_positions}"
                + (
                    f", reduced by {preservation.mode.value} mode"
                    if max_positions != self.config.risk.max_concurrent_positions
                    else ""
                )
                + ")",
                positions_open=state.position_count,
                preservation_mode=preservation.mode.value,
            )

        # -- 3c. a defensive mode raises the bar for what is worth taking ----- #
        if self.config.preservation.enabled:
            required = self.preservation.min_opportunity_score(self.config.opportunity.min_score)
            if opportunity.opportunity_score.total < required:
                return self._reject(
                    RejectionReason.LOW_OPPORTUNITY_SCORE,
                    f"score {opportunity.opportunity_score.total:.1f} is below the "
                    f"{required:.1f} required in {preservation.mode.value} mode",
                    preservation_mode=preservation.mode.value,
                )

        # -- 4. cooldown ------------------------------------------------------ #
        cooldown = self.cooldowns.check(symbol, opportunity.strategy)
        if cooldown.active:
            return self._reject(
                RejectionReason.COOLDOWN_ACTIVE,
                f"{symbol} is cooling down for another "
                f"{cooldown.remaining_sec:.0f}s ({cooldown.reason})",
                remaining_sec=cooldown.remaining_sec,
            )

        # -- 5. strategy suspension ------------------------------------------- #
        strategy = opportunity.strategy
        expiry = self.suspended_strategies.get(strategy)
        if expiry is not None:
            if self._now() < expiry:
                return self._reject(
                    RejectionReason.STRATEGY_DISABLED,
                    f"strategy {strategy} is suspended for another {expiry - self._now():.0f}s",
                )
            del self.suspended_strategies[strategy]

        # -- 5b. matrix evidence ---------------------------------------------- #
        # A strategy that loses money overall may be excellent in one regime and
        # terrible in another; the aggregate hides both facts.
        matrix_multiplier = self.matrices.multiplier(strategy, opportunity.regime, symbol)
        if matrix_multiplier <= 0.0:
            return self._reject(
                RejectionReason.STRATEGY_DISABLED,
                f"{strategy} has a losing record on {symbol} in "
                f"{opportunity.regime.value} across enough trades to matter",
                matrix_multiplier=matrix_multiplier,
            )

        # -- 6. a stop is mandatory ------------------------------------------- #
        if signal.stop_loss <= 0:
            return self._reject(
                RejectionReason.INVALID_STOP,
                "no stop loss; a position without protection is never permitted",
            )
        if direction is Direction.LONG and signal.stop_loss >= signal.entry_price:
            return self._reject(RejectionReason.INVALID_STOP, "LONG stop is not below entry")
        if direction is Direction.SHORT and signal.stop_loss <= signal.entry_price:
            return self._reject(RejectionReason.INVALID_STOP, "SHORT stop is not above entry")
        if signal.take_profit <= 0:
            return self._reject(RejectionReason.INVALID_TARGET, "no take profit")

        # -- 7. sizing --------------------------------------------------------- #
        risk_fraction = self.allocator.risk_fraction_for(
            strategy,
            self.config.risk.risk_per_trade,
            list(self.allocator.performance) or [strategy],
        )
        if self.config.preservation.enabled:
            # Preservation scales the risk fraction; it never scales it up.
            risk_fraction *= self.preservation.risk_multiplier

        # Likewise the matrices: a combination with a poor record is sized
        # smaller, not merely allowed or refused.
        risk_fraction *= matrix_multiplier

        remaining_budget = self.portfolio.remaining_risk_budget(state)
        if remaining_budget <= 0:
            return self._reject(
                RejectionReason.RISK_BUDGET_EXCEEDED,
                f"portfolio risk budget is exhausted "
                f"({state.total_open_risk:.4f} of "
                f"{state.equity * self.config.risk.max_total_risk:.4f} in use)",
            )
        # Never size a trade larger than the remaining budget allows.
        risk_fraction = min(risk_fraction, safe_div(remaining_budget, context.equity, 0.0))

        sizing = self.sizer.size(
            equity=context.equity,
            risk_fraction=risk_fraction,
            entry_price=signal.entry_price,
            stop_price=signal.stop_loss,
            direction=direction,
            symbol_info=context.symbol_info,
            available_margin=min(
                # The reserve is not idle money: it pays funding, fees and
                # adverse margin moves on positions already open.
                self.preservation.deployable(context.available_balance)
                if self.config.preservation.enabled
                else context.available_balance,
                context.equity * self.config.risk.max_margin_usage - state.margin_used,
            ),
            volatility=opportunity.market.volatility,
        )
        if not sizing.ok:
            return self._reject(
                sizing.reason or RejectionReason.SIZE_BELOW_MINIMUM,
                sizing.detail,
                **sizing.checks,
            )

        # -- 8. correlation ---------------------------------------------------- #
        held = {
            held_symbol: (
                position.direction,
                position.quantity * context.prices.get(held_symbol, position.entry_price),
            )
            for held_symbol, position in context.positions.items()
        }
        assessment = self.correlation.assess(symbol, direction, sizing.notional, held)
        if not assessment.acceptable:
            return self._reject(
                RejectionReason.CORRELATION_LIMIT,
                assessment.detail,
                max_pair_correlation=assessment.max_pair_correlation,
                portfolio_correlation=assessment.portfolio_correlation,
                effective_positions=assessment.effective_positions,
            )

        # -- 9. portfolio limits ------------------------------------------------ #
        breached, limit, detail = self.portfolio.would_breach(
            state,
            symbol,
            direction,
            sizing.notional,
            sizing.risk_amount,
            sizing.margin_required,
        )
        if breached:
            reason = {
                "MAX_POSITIONS": RejectionReason.MAX_POSITIONS,
                "TOTAL_RISK": RejectionReason.RISK_BUDGET_EXCEEDED,
                "SYMBOL_EXPOSURE": RejectionReason.SYMBOL_EXPOSURE_LIMIT,
                "DIRECTION_EXPOSURE": RejectionReason.DIRECTION_EXPOSURE_LIMIT,
                "TOTAL_EXPOSURE": RejectionReason.EXPOSURE_LIMIT,
                "MARGIN_USAGE": RejectionReason.MARGIN_LIMIT,
                "AVAILABLE_BALANCE": RejectionReason.INSUFFICIENT_BALANCE,
                "EQUITY": RejectionReason.INSUFFICIENT_BALANCE,
            }.get(limit, RejectionReason.RISK_BUDGET_EXCEEDED)
            return self._reject(reason, detail, limit=limit)

        # -- approved ------------------------------------------------------------ #
        intent = OrderIntent(
            intent_id=new_id(),
            symbol=symbol,
            direction=direction,
            side=OrderSide.for_entry(direction),
            order_type=(
                OrderType.MARKET
                if self.config.execution.entry_order_type == "MARKET"
                else OrderType.LIMIT
            ),
            quantity=sizing.quantity,
            price=(
                None if self.config.execution.entry_order_type == "MARKET" else signal.entry_price
            ),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            leverage=sizing.leverage,
            notional=sizing.notional,
            risk_amount=sizing.risk_amount,
            strategy=strategy,
            regime=opportunity.regime,
            opportunity_score=opportunity.opportunity_score.total,
            expected_net_edge=opportunity.expected_net_edge,
            metadata={
                "reference_price": signal.entry_price,
                "liquidation_price": sizing.liquidation_price,
                "liquidation_distance_multiple": sizing.liquidation_distance_multiple,
                "margin_required": sizing.margin_required,
                "risk_fraction": safe_div(sizing.risk_amount, context.equity, 0.0),
                "correlation": assessment.portfolio_correlation,
                "effective_positions": assessment.effective_positions,
                "strategies": list(signal.strategies),
                "consensus_score": signal.consensus_score,
            },
        )

        self.approvals += 1
        decision = RiskDecision.approve(
            intent,
            equity=context.equity,
            risk_amount=sizing.risk_amount,
            risk_fraction=safe_div(sizing.risk_amount, context.equity, 0.0),
            leverage=sizing.leverage,
            notional=sizing.notional,
            margin_required=sizing.margin_required,
            liquidation_distance_multiple=sizing.liquidation_distance_multiple,
            portfolio_open_risk=state.total_open_risk,
            portfolio_open_risk_pct=state.open_risk_fraction,
            portfolio_exposure_ratio=state.exposure_ratio,
            correlation=assessment.portfolio_correlation,
            effective_positions=assessment.effective_positions,
            positions_open=state.position_count,
            **sizing.checks,
        )
        self.decisions.append(decision)

        log.info(
            "risk_approved",
            symbol=symbol,
            direction=direction.value,
            strategy=strategy,
            quantity=sizing.quantity,
            notional=round(sizing.notional, 2),
            leverage=sizing.leverage,
            risk=round(sizing.risk_amount, 4),
            risk_pct=round(safe_div(sizing.risk_amount, context.equity, 0) * 100, 3),
            stop=signal.stop_loss,
            target=signal.take_profit,
            liquidation_multiple=round(sizing.liquidation_distance_multiple, 1),
        )
        return decision

    # ------------------------------------------------------------------ #
    # Post-trade bookkeeping
    # ------------------------------------------------------------------ #
    def record_trade_closed(
        self,
        symbol: str,
        strategy: str,
        won: bool,
        r_multiple: float,
        volatility: float = 0.0,
        reason: str = "",
        regime: MarketRegime | str = MarketRegime.SIDEWAYS,
        pnl: float = 0.0,
    ) -> None:
        """Update every component that learns from a completed trade."""
        self.kill_switches.record_trade_result(won)
        self.cooldowns.register_close(symbol, won, strategy, volatility, reason)
        self.allocator.record_trade(strategy, r_multiple)
        self.matrices.record(strategy, regime, symbol, won, r_multiple, pnl)

        should_disable, detail = self.strategy_kill_switch.should_disable(strategy)
        if should_disable and strategy not in self.suspended_strategies:
            self.suspended_strategies[strategy] = self.strategy_kill_switch.disable_until()
            log.critical("strategy_kill_switch_tripped", strategy=strategy, reason=detail)

    def record_order_rejected(self, symbol: str) -> None:
        self.kill_switches.record_order_rejection()
        self.cooldowns.register_rejection(symbol, 60.0)

    def record_api_error(self) -> None:
        self.kill_switches.record_api_error()

    def record_slippage(self, slippage: float) -> None:
        self.kill_switches.record_slippage(slippage)

    def record_reconciliation_mismatch(self) -> None:
        self.kill_switches.record_reconciliation_mismatch()

    # ------------------------------------------------------------------ #
    def portfolio_state(self, context: RiskContext) -> PortfolioState:
        return self.portfolio.state(
            equity=context.equity,
            available_balance=context.available_balance,
            positions=context.positions,
            prices=context.prices,
            unrealized_pnl=context.unrealized_pnl,
            realized_pnl_today=context.realized_pnl_today,
        )

    def correlation_penalties(
        self, candidates: list[str], positions: dict[str, Position], prices: dict[str, float]
    ) -> dict[str, float]:
        held = {
            symbol: (
                position.direction,
                position.quantity * prices.get(symbol, position.entry_price),
            )
            for symbol, position in positions.items()
        }
        return self.correlation.penalties_for(candidates, held)

    def strategy_weights(self, strategies: list[str]) -> dict[str, float]:
        return self.allocator.weights(strategies)

    def _reject(self, reason: RejectionReason, detail: str, **checks: Any) -> RiskDecision:
        self.rejections[reason.value] = self.rejections.get(reason.value, 0) + 1
        decision = RiskDecision.reject(reason, detail, **checks)
        self.decisions.append(decision)
        log.debug("risk_rejected", reason=reason.value, detail=detail)
        return decision

    def stats(self) -> dict[str, Any]:
        total = self.approvals + sum(self.rejections.values())
        return {
            "approvals": self.approvals,
            "rejections": dict(self.rejections),
            "approval_rate": safe_div(self.approvals, total, 0.0),
            "kill_switches": self.kill_switches.status(),
            "cooldowns": self.cooldowns.active_cooldowns(),
            "suspended_strategies": dict(self.suspended_strategies),
            "allocation": self.allocator.weights(list(self.allocator.performance)),
            "strategy_performance": self.allocator.report(),
            "preservation": self.preservation.stats(),
            "matrices": {
                "strategy_regime": self.matrices.strategy_regime.stats(),
                "symbol_strategy": self.matrices.symbol_strategy.stats(),
            },
        }


def regime_permits_entry(regime: MarketRegime) -> bool:
    """Convenience predicate mirroring the regime gate, for callers upstream."""
    return not regime.blocks_entries
