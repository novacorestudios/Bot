"""Position sizing and leverage.

The only place a position size is decided. No strategy, no AI layer and no
dashboard action can produce a size — they can only produce an opportunity that
this module then sizes or refuses.

The core calculation is simple and non-negotiable:

    quantity = (equity × risk_fraction) / stop_distance

Risk is defined by the **distance to the stop**, never by the notional and never
by the leverage. That is what makes 0.5 % per trade mean the same thing on a
tight-stop scalp and a wide-stop swing.

Leverage does not change the risk. It changes only how much margin the position
consumes. A 600 USDT position with a 0.5 % stop risks 3 USDT at 1x and 3 USDT at
10x — the difference is that at 10x the liquidation price sits close enough that
the exchange may close the position before the stop does. That is the real
danger of leverage on a small account, and it is why ``liquidation_distance``
below is a hard gate rather than a warning.

Small accounts hit a specific wall. With 75 USDT and 0.5 % risk (0.375 USDT), a
0.5 % stop implies a 75 USDT position — but many symbols require a 5-20 USDT
minimum notional with step sizes that may not divide evenly. When the correctly
sized position cannot be represented, the answer is to **skip the trade**, never
to round up into more risk than the budget allows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tradebot.core.config import RiskConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp, round_quantity, safe_div
from tradebot.core.types import Direction, RejectionReason, SymbolInfo
from tradebot.exchange.binance.filters import min_quantity_for_notional

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SizingResult:
    """A computed size, or a reason there cannot be one."""

    ok: bool
    quantity: float = 0.0
    notional: float = 0.0
    leverage: int = 1
    risk_amount: float = 0.0
    margin_required: float = 0.0
    liquidation_price: float = 0.0
    liquidation_distance_multiple: float = 0.0
    reason: RejectionReason | None = None
    detail: str = ""
    checks: dict[str, float] = field(default_factory=dict)
    #: Observational only (V3.2 diagnostics). The two inputs that decide whether
    #: a size is representable at all, carried out so a diagnostic can see the
    #: distribution rather than only the verdict. Nothing reads them to decide.
    stop_distance: float | None = None
    raw_quantity: float | None = None

    @classmethod
    def reject(cls, reason: RejectionReason, detail: str, **checks: float) -> SizingResult:
        return cls(ok=False, reason=reason, detail=detail, checks=checks)


class PositionSizer:
    """Turns (equity, risk fraction, stop distance) into a valid quantity."""

    def __init__(self, config: RiskConfig, max_min_notional_ratio: float = 0.5) -> None:
        self.config = config
        self.max_min_notional_ratio = max_min_notional_ratio

    # ------------------------------------------------------------------ #
    def required_leverage(self, notional: float, available_margin: float) -> int:
        """Smallest leverage that makes this notional affordable.

        Leverage is derived from need, not chosen for ambition. If the position
        fits within available margin at 1x, 1x is used.
        """
        if available_margin <= 0:
            return self.config.max_leverage
        needed = safe_div(notional, available_margin, float(self.config.max_leverage))
        return max(self.config.min_leverage, int(needed) + (1 if needed % 1 else 0))

    def volatility_adjusted_max_leverage(self, volatility: float) -> int:
        """Reduce the leverage ceiling as volatility rises.

        A position that is comfortable at 1 % daily range is not comfortable at
        5 %, because the liquidation price is a fixed fraction away and the
        market reaches it sooner.
        """
        cfg = self.config
        if volatility <= 0:
            return cfg.max_leverage
        threshold = cfg.leverage_volatility_threshold
        if volatility <= threshold:
            return cfg.max_leverage
        # Halve the ceiling for each doubling of volatility past the threshold.
        ratio = volatility / threshold
        reduced = int(cfg.max_leverage / ratio)
        return max(cfg.min_leverage, min(cfg.max_leverage, reduced))

    def estimate_liquidation(
        self,
        entry: float,
        direction: Direction,
        leverage: int,
        maintenance_margin_rate: float = 0.004,
    ) -> float:
        """Approximate isolated-margin liquidation price.

        Isolated margin, ignoring fees:
            long  ≈ entry × (1 - 1/L + mmr)
            short ≈ entry × (1 + 1/L - mmr)

        This is an ESTIMATE. Binance's exact formula depends on the notional
        bracket's maintenance margin and the maintenance amount. The engine uses
        it conservatively — to reject positions whose liquidation sits too close
        to the stop — and reads the exchange's own liquidationPrice once the
        position exists.
        """
        if leverage <= 0 or entry <= 0:
            return 0.0
        buffer = (1.0 / leverage) - maintenance_margin_rate
        if direction is Direction.LONG:
            return max(0.0, entry * (1.0 - buffer))
        return entry * (1.0 + buffer)

    # ------------------------------------------------------------------ #
    def size(
        self,
        equity: float,
        risk_fraction: float,
        entry_price: float,
        stop_price: float,
        direction: Direction,
        symbol_info: SymbolInfo,
        available_margin: float | None = None,
        total_margin_available: float | None = None,
        volatility: float = 0.0,
        max_notional: float | None = None,
    ) -> SizingResult:
        """Compute the position size, or refuse with a reason."""
        cfg = self.config

        if equity <= 0:
            return SizingResult.reject(RejectionReason.INSUFFICIENT_BALANCE, f"equity is {equity}")
        if entry_price <= 0:
            return SizingResult.reject(
                RejectionReason.INVALID_STOP, f"entry price is {entry_price}"
            )

        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return SizingResult.reject(
                RejectionReason.INVALID_STOP,
                "stop distance is zero; risk would be undefined",
            )

        # Direction sanity: a stop on the wrong side is a bug upstream, and
        # sizing it would produce a position that can never be protected.
        if direction is Direction.LONG and stop_price >= entry_price:
            return SizingResult.reject(RejectionReason.INVALID_STOP, "LONG stop is not below entry")
        if direction is Direction.SHORT and stop_price <= entry_price:
            return SizingResult.reject(
                RejectionReason.INVALID_STOP, "SHORT stop is not above entry"
            )

        risk_fraction = clamp(risk_fraction, cfg.min_risk_per_trade, cfg.max_risk_per_trade)
        risk_amount = equity * risk_fraction

        # -- the core calculation ------------------------------------------ #
        raw_quantity = risk_amount / stop_distance

        def observed(result: SizingResult) -> SizingResult:
            """Attach the two inputs that decided representability.

            `replace` copies every other field of the frozen dataclass verbatim,
            so this annotates the decision without being able to alter it. The
            returns above this point leave both `None`, which is the honest
            record for a candidate rejected before a quantity ever existed.
            """
            return replace(result, stop_distance=stop_distance, raw_quantity=raw_quantity)

        # -- exposure ceiling ----------------------------------------------- #
        notional_cap = equity * cfg.max_symbol_exposure
        if max_notional is not None:
            notional_cap = min(notional_cap, max_notional)
        if raw_quantity * entry_price > notional_cap:
            raw_quantity = notional_cap / entry_price

        quantity = round_quantity(raw_quantity, symbol_info.step_size)
        if quantity <= 0:
            return observed(
                SizingResult.reject(
                    RejectionReason.SIZE_BELOW_MINIMUM,
                    f"risk-correct quantity {raw_quantity:.10g} rounds to zero at "
                    f"step size {symbol_info.step_size}",
                    raw_quantity=raw_quantity,
                )
            )

        if symbol_info.min_qty > 0 and quantity < symbol_info.min_qty:
            return observed(
                SizingResult.reject(
                    RejectionReason.SIZE_BELOW_MINIMUM,
                    f"quantity {quantity} below the symbol minimum "
                    f"{symbol_info.min_qty}; sizing up would exceed the risk budget",
                    quantity=quantity,
                    min_qty=symbol_info.min_qty,
                )
            )

        notional = quantity * entry_price

        # -- minimum notional: the small-account wall ------------------------ #
        if symbol_info.min_notional > 0 and notional < symbol_info.min_notional:
            required = min_quantity_for_notional(symbol_info, entry_price)
            implied_risk = required * stop_distance
            return observed(
                SizingResult.reject(
                    RejectionReason.NOTIONAL_BELOW_MINIMUM,
                    f"risk-correct notional {notional:.4f} is below the symbol "
                    f"minimum {symbol_info.min_notional}. Meeting it would require "
                    f"{required:.10g} units, risking {implied_risk:.4f} "
                    f"({safe_div(implied_risk, equity, 0) * 100:.2f}% of equity) "
                    f"instead of {risk_amount:.4f}. Skipping rather than oversizing.",
                    notional=notional,
                    min_notional=symbol_info.min_notional,
                    implied_risk=implied_risk,
                    budgeted_risk=risk_amount,
                )
            )

        actual_risk = quantity * stop_distance

        # -- leverage -------------------------------------------------------- #
        balance_margin_budget = (
            available_margin if available_margin is not None else equity * cfg.max_margin_usage
        )
        total_margin_budget = (
            total_margin_available
            if total_margin_available is not None
            else cfg.max_total_allocated_margin
        )
        margin_budget = max(
            0.0,
            min(balance_margin_budget, cfg.max_margin_per_trade, total_margin_budget),
        )
        needed_leverage = self.required_leverage(notional, margin_budget)
        volatility_ceiling = self.volatility_adjusted_max_leverage(volatility)
        absolute_leverage_ceiling = min(cfg.max_leverage, symbol_info.max_leverage)
        required_margin_at_absolute_ceiling = notional / max(1, absolute_leverage_ceiling)

        if required_margin_at_absolute_ceiling > cfg.max_margin_per_trade:
            return observed(
                SizingResult.reject(
                    RejectionReason.PER_TRADE_MARGIN_LIMIT,
                    f"risk-correct position needs at least "
                    f"{required_margin_at_absolute_ceiling:.4f} margin at "
                    f"{absolute_leverage_ceiling}x, above the per-trade cap "
                    f"{cfg.max_margin_per_trade:.4f}",
                    margin_required=required_margin_at_absolute_ceiling,
                    margin_cap=cfg.max_margin_per_trade,
                    leverage_ceiling=float(absolute_leverage_ceiling),
                )
            )
        if (
            total_margin_available is not None
            and required_margin_at_absolute_ceiling > total_margin_available
        ):
            return observed(
                SizingResult.reject(
                    RejectionReason.TOTAL_MARGIN_LIMIT,
                    f"risk-correct position needs at least "
                    f"{required_margin_at_absolute_ceiling:.4f} margin but only "
                    f"{max(0.0, total_margin_available):.4f} remains under the total cap",
                    margin_required=required_margin_at_absolute_ceiling,
                    total_margin_available=max(0.0, total_margin_available),
                    total_margin_cap=cfg.max_total_allocated_margin,
                )
            )

        leverage = min(needed_leverage, volatility_ceiling, absolute_leverage_ceiling)
        leverage = max(cfg.min_leverage, leverage)

        if needed_leverage > volatility_ceiling:
            return observed(
                SizingResult.reject(
                    RejectionReason.LEVERAGE_LIMIT,
                    f"position needs {needed_leverage}x but volatility "
                    f"{volatility:.4f} caps leverage at {volatility_ceiling}x",
                    needed_leverage=needed_leverage,
                    volatility_ceiling=volatility_ceiling,
                    volatility=volatility,
                )
            )
        if needed_leverage > symbol_info.max_leverage:
            return observed(
                SizingResult.reject(
                    RejectionReason.LEVERAGE_LIMIT,
                    f"position needs {needed_leverage}x but the symbol allows at "
                    f"most {symbol_info.max_leverage}x",
                    needed_leverage=needed_leverage,
                    symbol_max=symbol_info.max_leverage,
                )
            )

        margin_required = notional / leverage
        if margin_required > cfg.max_margin_per_trade:
            return observed(
                SizingResult.reject(
                    RejectionReason.PER_TRADE_MARGIN_LIMIT,
                    f"margin {margin_required:.4f} exceeds the per-trade cap "
                    f"{cfg.max_margin_per_trade:.4f}",
                    margin_required=margin_required,
                    margin_cap=cfg.max_margin_per_trade,
                )
            )
        if total_margin_available is not None and margin_required > total_margin_available:
            return observed(
                SizingResult.reject(
                    RejectionReason.TOTAL_MARGIN_LIMIT,
                    f"margin {margin_required:.4f} exceeds the "
                    f"{max(0.0, total_margin_available):.4f} remaining total allocation",
                    margin_required=margin_required,
                    total_margin_available=max(0.0, total_margin_available),
                    total_margin_cap=cfg.max_total_allocated_margin,
                )
            )
        if margin_required > balance_margin_budget:
            return observed(
                SizingResult.reject(
                    RejectionReason.MARGIN_LIMIT,
                    f"margin {margin_required:.4f} exceeds the available "
                    f"{max(0.0, balance_margin_budget):.4f}",
                    margin_required=margin_required,
                    margin_budget=max(0.0, balance_margin_budget),
                )
            )

        # -- liquidation must sit well beyond the stop ----------------------- #
        maintenance_rate = self._maintenance_rate(symbol_info, notional)
        liquidation = self.estimate_liquidation(entry_price, direction, leverage, maintenance_rate)
        liquidation_distance = abs(entry_price - liquidation)
        distance_multiple = safe_div(liquidation_distance, stop_distance, 0.0)

        if distance_multiple < cfg.min_liquidation_distance_multiple:
            return observed(
                SizingResult.reject(
                    RejectionReason.LIQUIDATION_TOO_CLOSE,
                    f"at {leverage}x, liquidation sits {distance_multiple:.2f}x the "
                    f"stop distance away; {cfg.min_liquidation_distance_multiple}x "
                    f"is required. The exchange would close this position before the "
                    f"stop could.",
                    leverage=float(leverage),
                    distance_multiple=distance_multiple,
                    liquidation_price=liquidation,
                )
            )

        return observed(
            SizingResult(
                ok=True,
                quantity=quantity,
                notional=notional,
                leverage=leverage,
                risk_amount=actual_risk,
                margin_required=margin_required,
                liquidation_price=liquidation,
                liquidation_distance_multiple=distance_multiple,
                checks={
                    "stop_distance": stop_distance,
                    "stop_distance_pct": safe_div(stop_distance, entry_price, 0.0),
                    "risk_fraction_used": safe_div(actual_risk, equity, 0.0),
                    "budgeted_risk": risk_amount,
                    "needed_leverage": float(needed_leverage),
                    "margin_required": margin_required,
                    "margin_per_trade_cap": cfg.max_margin_per_trade,
                    "total_margin_available": max(0.0, total_margin_budget),
                    "volatility_ceiling": float(volatility_ceiling),
                    "maintenance_rate": maintenance_rate,
                },
            )
        )

    @staticmethod
    def _maintenance_rate(symbol_info: SymbolInfo, notional: float) -> float:
        """Maintenance margin rate for this notional, from the symbol's brackets."""
        bracket = symbol_info.bracket_for_notional(notional)
        if bracket is not None and bracket.maint_margin_ratio > 0:
            return bracket.maint_margin_ratio
        return 0.004  # Binance's typical lowest tier
