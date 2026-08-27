"""Portfolio risk accounting.

Tracks what is currently at risk across all open positions, so a new trade can
be judged against the whole book rather than in isolation. Every per-trade check
can pass while the portfolio as a whole is over-committed; this is where that is
caught.

"Open risk" means **distance to stop × quantity**, summed. Not notional, not
margin. A 1000 USDT position with a stop 0.2 % away risks 2 USDT; a 100 USDT
position with a stop 5 % away risks 5 USDT. The second is the larger risk despite
being a tenth of the notional, and only stop-distance accounting sees that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tradebot.core.config import RiskConfig
from tradebot.core.mathutil import safe_div
from tradebot.core.types import Direction, Position


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """A complete picture of current exposure. Everything the brief asks for."""

    equity: float
    available_balance: float
    total_open_risk: float
    total_long_exposure: float
    total_short_exposure: float
    total_exposure: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl_today: float
    position_count: int
    symbols: tuple[str, ...] = ()
    per_symbol_exposure: dict[str, float] = field(default_factory=dict)
    unprotected_positions: tuple[str, ...] = ()

    @property
    def open_risk_fraction(self) -> float:
        return safe_div(self.total_open_risk, self.equity, 0.0)

    @property
    def exposure_ratio(self) -> float:
        return safe_div(self.total_exposure, self.equity, 0.0)

    @property
    def long_ratio(self) -> float:
        return safe_div(self.total_long_exposure, self.equity, 0.0)

    @property
    def short_ratio(self) -> float:
        return safe_div(self.total_short_exposure, self.equity, 0.0)

    @property
    def margin_usage(self) -> float:
        return safe_div(self.margin_used, self.equity, 0.0)

    @property
    def net_direction_exposure(self) -> float:
        """Signed net exposure. Near zero means the book is directionally flat."""
        return self.total_long_exposure - self.total_short_exposure

    def as_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "equity": round(self.equity, 4),
            "available_balance": round(self.available_balance, 4),
            "total_open_risk": round(self.total_open_risk, 4),
            "open_risk_pct": round(self.open_risk_fraction * 100, 3),
            "total_long_exposure": round(self.total_long_exposure, 2),
            "total_short_exposure": round(self.total_short_exposure, 2),
            "total_exposure": round(self.total_exposure, 2),
            "exposure_ratio": round(self.exposure_ratio, 3),
            "margin_used": round(self.margin_used, 4),
            "margin_usage": round(self.margin_usage, 3),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "realized_pnl_today": round(self.realized_pnl_today, 4),
            "position_count": self.position_count,
            "symbols": list(self.symbols),
            "unprotected_positions": list(self.unprotected_positions),
        }


class PortfolioTracker:
    """Computes portfolio state from open positions and account balances."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def state(
        self,
        equity: float,
        available_balance: float,
        positions: dict[str, Position],
        prices: dict[str, float],
        unrealized_pnl: float = 0.0,
        realized_pnl_today: float = 0.0,
    ) -> PortfolioState:
        long_exposure = 0.0
        short_exposure = 0.0
        open_risk = 0.0
        margin_used = 0.0
        per_symbol: dict[str, float] = {}
        unprotected: list[str] = []
        computed_unrealized = 0.0

        for symbol, position in positions.items():
            price = prices.get(symbol, position.entry_price)
            notional = position.quantity * price

            if position.direction is Direction.LONG:
                long_exposure += notional
            else:
                short_exposure += notional

            per_symbol[symbol] = per_symbol.get(symbol, 0.0) + notional
            margin_used += position.margin(price)
            computed_unrealized += position.unrealized_pnl(price)

            # Open risk = what this position loses if its stop is hit from here.
            if position.stop_loss > 0:
                # Once price has moved past the stop in our favour, the position
                # can no longer lose money at the stop — risk is negative, but
                # we floor it at zero rather than letting a winner offset a
                # loser's risk budget.
                risk_per_unit = (
                    price - position.stop_loss
                    if position.direction is Direction.LONG
                    else position.stop_loss - price
                )
                open_risk += max(0.0, risk_per_unit * position.quantity)
            else:
                # No stop: the entire position is at risk as far as we know.
                unprotected.append(symbol)
                open_risk += notional

        return PortfolioState(
            equity=equity,
            available_balance=available_balance,
            total_open_risk=open_risk,
            total_long_exposure=long_exposure,
            total_short_exposure=short_exposure,
            total_exposure=long_exposure + short_exposure,
            margin_used=margin_used,
            unrealized_pnl=unrealized_pnl or computed_unrealized,
            realized_pnl_today=realized_pnl_today,
            position_count=len(positions),
            symbols=tuple(positions),
            per_symbol_exposure=per_symbol,
            unprotected_positions=tuple(unprotected),
        )

    # ------------------------------------------------------------------ #
    def would_breach(
        self,
        state: PortfolioState,
        symbol: str,
        direction: Direction,
        notional: float,
        risk_amount: float,
        margin: float,
    ) -> tuple[bool, str, str]:
        """Check a prospective position against every portfolio limit.

        Returns (breached, limit_name, detail).
        """
        cfg = self.config
        equity = state.equity
        if equity <= 0:
            return True, "EQUITY", "equity is not positive"

        if state.position_count >= cfg.max_concurrent_positions:
            return (
                True,
                "MAX_POSITIONS",
                f"{state.position_count} positions already open "
                f"(limit {cfg.max_concurrent_positions})",
            )

        projected_risk = state.total_open_risk + risk_amount
        if projected_risk > equity * cfg.max_total_risk:
            return (
                True,
                "TOTAL_RISK",
                f"open risk would be {projected_risk:.4f} "
                f"({safe_div(projected_risk, equity, 0) * 100:.2f}% of equity), "
                f"above the {cfg.max_total_risk * 100:.2f}% budget",
            )

        symbol_exposure = state.per_symbol_exposure.get(symbol, 0.0) + notional
        if symbol_exposure > equity * cfg.max_symbol_exposure:
            return (
                True,
                "SYMBOL_EXPOSURE",
                f"{symbol} exposure would be {symbol_exposure:.2f} "
                f"({safe_div(symbol_exposure, equity, 0):.2f}x equity, limit "
                f"{cfg.max_symbol_exposure}x)",
            )

        if direction is Direction.LONG:
            directional = state.total_long_exposure + notional
        else:
            directional = state.total_short_exposure + notional
        if directional > equity * cfg.max_direction_exposure:
            return (
                True,
                "DIRECTION_EXPOSURE",
                f"{direction.value} exposure would be {directional:.2f} "
                f"({safe_div(directional, equity, 0):.2f}x equity, limit "
                f"{cfg.max_direction_exposure}x)",
            )

        total = state.total_exposure + notional
        if total > equity * cfg.max_total_exposure:
            return (
                True,
                "TOTAL_EXPOSURE",
                f"total exposure would be {total:.2f} "
                f"({safe_div(total, equity, 0):.2f}x equity, limit "
                f"{cfg.max_total_exposure}x)",
            )

        projected_margin = state.margin_used + margin
        if projected_margin > equity * cfg.max_margin_usage:
            return (
                True,
                "MARGIN_USAGE",
                f"margin usage would be "
                f"{safe_div(projected_margin, equity, 0) * 100:.1f}%, above "
                f"{cfg.max_margin_usage * 100:.0f}%",
            )

        if margin > state.available_balance:
            return (
                True,
                "AVAILABLE_BALANCE",
                f"margin {margin:.4f} exceeds available balance {state.available_balance:.4f}",
            )

        return False, "", "within all portfolio limits"

    def remaining_risk_budget(self, state: PortfolioState) -> float:
        """How much more risk the portfolio may take on, in quote currency."""
        allowed = state.equity * self.config.max_total_risk
        return max(0.0, allowed - state.total_open_risk)
