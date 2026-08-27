"""Dynamic strategy allocation.

Distributes the risk budget across strategies according to how they have
actually performed, rather than treating them as equals forever.

Two constraints make this safe:

* **Bounded.** A strategy's weight is clamped to ``[min_weight, max_weight]``
  relative to equal weight, so no strategy can take over the account no matter
  how good a run it has had. A strategy on a hot streak is often just a strategy
  whose market conditions happened to persist.
* **Evidence-gated.** Weights stay at parity until a strategy has at least
  ``min_trades_for_adjustment`` trades. Reallocating on ten trades is fitting
  noise, and doing so with real money compounds the error.

Allocation influences the risk budget and the consensus weighting. It never
enables or disables a strategy — that is the kill switch's job — and it never
alters a strategy's parameters, which the brief explicitly forbids during live
trading.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.core.config import AllocationConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp, safe_div

log = get_logger(__name__)


@dataclass(slots=True)
class StrategyPerformance:
    """Rolling performance for one strategy, in R-multiples.

    R-multiples rather than currency: they are comparable across position sizes
    and across periods when equity was different.
    """

    trades: int = 0
    wins: int = 0
    r_sum: float = 0.0
    r_squared_sum: float = 0.0
    gross_profit_r: float = 0.0
    gross_loss_r: float = 0.0
    peak_equity_r: float = 0.0
    trough_from_peak_r: float = 0.0
    cumulative_r: float = 0.0

    def record(self, r_multiple: float) -> None:
        self.trades += 1
        if r_multiple > 0:
            self.wins += 1
            self.gross_profit_r += r_multiple
        else:
            self.gross_loss_r += abs(r_multiple)
        self.r_sum += r_multiple
        self.r_squared_sum += r_multiple**2
        self.cumulative_r += r_multiple
        self.peak_equity_r = max(self.peak_equity_r, self.cumulative_r)
        self.trough_from_peak_r = max(
            self.trough_from_peak_r, self.peak_equity_r - self.cumulative_r
        )

    @property
    def win_rate(self) -> float:
        return safe_div(self.wins, self.trades, 0.0)

    @property
    def expectancy_r(self) -> float:
        """Average R per trade. The single most useful number for a strategy."""
        return safe_div(self.r_sum, self.trades, 0.0)

    @property
    def profit_factor(self) -> float:
        if self.gross_loss_r <= 0:
            return float("inf") if self.gross_profit_r > 0 else 0.0
        return self.gross_profit_r / self.gross_loss_r

    @property
    def max_drawdown_r(self) -> float:
        return self.trough_from_peak_r

    @property
    def std_r(self) -> float:
        if self.trades < 2:
            return 0.0
        mean = self.expectancy_r
        variance = max(0.0, self.r_squared_sum / self.trades - mean**2)
        return variance**0.5

    @property
    def sharpe_like(self) -> float:
        """Expectancy divided by dispersion — per trade, not annualised."""
        std = self.std_r
        return safe_div(self.expectancy_r, std, 0.0) if std > 0 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "profit_factor": round(self.profit_factor, 4)
            if self.profit_factor != float("inf")
            else -1.0,
            "max_drawdown_r": round(self.max_drawdown_r, 4),
            "cumulative_r": round(self.cumulative_r, 4),
            "sharpe_like": round(self.sharpe_like, 4),
        }


class StrategyAllocator:
    """Computes bounded, evidence-gated risk weights per strategy."""

    def __init__(self, config: AllocationConfig) -> None:
        self.config = config
        self.performance: dict[str, StrategyPerformance] = {}

    def performance_for(self, strategy: str) -> StrategyPerformance:
        found = self.performance.get(strategy)
        if found is None:
            found = StrategyPerformance()
            self.performance[strategy] = found
        return found

    def record_trade(self, strategy: str, r_multiple: float) -> None:
        self.performance_for(strategy).record(r_multiple)

    # ------------------------------------------------------------------ #
    def weights(self, strategies: list[str]) -> dict[str, float]:
        """Risk multipliers per strategy, centred on 1.0.

        Returns all-1.0 when allocation is disabled or evidence is insufficient.
        """
        cfg = self.config
        if not strategies:
            return {}
        if not cfg.enabled:
            return dict.fromkeys(strategies, 1.0)

        scores: dict[str, float] = {}
        eligible: list[str] = []
        for name in strategies:
            perf = self.performance_for(name)
            if perf.trades < cfg.min_trades_for_adjustment:
                scores[name] = 1.0  # parity until it has earned otherwise
                continue
            eligible.append(name)
            # Expectancy is the driver; dispersion damps a strategy whose good
            # average comes from a few outliers.
            scores[name] = max(0.0, perf.expectancy_r) * (1.0 + perf.sharpe_like)

        if not eligible:
            return dict.fromkeys(strategies, 1.0)

        eligible_total = sum(scores[name] for name in eligible)
        if eligible_total <= 0:
            # Every eligible strategy has non-positive expectancy. Do not try to
            # pick the least bad — hold them all at the floor and let the
            # strategy kill switch handle the genuinely broken ones.
            out = dict.fromkeys(strategies, 1.0)
            for name in eligible:
                out[name] = cfg.min_weight
            return out

        average = eligible_total / len(eligible)
        out: dict[str, float] = {}
        for name in strategies:
            if name not in eligible:
                out[name] = 1.0
                continue
            relative = safe_div(scores[name], average, 1.0)
            out[name] = clamp(relative, cfg.min_weight, cfg.max_weight)

        return out

    def risk_fraction_for(
        self, strategy: str, base_fraction: float, strategies: list[str] | None = None
    ) -> float:
        """Scale the base risk-per-trade by this strategy's allocation weight."""
        names = strategies or list(self.performance) or [strategy]
        if strategy not in names:
            names = [*names, strategy]
        weight = self.weights(names).get(strategy, 1.0)
        return base_fraction * weight

    def report(self) -> dict[str, dict[str, float]]:
        return {name: perf.as_dict() for name, perf in self.performance.items()}


class StrategyKillSwitch:
    """Suspends a strategy whose recent record says it has stopped working.

    Distinct from the account-level kill switches: this disables ONE strategy
    while the rest continue. It is deliberately conservative — a strategy must
    have enough trades to be judged, and the suspension is temporary, because a
    strategy usually stops working because its regime left, and regimes return.
    """

    def __init__(self, config, allocator: StrategyAllocator, clock=None) -> None:
        self.config = config
        self.allocator = allocator
        self._clock = clock

    def _now(self) -> float:
        import time as _time

        return self._clock.now() if self._clock is not None else _time.time()

    def should_disable(self, strategy: str) -> tuple[bool, str]:
        """Should this strategy be suspended? Returns (yes/no, reason)."""
        cfg = self.config
        if not cfg.enabled:
            return False, ""

        perf = self.allocator.performance_for(strategy)
        if perf.trades < cfg.min_trades:
            return False, f"only {perf.trades} trades; {cfg.min_trades} needed to judge"

        if perf.profit_factor < cfg.min_profit_factor:
            return True, (
                f"profit factor {perf.profit_factor:.2f} below "
                f"{cfg.min_profit_factor} over {perf.trades} trades"
            )
        if perf.expectancy_r < cfg.min_expectancy:
            return True, (
                f"expectancy {perf.expectancy_r:.4f}R below "
                f"{cfg.min_expectancy} over {perf.trades} trades"
            )
        return False, "performing within limits"

    def disable_until(self) -> float:
        return self._now() + self.config.disable_seconds
