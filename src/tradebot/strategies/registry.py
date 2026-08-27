"""Strategy registry.

Builds the enabled strategy set from configuration, and evaluates the ones the
current regime permits.

Two guarantees enforced here:

* **Regime gating is not advisory.** A strategy absent from the regime's weight
  map is not evaluated at all — not evaluated and then discounted. In PANIC the
  map is empty, so nothing runs.
* **Isolation.** One strategy raising cannot prevent the others from being
  evaluated, and cannot abort the cycle. Its failure is counted, and a strategy
  that keeps failing is disabled by the health monitor.
"""

from __future__ import annotations

from typing import Any

from tradebot.core.config import TunableConfig
from tradebot.core.logging import get_logger
from tradebot.core.types import Direction, MarketRegime, Signal
from tradebot.strategies.base import MarketView, Strategy
from tradebot.strategies.breakout import BreakoutStrategy
from tradebot.strategies.mean_reversion import MeanReversionStrategy
from tradebot.strategies.momentum import MomentumStrategy
from tradebot.strategies.support_resistance import SupportResistanceStrategy
from tradebot.strategies.trend_following import TrendFollowingStrategy
from tradebot.strategies.volatility_expansion import VolatilityExpansionStrategy
from tradebot.strategies.volume_spike import VolumeSpikeStrategy
from tradebot.strategies.vwap import VWAPStrategy

log = get_logger(__name__)

#: Every strategy known to the engine, keyed by its config name.
STRATEGY_CLASSES: dict[str, type[Strategy]] = {
    MomentumStrategy.name: MomentumStrategy,
    TrendFollowingStrategy.name: TrendFollowingStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    VolumeSpikeStrategy.name: VolumeSpikeStrategy,
    VolatilityExpansionStrategy.name: VolatilityExpansionStrategy,
    VWAPStrategy.name: VWAPStrategy,
    SupportResistanceStrategy.name: SupportResistanceStrategy,
}


class StrategyRegistry:
    """Holds the live strategy instances and runs the regime-permitted subset."""

    def __init__(
        self, strategies: dict[str, Strategy], regime_weights: dict[str, dict[str, float]]
    ) -> None:
        self.strategies = strategies
        self.regime_weights = regime_weights
        #: Strategies suspended by the strategy kill switch, with expiry times.
        self.disabled_until: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: TunableConfig) -> StrategyRegistry:
        strategies: dict[str, Strategy] = {}
        for name, params in config.strategies.items():
            factory = STRATEGY_CLASSES.get(name)
            if factory is None:
                log.warning(
                    "unknown_strategy_in_config", strategy=name, known=sorted(STRATEGY_CLASSES)
                )
                continue
            if not params.get("enabled", True):
                log.info("strategy_disabled_by_config", strategy=name)
                continue
            strategies[name] = factory(params, config.stops, config.targets)

        missing = set(STRATEGY_CLASSES) - set(config.strategies)
        if missing:
            log.warning("strategies_absent_from_config", missing=sorted(missing))

        log.info("strategies_loaded", enabled=sorted(strategies))
        return cls(strategies, config.regime.strategy_weights)

    # ------------------------------------------------------------------ #
    def weights_for(self, regime: MarketRegime) -> dict[str, float]:
        """Strategy weights permitted in this regime, restricted to loaded ones."""
        configured = self.regime_weights.get(regime.value, {})
        return {name: weight for name, weight in configured.items() if name in self.strategies}

    def active(self, regime: MarketRegime, now: float = 0.0) -> dict[str, float]:
        """Permitted strategies that are not currently suspended."""
        return {
            name: weight
            for name, weight in self.weights_for(regime).items()
            if not self.is_disabled(name, now)
        }

    def is_disabled(self, name: str, now: float = 0.0) -> bool:
        until = self.disabled_until.get(name)
        if until is None:
            return False
        if now and now >= until:
            del self.disabled_until[name]
            return False
        return True

    def disable(self, name: str, until: float, reason: str = "") -> None:
        """Suspend a strategy — used by the strategy kill switch."""
        self.disabled_until[name] = until
        log.warning("strategy_suspended", strategy=name, until=until, reason=reason)

    def enable(self, name: str) -> None:
        if self.disabled_until.pop(name, None) is not None:
            log.info("strategy_resumed", strategy=name)

    # ------------------------------------------------------------------ #
    def evaluate(self, view: MarketView, now: float = 0.0) -> tuple[list[Signal], dict[str, float]]:
        """Run the permitted strategies. Returns (signals, weights).

        Includes WAIT signals in the result: the aggregator needs to know that a
        strategy considered the market and declined, which is different from the
        strategy not having run at all, and the audit log records both.
        """
        weights = self.active(view.regime, now)
        if not weights:
            return [], {}

        signals: list[Signal] = []
        for name, _weight in weights.items():
            strategy = self.strategies[name]
            # Strategy.generate() already contains its own error isolation.
            signals.append(strategy.generate(view))
        return signals, weights

    def actionable(
        self, view: MarketView, now: float = 0.0
    ) -> tuple[list[Signal], dict[str, float]]:
        """Only the signals with a direction."""
        signals, weights = self.evaluate(view, now)
        return [s for s in signals if s.direction is not Direction.WAIT], weights

    def stats(self) -> dict[str, Any]:
        return {
            "loaded": sorted(self.strategies),
            "suspended": sorted(self.disabled_until),
            "signals_emitted": {
                name: strategy.signals_emitted for name, strategy in self.strategies.items()
            },
            "errors": {
                name: strategy.errors
                for name, strategy in self.strategies.items()
                if strategy.errors
            },
        }
