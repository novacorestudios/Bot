"""Exit conditions the exchange cannot evaluate for us.

A resting stop and a resting take-profit cover the two price levels. Everything
else — the clock, the regime, a signal that has turned against us, an edge that
has evaporated — has to be evaluated here, on every monitor cycle.

Three of these were configurable in V1 and implemented in none of it:
``exit_on_signal_flip`` and ``exit_on_negative_edge`` were read from YAML and
never consulted (AUDIT_REPORT.md M-8). A flag that does nothing is worse than a
missing feature, because it reads as a guarantee.

The fourth condition here is the **local safety net**. Protective orders live on
the exchange, which is right — they survive a crashed bot. But they can be
cancelled by hand, rejected at placement, or lost in a reconciliation gap, and a
position whose stop has quietly gone is the single most expensive failure this
system can have. So the engine also checks the price against the stop itself and
closes at market when the exchange evidently did not.

Ordering matters and is deliberate, worst-first: a breached stop is acted on
before a time limit, and both before a merely deteriorating thesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradebot.core.config import TunableConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import Direction, ExitReason

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExitDecision:
    """Whether to close, and why."""

    reason: ExitReason | None
    detail: str = ""
    urgent: bool = False

    @property
    def should_exit(self) -> bool:
        return self.reason is not None


HOLD = ExitDecision(reason=None)


@dataclass(frozen=True, slots=True)
class ExitContext:
    """Everything the exit rules read. No side effects, no I/O."""

    price: float
    now_ms: int
    #: Consensus direction for this symbol right now, if we have one.
    signal_direction: Direction | None = None
    signal_confidence: float = 0.0
    #: True when the regime for this symbol now blocks entries.
    regime_blocks: bool = False
    #: Expected net edge of CONTINUING to hold, per unit of notional.
    holding_edge: float | None = None
    #: True when we believe the exchange no longer holds a protective stop.
    stop_order_missing: bool = False


class ExitEvaluator:
    """Applies the configured exit rules to one position.

    Pure and synchronous on purpose: the decision is a function of the position
    and the context, so it can be tested exhaustively without an event loop or a
    fake exchange.
    """

    def __init__(self, config: TunableConfig) -> None:
        self.config = config
        self.counts: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def evaluate(self, position: Any, context: ExitContext) -> ExitDecision:
        """Worst-first. The first matching rule wins."""
        for decision in (
            self._protective_level_breached(position, context),
            self._time_limit(position, context),
            self._regime_change(context),
            self._signal_flip(position, context),
            self._negative_edge(context),
        ):
            # `should_exit` is exactly "reason is not None", but narrowing on
            # the attribute keeps the type checker satisfied without an assert
            # — which -O strips, turning this into an AttributeError in
            # production and nowhere else.
            reason = decision.reason
            if reason is not None:
                self.counts[reason.value] = self.counts.get(reason.value, 0) + 1
                return decision
        return HOLD

    # ------------------------------------------------------------------ #
    def _protective_level_breached(self, position: Any, context: ExitContext) -> ExitDecision:
        """The local safety net.

        If price has traded through the stop and we are still holding, the
        exchange's protection did not do its job — it was cancelled, never
        placed, or lost. Closing at market here is worse than the stop price we
        wanted, and far better than an unbounded loss.
        """
        price = context.price
        if price <= 0:
            return HOLD

        sign = position.direction.sign
        stop = position.stop_loss
        if stop > 0 and (price - stop) * sign <= 0:
            log.critical(
                "stop_breached_locally",
                symbol=position.symbol,
                price=price,
                stop=stop,
                message="the exchange stop did not fire; closing at market",
            )
            return ExitDecision(
                ExitReason.STOP_LOSS,
                f"price {price} is through the stop {stop} and we are still open",
                urgent=True,
            )

        target = position.take_profit
        if target > 0 and (target - price) * sign <= 0:
            return ExitDecision(
                ExitReason.TAKE_PROFIT,
                f"price {price} reached the target {target}",
                urgent=True,
            )

        if context.stop_order_missing:
            return ExitDecision(
                ExitReason.RISK_EVENT,
                "no protective stop is resting on the exchange",
                urgent=True,
            )
        return HOLD

    def _time_limit(self, position: Any, context: ExitContext) -> ExitDecision:
        """The hard cap. A scalp that has not worked in an hour is not a scalp."""
        limit = self.config.trade.max_duration_sec
        held = position.duration_sec(context.now_ms)
        if held >= limit:
            return ExitDecision(
                ExitReason.TIME_LIMIT,
                f"held {held:.0f}s, at the {limit}s cap",
            )
        return HOLD

    def _regime_change(self, context: ExitContext) -> ExitDecision:
        if not self.config.trade.exit_on_regime_change:
            return HOLD
        if context.regime_blocks:
            return ExitDecision(
                ExitReason.REGIME_CHANGE,
                "the regime no longer permits this position",
            )
        return HOLD

    def _signal_flip(self, position: Any, context: ExitContext) -> ExitDecision:
        """The consensus that opened this position now points the other way.

        Only a genuine reversal counts: WAIT is the absence of an opinion, not
        an opinion against us, and closing on every lull would churn the account
        into its own fees.
        """
        if not self.config.trade.exit_on_signal_flip:
            return HOLD
        direction = context.signal_direction
        if direction is None or direction is Direction.WAIT:
            return HOLD
        if direction is position.direction:
            return HOLD
        if context.signal_confidence < self.config.aggregator.min_signal_confidence:
            return HOLD
        return ExitDecision(
            ExitReason.SIGNAL_FLIP,
            f"consensus flipped to {direction.value} "
            f"at {context.signal_confidence:.0f}% confidence",
        )

    def _negative_edge(self, context: ExitContext) -> ExitDecision:
        """Continuing to hold no longer has positive expected value.

        This is about the edge of the REMAINING trade, not the one we entered:
        the costs already paid are sunk, and what matters is whether the
        distance still to run, against the risk still being carried, is worth
        the exit cost.
        """
        if not self.config.trade.exit_on_negative_edge:
            return HOLD
        edge = context.holding_edge
        if edge is None or edge >= 0:
            return HOLD
        return ExitDecision(
            ExitReason.NEGATIVE_EDGE,
            f"expected net edge of holding is {edge * 100:.4f}%",
        )

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, int]:
        return dict(self.counts)


def holding_edge(
    position: Any,
    price: float,
    win_probability: float,
    round_trip_cost: float,
) -> float:
    """Expected net edge, per unit of notional, of continuing to hold.

    Both legs are measured from the CURRENT price, because that is what is
    actually still at stake. Everything up to now is realised whether we hold or
    not, so including it would keep a losing position open on the strength of
    the loss it has already taken — the sunk-cost fallacy, expressed in code.
    """
    if price <= 0:
        return 0.0

    sign = position.direction.sign
    remaining_win = safe_div((position.take_profit - price) * sign, price, 0.0)
    remaining_loss = safe_div((price - position.stop_loss) * sign, price, 0.0)

    # Past the target or through the stop: the level rules handle it, and the
    # arithmetic below would be meaningless with a negative distance.
    if remaining_win <= 0 or remaining_loss <= 0:
        return 0.0

    p = min(max(win_probability, 0.0), 1.0)
    expected_gross = p * remaining_win - (1.0 - p) * remaining_loss
    return expected_gross - round_trip_cost
