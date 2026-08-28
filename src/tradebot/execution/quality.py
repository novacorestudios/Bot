"""Execution quality: what we expected versus what we got.

The edge filter decides whether a trade is worth taking by comparing an expected
net edge against a threshold. Every input to that comparison is an estimate —
the spread we will cross, the slippage we will suffer, the fee tier we are on,
the funding we will pay. If those estimates are systematically optimistic, the
filter approves trades that were never profitable, and the resulting losses look
like strategy failure rather than what they are: a measurement error.

This module measures the error. For every order it records the price the
decision was made at, the price actually obtained, and the difference, then
reports the bias per symbol and per order type. That bias is what
:meth:`ExecutionQuality.slippage_adjustment` feeds back into the cost model, so
the edge filter's threshold rises to meet reality rather than the other way
round.

Two deliberate choices:

* **Median, not mean.** One 40-bps fill during a news spike would drag a mean
  estimate for hours. The median describes the fill we should actually expect.
* **A minimum sample before any adjustment.** Three fills are not evidence of a
  bias, and adjusting on them would make the system chase noise.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import Direction

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One order's expected-versus-actual, in fractions of price."""

    symbol: str
    direction: Direction
    order_type: str
    is_entry: bool
    reference_price: float
    fill_price: float
    quantity: float
    expected_cost: float  # what the cost model predicted, as a fraction
    latency_ms: float = 0.0
    at_ms: int = 0

    @property
    def notional(self) -> float:
        return self.fill_price * self.quantity

    @property
    def slippage(self) -> float:
        """Signed, and always in the direction that HURTS.

        Positive means the fill was worse than the reference. Sign is what makes
        this useful: an unsigned magnitude cannot distinguish a market that
        moves against us from one that fills us better than quoted, and the
        second is not a cost to compensate for.
        """
        if self.reference_price <= 0:
            return 0.0
        raw = (self.fill_price - self.reference_price) / self.reference_price
        # Buying higher, or selling lower, than the reference is adverse.
        adverse_sign = 1.0 if self.is_buy else -1.0
        return raw * adverse_sign

    @property
    def is_buy(self) -> bool:
        """An entry LONG buys; an exit LONG sells. Both matter."""
        long_side = self.direction is Direction.LONG
        return long_side if self.is_entry else not long_side

    @property
    def slippage_cost(self) -> float:
        """Slippage expressed in quote currency."""
        return self.slippage * self.notional

    @property
    def cost_error(self) -> float:
        """Actual minus predicted. Positive means we under-estimated the cost."""
        return self.slippage - self.expected_cost


@dataclass(slots=True)
class SymbolQuality:
    """Rolling execution statistics for one symbol."""

    symbol: str
    slippages: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    cost_errors: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    fills: int = 0
    adverse_fills: int = 0

    @property
    def median_slippage(self) -> float:
        return statistics.median(self.slippages) if self.slippages else 0.0

    @property
    def median_cost_error(self) -> float:
        return statistics.median(self.cost_errors) if self.cost_errors else 0.0

    @property
    def worst_slippage(self) -> float:
        return max(self.slippages) if self.slippages else 0.0

    @property
    def adverse_rate(self) -> float:
        return safe_div(self.adverse_fills, self.fills, 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "fills": self.fills,
            "median_slippage_bps": round(self.median_slippage * 10_000, 2),
            "worst_slippage_bps": round(self.worst_slippage * 10_000, 2),
            "median_cost_error_bps": round(self.median_cost_error * 10_000, 2),
            "adverse_rate": round(self.adverse_rate, 3),
        }


class ExecutionQuality:
    """Measures the gap between predicted and realised execution cost."""

    def __init__(
        self,
        min_samples: int = 10,
        max_adjustment: float = 0.002,
        history: int = 500,
    ) -> None:
        #: Below this many fills, no adjustment is made: a handful of fills is
        #: not evidence of a bias, and reacting to them chases noise.
        self.min_samples = min_samples
        #: Ceiling on the correction, so one pathological session cannot make
        #: the cost model so pessimistic that nothing ever trades again.
        self.max_adjustment = max_adjustment

        self._records: deque[ExecutionRecord] = deque(maxlen=history)
        self._symbols: dict[str, SymbolQuality] = {}
        self._by_order_type: dict[str, list[float]] = {}

        self.recorded = 0

    # ------------------------------------------------------------------ #
    def record(self, record: ExecutionRecord) -> None:
        if record.reference_price <= 0 or record.fill_price <= 0:
            return

        self._records.append(record)
        self.recorded += 1

        quality = self._symbols.get(record.symbol)
        if quality is None:
            quality = SymbolQuality(record.symbol)
            self._symbols[record.symbol] = quality

        slippage = record.slippage
        quality.slippages.append(slippage)
        quality.cost_errors.append(record.cost_error)
        quality.fills += 1
        if slippage > 0:
            quality.adverse_fills += 1

        self._by_order_type.setdefault(record.order_type, []).append(slippage)

        if slippage > self.max_adjustment:
            log.warning(
                "execution_quality_poor_fill",
                symbol=record.symbol,
                slippage_bps=round(slippage * 10_000, 2),
                expected_bps=round(record.expected_cost * 10_000, 2),
                order_type=record.order_type,
            )

    # ------------------------------------------------------------------ #
    def slippage_adjustment(self, symbol: str | None = None) -> float:
        """How much to ADD to the cost model's slippage estimate.

        Never negative: an optimistic correction would let the edge filter
        approve trades on the strength of a lucky run of fills. The feedback
        loop is allowed to make the system more careful, never less.
        """
        errors = self._errors_for(symbol)
        if len(errors) < self.min_samples:
            return 0.0
        bias = statistics.median(errors)
        return min(max(0.0, bias), self.max_adjustment)

    def _errors_for(self, symbol: str | None) -> list[float]:
        if symbol is not None:
            quality = self._symbols.get(symbol)
            return list(quality.cost_errors) if quality else []
        return [record.cost_error for record in self._records]

    def expected_slippage(self, symbol: str) -> float:
        """The slippage this symbol actually delivers, once we have evidence."""
        quality = self._symbols.get(symbol)
        if quality is None or quality.fills < self.min_samples:
            return 0.0
        return max(0.0, quality.median_slippage)

    def is_calibrated(self, symbol: str | None = None) -> bool:
        return len(self._errors_for(symbol)) >= self.min_samples

    # ------------------------------------------------------------------ #
    def worst_symbols(self, limit: int = 5) -> list[dict[str, Any]]:
        """Where execution is costing the most — candidates for exclusion."""
        ranked = sorted(
            (q for q in self._symbols.values() if q.fills >= 3),
            key=lambda q: q.median_slippage,
            reverse=True,
        )
        return [q.as_dict() for q in ranked[:limit]]

    def stats(self) -> dict[str, Any]:
        all_slippage = [record.slippage for record in self._records]
        all_errors = [record.cost_error for record in self._records]
        return {
            "recorded": self.recorded,
            "sample": len(self._records),
            "calibrated": self.is_calibrated(),
            "median_slippage_bps": round(statistics.median(all_slippage) * 10_000, 2)
            if all_slippage
            else 0.0,
            "median_cost_error_bps": round(statistics.median(all_errors) * 10_000, 2)
            if all_errors
            else 0.0,
            "adjustment_bps": round(self.slippage_adjustment() * 10_000, 2),
            "by_order_type": {
                name: round(statistics.median(values) * 10_000, 2)
                for name, values in self._by_order_type.items()
                if values
            },
            "symbols_tracked": len(self._symbols),
        }

    def report(self, limit: int = 25) -> list[dict[str, Any]]:
        ranked = sorted(self._symbols.values(), key=lambda q: q.median_slippage, reverse=True)
        return [q.as_dict() for q in ranked[:limit]]
