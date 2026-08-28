"""Performance matrices: which strategy works where, and on what.

The strategy allocator already tracks each strategy's overall performance. That
is not enough to answer the question that actually matters for a multi-strategy
system: *a strategy that loses money overall may be excellent in one regime and
terrible in another, and the aggregate hides both facts.* Mean reversion in a
strong trend and momentum in chop are not weak strategies; they are strategies
being run in the wrong conditions.

Two matrices:

* **strategy × regime** — should this strategy run at all in this regime?
* **symbol × strategy** — some strategies simply do not work on some symbols,
  usually for microstructure reasons (a wide book, a thin tape, a funding regime
  that eats the edge).

Both are *evidence*, not authority. They produce a multiplier the allocator can
apply and a report an operator can read; neither can place an order, and neither
can raise a weight above 1.0 — a matrix may only ever say "do less of this".
Nothing here trades. A cell with too few trades reports "insufficient evidence"
rather than a confident number, because with a 75 USDT account, three trades in
a cell is noise and acting on it would be curve fitting against your own history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradebot.core.logging import get_logger
from tradebot.core.mathutil import safe_div
from tradebot.core.types import MarketRegime

log = get_logger(__name__)


@dataclass(slots=True)
class Cell:
    """One (row, column) of a matrix: the record of trades in that bucket."""

    trades: int = 0
    wins: int = 0
    r_sum: float = 0.0
    pnl: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0

    @property
    def win_rate(self) -> float:
        return safe_div(self.wins, self.trades, 0.0)

    @property
    def expectancy_r(self) -> float:
        """Average R per trade. The single most useful number in the cell."""
        return safe_div(self.r_sum, self.trades, 0.0)

    def record(self, won: bool, r_multiple: float, pnl: float) -> None:
        self.trades += 1
        self.wins += int(won)
        self.r_sum += r_multiple
        self.pnl += pnl
        self.best_r = max(self.best_r, r_multiple) if self.trades > 1 else r_multiple
        self.worst_r = min(self.worst_r, r_multiple) if self.trades > 1 else r_multiple

    def as_dict(self, min_trades: int) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "win_rate": round(self.win_rate, 3),
            "expectancy_r": round(self.expectancy_r, 3),
            "pnl": round(self.pnl, 4),
            "best_r": round(self.best_r, 2),
            "worst_r": round(self.worst_r, 2),
            "sufficient_evidence": self.trades >= min_trades,
        }


class PerformanceMatrix:
    """A two-dimensional record of results, with a multiplier per cell."""

    def __init__(
        self,
        name: str,
        min_trades: int = 12,
        min_multiplier: float = 0.0,
        floor_expectancy_r: float = -0.25,
    ) -> None:
        self.name = name
        #: Below this many trades a cell reports no opinion. Three trades in a
        #: cell is noise; acting on it is curve fitting against your own history.
        self.min_trades = min_trades
        #: How far a cell may be suppressed. Zero means "stop entirely".
        self.min_multiplier = min_multiplier
        #: Expectancy at or below which a cell is fully suppressed.
        self.floor_expectancy_r = floor_expectancy_r

        self._cells: dict[tuple[str, str], Cell] = {}

    # ------------------------------------------------------------------ #
    def record(self, row: str, column: str, won: bool, r_multiple: float, pnl: float = 0.0) -> None:
        key = (row, column)
        cell = self._cells.get(key)
        if cell is None:
            cell = Cell()
            self._cells[key] = cell
        cell.record(won, r_multiple, pnl)

    def cell(self, row: str, column: str) -> Cell | None:
        return self._cells.get((row, column))

    def has_evidence(self, row: str, column: str) -> bool:
        cell = self._cells.get((row, column))
        return cell is not None and cell.trades >= self.min_trades

    # ------------------------------------------------------------------ #
    def multiplier(self, row: str, column: str) -> float:
        """A weight in [min_multiplier, 1.0] for this combination.

        Never above 1.0. A matrix may say "do less of this"; it may not say "do
        more", because a cell that looks excellent on twelve trades is far more
        likely to be luck than skill, and betting up on it is exactly how a
        small account converts a good run into a large drawdown.
        """
        cell = self._cells.get((row, column))
        if cell is None or cell.trades < self.min_trades:
            return 1.0  # no evidence is not evidence of a problem

        expectancy = cell.expectancy_r
        if expectancy >= 0.0:
            return 1.0
        if expectancy <= self.floor_expectancy_r:
            return self.min_multiplier

        # Linear between "breaking even" and "the floor".
        span = abs(self.floor_expectancy_r)
        fraction = 1.0 - (abs(expectancy) / span) if span > 0 else 1.0
        return max(self.min_multiplier, min(1.0, fraction))

    def blocked(self, row: str, column: str) -> bool:
        """True when this combination is suppressed entirely."""
        return self.multiplier(row, column) <= 0.0 and self.has_evidence(row, column)

    # ------------------------------------------------------------------ #
    def rows(self) -> list[str]:
        return sorted({row for row, _ in self._cells})

    def columns(self) -> list[str]:
        return sorted({column for _, column in self._cells})

    def as_table(self) -> dict[str, dict[str, dict[str, Any]]]:
        table: dict[str, dict[str, dict[str, Any]]] = {}
        for (row, column), cell in self._cells.items():
            table.setdefault(row, {})[column] = {
                **cell.as_dict(self.min_trades),
                "multiplier": round(self.multiplier(row, column), 3),
            }
        return table

    def worst_cells(self, limit: int = 5) -> list[dict[str, Any]]:
        """Combinations with enough evidence to say they are not working."""
        ranked = sorted(
            (
                (row, column, cell)
                for (row, column), cell in self._cells.items()
                if cell.trades >= self.min_trades
            ),
            key=lambda item: item[2].expectancy_r,
        )
        return [
            {"row": row, "column": column, **cell.as_dict(self.min_trades)}
            for row, column, cell in ranked[:limit]
        ]

    def stats(self) -> dict[str, Any]:
        with_evidence = sum(1 for c in self._cells.values() if c.trades >= self.min_trades)
        return {
            "name": self.name,
            "cells": len(self._cells),
            "cells_with_evidence": with_evidence,
            "min_trades": self.min_trades,
            "total_trades": sum(c.trades for c in self._cells.values()),
        }


class MatrixSet:
    """The two matrices the engine keeps, and the combined multiplier.

    Recording is unconditional; whether the result influences sizing is a
    configuration decision, off by default. See ``MatricesConfig``.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.feedback_enabled = bool(getattr(config, "feedback_enabled", False))
        self.strategy_regime = PerformanceMatrix(
            "strategy_regime",
            min_trades=int(getattr(config, "strategy_regime_min_trades", 30)),
            min_multiplier=float(getattr(config, "min_multiplier", 0.0)),
            floor_expectancy_r=float(getattr(config, "floor_expectancy_r", -0.25)),
        )
        self.symbol_strategy = PerformanceMatrix(
            "symbol_strategy",
            min_trades=int(getattr(config, "symbol_strategy_min_trades", 25)),
            min_multiplier=float(getattr(config, "min_multiplier", 0.0)),
            floor_expectancy_r=float(getattr(config, "floor_expectancy_r", -0.25)),
        )

    def record(
        self,
        strategy: str,
        regime: MarketRegime | str,
        symbol: str,
        won: bool,
        r_multiple: float,
        pnl: float = 0.0,
    ) -> None:
        regime_name = regime.value if isinstance(regime, MarketRegime) else str(regime)
        self.strategy_regime.record(strategy, regime_name, won, r_multiple, pnl)
        self.symbol_strategy.record(symbol, strategy, won, r_multiple, pnl)

    def multiplier(self, strategy: str, regime: MarketRegime | str, symbol: str) -> float:
        """Both matrices applied together.

        Returns 1.0 — no influence at all — unless feedback is explicitly
        enabled. When it is: multiplied, not averaged, because a strategy that
        is bad in this regime AND bad on this symbol is worse than either fact
        alone, and averaging would let one good half excuse the other.
        """
        if not self.feedback_enabled:
            return 1.0
        regime_name = regime.value if isinstance(regime, MarketRegime) else str(regime)
        return self.strategy_regime.multiplier(strategy, regime_name) * (
            self.symbol_strategy.multiplier(symbol, strategy)
        )

    def blocked(self, strategy: str, regime: MarketRegime | str, symbol: str) -> bool:
        return self.multiplier(strategy, regime, symbol) <= 0.0

    def report(self) -> dict[str, Any]:
        return {
            "feedback_enabled": self.feedback_enabled,
            "strategy_regime": {
                "stats": self.strategy_regime.stats(),
                "table": self.strategy_regime.as_table(),
                "worst": self.strategy_regime.worst_cells(),
            },
            "symbol_strategy": {
                "stats": self.symbol_strategy.stats(),
                "table": self.symbol_strategy.as_table(),
                "worst": self.symbol_strategy.worst_cells(),
            },
        }
