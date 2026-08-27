"""Correlation engine.

A portfolio of BTC-long, ETH-long, SOL-long and SUI-long is not four
independent bets. In a crypto drawdown those positions move together, so the
portfolio's real risk is close to one position of four times the size — while
every per-trade risk check happily reports 0.5 % each.

This module measures that. It computes the correlation matrix of recent returns
across held and candidate symbols, and expresses concentration two ways:

* **Pairwise** — is this candidate too similar to something already held?
* **Portfolio-level** — how many *independent* bets does the portfolio actually
  contain? The "effective number of positions" is

      N_eff = (Σw)² / (wᵀ · C · w)

  which equals N when everything is uncorrelated and collapses toward 1 as
  correlations approach 1. Four perfectly correlated positions give N_eff = 1,
  which is the honest description of that portfolio.

Direction matters: two long positions in correlated assets compound, while a
long and a short in the same pair partially hedge. Signed exposure handles this
without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tradebot.core.config import RiskConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp, safe_div
from tradebot.core.types import Direction
from tradebot.market.candles import CandleStore

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CorrelationAssessment:
    """What adding this position would do to portfolio concentration."""

    symbol: str
    max_pair_correlation: float
    max_pair_symbol: str | None
    portfolio_correlation: float
    effective_positions: float
    effective_positions_ratio: float
    penalty: float  # 0..1, for the opportunity score
    acceptable: bool
    detail: str = ""
    pairs: dict[str, float] = field(default_factory=dict)


class CorrelationEngine:
    """Measures return correlation between symbols and portfolio concentration."""

    def __init__(self, config: RiskConfig, candles: CandleStore) -> None:
        self.config = config
        self.candles = candles
        self._cache: dict[tuple[str, str], tuple[int, float]] = {}

    # ------------------------------------------------------------------ #
    def returns_for(self, symbol: str, bars: int | None = None) -> np.ndarray:
        """Log returns on the configured correlation timeframe."""
        cfg = self.config
        series = self.candles.get(symbol, cfg.correlation_timeframe)
        if series is None or len(series) < 20:
            return np.array([], dtype=np.float64)
        closes = series.closes[-(bars or cfg.correlation_lookback_bars) :]
        if closes.size < 20:
            return np.array([], dtype=np.float64)
        positive = closes > 0
        if not positive.all():
            return np.array([], dtype=np.float64)
        return np.diff(np.log(closes))

    def correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Pearson correlation of returns; 0.0 when it cannot be computed.

        Returning 0 rather than raising means a symbol with no history is
        treated as uncorrelated — optimistic, so the caller compensates by
        requiring history before relying on the number.
        """
        if symbol_a == symbol_b:
            return 1.0
        key = (symbol_a, symbol_b) if symbol_a < symbol_b else (symbol_b, symbol_a)

        returns_a = self.returns_for(symbol_a)
        returns_b = self.returns_for(symbol_b)
        if returns_a.size < 20 or returns_b.size < 20:
            return 0.0

        length = min(returns_a.size, returns_b.size)
        a, b = returns_a[-length:], returns_b[-length:]
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return 0.0

        value = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(value):
            return 0.0
        value = clamp(value, -1.0, 1.0)
        self._cache[key] = (length, value)
        return value

    def matrix(self, symbols: list[str]) -> np.ndarray:
        """Correlation matrix for the given symbols, in order."""
        n = len(symbols)
        out = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                value = self.correlation(symbols[i], symbols[j])
                out[i, j] = out[j, i] = value
        return out

    # ------------------------------------------------------------------ #
    def effective_positions(self, symbols: list[str], weights: list[float]) -> float:
        """Number of INDEPENDENT bets the portfolio actually contains.

        N when uncorrelated, collapsing toward 1 as correlation rises. Weights
        are signed by direction, so a long and a short in correlated assets
        offset rather than compound.
        """
        if not symbols:
            return 0.0
        if len(symbols) == 1:
            return 1.0

        w = np.asarray(weights, dtype=np.float64)
        total = float(np.sum(np.abs(w)))
        if total <= 0:
            return 0.0

        matrix = self.matrix(symbols)
        variance = float(w @ matrix @ w)
        if variance <= 0:
            # Signed weights cancelled entirely: a fully hedged book. Treat it
            # as maximally diversified rather than dividing by zero.
            return float(len(symbols))
        return clamp(safe_div(total**2, variance, 1.0), 0.0, float(len(symbols)))

    def portfolio_correlation(self, symbols: list[str], weights: list[float]) -> float:
        """Exposure-weighted mean pairwise correlation, in [0, 1].

        SIGNED weights, deliberately. Using absolute values would score a long
        in A against a short in a correlated B as ~1.0 and reject it, when that
        pair is a spread whose legs offset. Concentration is about positions
        that move together IN PNL, which is what the signed product measures.
        Net-negative (hedged) portfolios clamp to 0.
        """
        if len(symbols) < 2:
            return 0.0
        matrix = self.matrix(symbols)
        w = np.asarray(weights, dtype=np.float64)
        total = 0.0
        weight_sum = 0.0
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                signed_weight = w[i] * w[j]
                total += matrix[i, j] * signed_weight
                weight_sum += abs(signed_weight)
        return clamp(safe_div(total, weight_sum, 0.0), 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def assess(
        self,
        candidate: str,
        candidate_direction: Direction,
        candidate_weight: float,
        held: dict[str, tuple[Direction, float]],
    ) -> CorrelationAssessment:
        """Would adding this position leave the portfolio acceptably diversified?

        ``held`` maps symbol -> (direction, exposure). ``candidate_weight`` is
        the exposure the new position would add.
        """
        cfg = self.config

        if not held:
            return CorrelationAssessment(
                symbol=candidate,
                max_pair_correlation=0.0,
                max_pair_symbol=None,
                portfolio_correlation=0.0,
                effective_positions=1.0,
                effective_positions_ratio=1.0,
                penalty=0.0,
                acceptable=True,
                detail="first position; nothing to correlate with",
            )

        # -- pairwise -------------------------------------------------------- #
        pairs: dict[str, float] = {}
        worst_value, worst_symbol = 0.0, None
        for symbol, (direction, _exposure) in held.items():
            raw = self.correlation(candidate, symbol)
            # Same direction: positive correlation compounds risk.
            # Opposite direction: NEGATIVE correlation compounds risk (the two
            # positions then move together in PnL terms).
            aligned = raw * (1 if direction is candidate_direction else -1)
            pairs[symbol] = raw
            if aligned > worst_value:
                worst_value, worst_symbol = aligned, symbol

        if worst_value > cfg.max_pair_correlation:
            return CorrelationAssessment(
                symbol=candidate,
                max_pair_correlation=worst_value,
                max_pair_symbol=worst_symbol,
                portfolio_correlation=1.0,
                effective_positions=0.0,
                effective_positions_ratio=0.0,
                penalty=1.0,
                acceptable=False,
                detail=(
                    f"{worst_value:.2f} correlated with the existing "
                    f"{worst_symbol} position (limit {cfg.max_pair_correlation}); "
                    f"this would be the same bet twice"
                ),
                pairs=pairs,
            )

        # -- portfolio level -------------------------------------------------- #
        symbols = [*held.keys(), candidate]
        weights = [exposure * direction.sign for direction, exposure in held.values()] + [
            candidate_weight * candidate_direction.sign
        ]

        portfolio_corr = self.portfolio_correlation(symbols, weights)
        effective = self.effective_positions(symbols, weights)
        ratio = safe_div(effective, float(len(symbols)), 1.0)

        acceptable = True
        detail = ""
        if portfolio_corr > cfg.max_portfolio_correlation:
            acceptable = False
            detail = (
                f"portfolio correlation would be {portfolio_corr:.2f}, above "
                f"{cfg.max_portfolio_correlation}"
            )
        elif ratio < cfg.min_effective_positions_ratio:
            acceptable = False
            detail = (
                f"{len(symbols)} positions would behave like {effective:.1f} "
                f"independent bets ({ratio:.0%} of nominal, floor "
                f"{cfg.min_effective_positions_ratio:.0%})"
            )

        return CorrelationAssessment(
            symbol=candidate,
            max_pair_correlation=worst_value,
            max_pair_symbol=worst_symbol,
            portfolio_correlation=portfolio_corr,
            effective_positions=effective,
            effective_positions_ratio=ratio,
            penalty=clamp(portfolio_corr, 0.0, 1.0),
            acceptable=acceptable,
            detail=detail or "within correlation limits",
            pairs=pairs,
        )

    def penalties_for(
        self, candidates: list[str], held: dict[str, tuple[Direction, float]]
    ) -> dict[str, float]:
        """Correlation penalties for the scanner, one per candidate symbol."""
        if not held:
            return {}
        out: dict[str, float] = {}
        for symbol in candidates:
            worst = 0.0
            for other, (direction, _) in held.items():
                if other == symbol:
                    continue
                worst = max(worst, abs(self.correlation(symbol, other)))
                _ = direction
            out[symbol] = clamp(worst, 0.0, 1.0)
        return out
