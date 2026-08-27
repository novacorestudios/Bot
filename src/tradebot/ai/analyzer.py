"""Advisory analysis layer.

**AI MUST NOT HAVE DIRECT AUTHORITY TO PLACE ORDERS.**

That is a structural property here, not a policy note. Look at what this module
imports: domain types and nothing else. There is no gateway, no
``ExecutionEngine``, no ``OrderIntent``, no ``RiskEngine``. This layer *cannot*
place an order, because it holds no reference to anything that can — the same
guarantee that makes strategies safe, applied again.

What it does instead is produce :class:`Advisory` objects: observations that
become one more *input* to scoring, evaluated by exactly the same gates as any
other input. An advisory can lower a score or raise a warning; it can never
raise a score past a threshold on its own, and it can never bypass the edge
filter or the risk engine.

Deliberately statistical rather than a language model. Every method here is
arithmetic that can be checked by hand, which matters because an unexplainable
input to a trading decision is worse than no input. The interface would accept a
model-based implementation, but the constraint above would still hold.

Enabled by `ai.enabled` in config, and **off by default**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from tradebot.core.logging import get_logger
from tradebot.core.mathutil import clamp, safe_div
from tradebot.core.types import Trade

log = get_logger(__name__)


class AdvisoryKind(StrEnum):
    ANOMALY = "ANOMALY"
    REGIME_WARNING = "REGIME_WARNING"
    STRATEGY_DEGRADATION = "STRATEGY_DEGRADATION"
    COST_MODEL_DRIFT = "COST_MODEL_DRIFT"
    CORRELATION_SHIFT = "CORRELATION_SHIFT"
    PATTERN = "PATTERN"


@dataclass(frozen=True, slots=True)
class Advisory:
    """An observation. Advisory only — it cannot cause or prevent an order.

    ``score_adjustment`` is capped at zero: an advisory may lower confidence in
    an opportunity but never raise it. Anything that could push a trade over a
    threshold would be authority by another name.
    """

    kind: AdvisoryKind
    severity: str  # INFO | WARNING | CRITICAL
    subject: str  # symbol, strategy, or "portfolio"
    message: str
    confidence: float = 50.0
    score_adjustment: float = 0.0  # <= 0 always
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score_adjustment > 0:
            raise ValueError(
                "an advisory may never increase a score; the AI layer has no "
                "authority to make a trade more likely"
            )


class MarketAnalyzer:
    """Statistical analysis producing advisories. Holds no trading references."""

    def __init__(
        self,
        enabled: bool = False,
        anomaly_detection: bool = True,
        post_trade_analysis: bool = True,
    ) -> None:
        self.enabled = enabled
        self.anomaly_detection = anomaly_detection
        self.post_trade_analysis = post_trade_analysis
        self.advisories: list[Advisory] = []

    # ------------------------------------------------------------------ #
    def detect_price_anomaly(
        self, symbol: str, closes: np.ndarray, volumes: np.ndarray, sigma_threshold: float = 4.0
    ) -> Advisory | None:
        """Flag a move that is extreme relative to the symbol's own history.

        Distinct from the PANIC regime, which uses fixed thresholds: this is
        relative to the symbol's own recent distribution, so it catches a 2 %
        move on a normally-placid pair that the absolute test would miss.
        """
        if not (self.enabled and self.anomaly_detection):
            return None
        if closes.size < 60:
            return None

        returns = np.diff(np.log(np.where(closes > 0, closes, np.nan)))
        returns = returns[np.isfinite(returns)]
        if returns.size < 30:
            return None

        recent = float(returns[-1])
        history = returns[:-1]
        std = float(np.std(history))
        if std <= 0:
            return None

        z = abs(recent - float(np.mean(history))) / std
        if z < sigma_threshold:
            return None

        volume_note = ""
        if volumes.size >= 30:
            baseline = float(np.mean(volumes[-30:-1]))
            if baseline > 0:
                ratio = volumes[-1] / baseline
                volume_note = f" on {ratio:.1f}x volume"

        return self._record(
            Advisory(
                kind=AdvisoryKind.ANOMALY,
                severity="WARNING" if z < sigma_threshold * 1.5 else "CRITICAL",
                subject=symbol,
                message=(
                    f"{symbol} moved {recent * 100:+.2f}% in one bar — {z:.1f} "
                    f"standard deviations from its own recent distribution"
                    f"{volume_note}"
                ),
                confidence=clamp(50.0 + z * 8.0, 50.0, 95.0),
                score_adjustment=-clamp(z * 4.0, 0.0, 30.0),
                evidence={"z_score": z, "return": recent, "std": std},
            )
        )

    # ------------------------------------------------------------------ #
    def analyse_strategy(
        self, strategy: str, trades: list[Trade], window: int = 30
    ) -> Advisory | None:
        """Flag a strategy whose recent record has deteriorated.

        Distinct from the strategy kill switch, which acts on the whole history
        against fixed thresholds. This compares the recent window against the
        strategy's own earlier record, so a strategy that was working and has
        stopped is flagged before the aggregate falls far enough to trip.
        """
        if not (self.enabled and self.post_trade_analysis):
            return None
        if len(trades) < window * 2:
            return None

        recent = trades[-window:]
        earlier = trades[:-window]

        recent_expectancy = safe_div(sum(t.r_multiple for t in recent), len(recent), 0.0)
        earlier_expectancy = safe_div(sum(t.r_multiple for t in earlier), len(earlier), 0.0)

        if earlier_expectancy <= 0 or recent_expectancy >= earlier_expectancy * 0.5:
            return None

        recent_wins = sum(1 for t in recent if t.net_pnl > 0)
        return self._record(
            Advisory(
                kind=AdvisoryKind.STRATEGY_DEGRADATION,
                severity="WARNING",
                subject=strategy,
                message=(
                    f"{strategy} expectancy fell from {earlier_expectancy:.3f}R to "
                    f"{recent_expectancy:.3f}R over its last {window} trades "
                    f"({recent_wins}/{window} wins). Often a regime change rather "
                    f"than a broken strategy — regimes return."
                ),
                confidence=70.0,
                score_adjustment=-10.0,
                evidence={
                    "recent_expectancy_r": recent_expectancy,
                    "earlier_expectancy_r": earlier_expectancy,
                    "recent_trades": len(recent),
                },
            )
        )

    # ------------------------------------------------------------------ #
    def analyse_cost_model(
        self, strategy: str, expected_edges: list[float], realised_edges: list[float]
    ) -> Advisory | None:
        """Flag a persistent gap between predicted and realised edge.

        This is the most consequential check in the module. The edge filter
        gates every trade, so if its predictions are systematically optimistic,
        every trade the bot takes is worse than it believed. Usually slippage or
        the win-probability estimate.
        """
        if not self.enabled or len(realised_edges) < 20:
            return None

        expected = float(np.mean(expected_edges))
        realised = float(np.mean(realised_edges))
        gap = realised - expected

        if expected <= 0 or gap >= -abs(expected) * 0.3:
            return None

        return self._record(
            Advisory(
                kind=AdvisoryKind.COST_MODEL_DRIFT,
                severity="CRITICAL",
                subject=strategy,
                message=(
                    f"{strategy} realised {realised * 100:.4f}% against a predicted "
                    f"{expected * 100:.4f}% over {len(realised_edges)} trades. The "
                    f"edge model is optimistic, so every trade it approves is worse "
                    f"than it believes. Re-fit slippage before trusting new signals."
                ),
                confidence=85.0,
                score_adjustment=-20.0,
                evidence={
                    "expected_mean": expected,
                    "realised_mean": realised,
                    "gap": gap,
                    "samples": len(realised_edges),
                },
            )
        )

    # ------------------------------------------------------------------ #
    def analyse_regime_performance(self, trades: list[Trade]) -> dict[str, dict[str, float]]:
        """Per-regime performance, for the operator rather than the engine.

        Returns data, not advisories: which regime a strategy works in is a
        configuration question a human should answer, not something the system
        should quietly adjust for itself.
        """
        if not self.enabled:
            return {}

        buckets: dict[str, list[Trade]] = {}
        for trade in trades:
            buckets.setdefault(trade.regime.value, []).append(trade)

        out: dict[str, dict[str, float]] = {}
        for regime, group in buckets.items():
            wins = sum(1 for t in group if t.net_pnl > 0)
            out[regime] = {
                "trades": len(group),
                "win_rate": round(safe_div(wins, len(group), 0.0), 4),
                "expectancy_r": round(
                    safe_div(sum(t.r_multiple for t in group), len(group), 0.0), 4
                ),
                "net_pnl": round(sum(t.net_pnl for t in group), 6),
            }
        return out

    def analyse_exit_reasons(self, trades: list[Trade]) -> dict[str, Any]:
        """Where trades actually end — often the most diagnostic breakdown.

        A high share of TIME_LIMIT exits means targets are set beyond what a
        60-minute holding period can deliver, which no amount of entry tuning
        will fix.
        """
        if not trades:
            return {}

        counts: dict[str, int] = {}
        pnl: dict[str, float] = {}
        for trade in trades:
            key = trade.exit_reason.value
            counts[key] = counts.get(key, 0) + 1
            pnl[key] = pnl.get(key, 0.0) + trade.net_pnl

        time_share = safe_div(counts.get("TIME_LIMIT", 0), len(trades), 0.0)
        notes: list[str] = []
        if time_share > 0.4:
            notes.append(
                f"{time_share:.0%} of trades ended on the time limit: targets "
                f"are likely set beyond what a 60-minute hold can reach"
            )
        stop_share = safe_div(counts.get("STOP_LOSS", 0), len(trades), 0.0)
        if stop_share > 0.6:
            notes.append(
                f"{stop_share:.0%} of trades hit their stop: stops may be inside "
                f"normal noise for these symbols"
            )

        return {
            "counts": counts,
            "pnl_by_reason": {k: round(v, 6) for k, v in pnl.items()},
            "notes": notes,
        }

    # ------------------------------------------------------------------ #
    def summarise(self, trades: list[Trade], equity: float, peak_equity: float) -> str:
        """A plain-language summary for a Telegram or dashboard report."""
        if not trades:
            return (
                "No trades yet. With opportunity-driven trading this is a "
                "valid state — check the rejection counts to see why."
            )

        wins = sum(1 for t in trades if t.net_pnl > 0)
        net = sum(t.net_pnl for t in trades)
        costs = sum(t.fees + t.funding for t in trades)
        gross = sum(t.gross_pnl for t in trades)
        drawdown = safe_div(peak_equity - equity, peak_equity, 0.0)

        lines = [
            f"{len(trades)} trades, {safe_div(wins, len(trades), 0.0):.0%} win "
            f"rate, net {net:+.4f}.",
            f"Costs consumed {safe_div(costs, abs(gross), 0.0):.0%} of gross "
            f"profit ({costs:.4f} of {gross:+.4f}).",
        ]
        if drawdown > 0.02:
            lines.append(f"Currently {drawdown:.1%} below the equity peak.")
        if len(trades) < 100:
            lines.append("Fewer than 100 trades: these figures are dominated by noise.")
        return " ".join(lines)

    # ------------------------------------------------------------------ #
    def _record(self, advisory: Advisory) -> Advisory:
        self.advisories.append(advisory)
        log.info(
            "advisory",
            kind=advisory.kind.value,
            subject=advisory.subject,
            severity=advisory.severity,
            message=advisory.message,
        )
        return advisory

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "kind": a.kind.value,
                "severity": a.severity,
                "subject": a.subject,
                "message": a.message,
                "confidence": a.confidence,
                "score_adjustment": a.score_adjustment,
            }
            for a in self.advisories[-limit:]
        ]

    def total_adjustment(self, subject: str) -> float:
        """Combined (always non-positive) score adjustment for a subject."""
        return clamp(
            sum(a.score_adjustment for a in self.advisories if a.subject == subject),
            -50.0,
            0.0,
        )
