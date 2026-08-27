"""Dynamic market scanner.

Each cycle: discover the tradable universe, gather the data needed to score it,
score every symbol, and publish the top N candidates. Nothing about which coins
appear is fixed — the ranking is recomputed from current data every cycle, and a
symbol that was first last cycle can be absent from this one.

Rate-limit discipline is the main engineering constraint. Scoring 200 symbols
naively would need 200 kline requests per cycle at weight 5-10 each, which is
most of the minute's budget. So:

* 24h tickers and book tickers are fetched **once for the whole market** (one
  request each) rather than per symbol.
* Klines are fetched only for a **prefilter shortlist** — the symbols that
  survive the cheap filters and rank highest on data we already have.
* Symbols already streaming over WebSocket use their in-memory candles and cost
  nothing at all.

The result is a full re-rank for a few hundred weight units, comfortably inside
the budget even on a five-minute cycle.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from tradebot.core.config import ScannerConfig
from tradebot.core.logging import get_logger
from tradebot.core.types import (
    BookTicker,
    Candidate,
    Direction,
    MarketRegime,
    MarketScore,
    SymbolInfo,
    Ticker24h,
)
from tradebot.market.candles import CandleStore
from tradebot.market.microstructure import CostModel, snapshot_from_book
from tradebot.market.regime import RegimeDetector, RegimeState
from tradebot.market.scoring import MarketScorer, ScoringInputs
from tradebot.market.universe import UniverseBuilder, UniverseReport

log = get_logger(__name__)


@dataclass(slots=True)
class ScanResult:
    """One completed scan cycle."""

    candidates: tuple[Candidate, ...]
    scores: dict[str, MarketScore]
    regimes: dict[str, RegimeState]
    universe: UniverseReport
    scanned: int
    duration_sec: float
    timestamp: int
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(c.symbol for c in self.candidates)

    def table(self) -> list[dict[str, Any]]:
        """Rows for the opportunity dashboard and the CLI `scan` command."""
        return [
            {
                "rank": c.rank,
                "symbol": c.symbol,
                "market_score": round(c.market_score.total, 1),
                "regime": c.regime.value,
                "volatility_pct": round(c.market_score.volatility * 100, 3),
                "liquidity_usd": round(c.market_score.liquidity_usd, 0),
                "spread_bps": round(c.market_score.spread_bps, 2),
                "funding": round(c.market_score.funding_rate, 6),
                "best_strategy": c.best_strategy or "-",
                "direction": c.direction.value,
                "confidence": round(c.confidence, 1),
                "expected_net_edge": (
                    round(c.expected_net_edge, 6) if c.expected_net_edge is not None else None
                ),
                "opportunity_score": (
                    round(c.opportunity_score, 1) if c.opportunity_score is not None else None
                ),
                "risk_level": c.risk_level,
            }
            for c in self.candidates
        ]


class MarketScanner:
    """Ranks the whole tradable universe and publishes the top N candidates."""

    def __init__(
        self,
        config: ScannerConfig,
        gateway: Any,
        candles: CandleStore,
        scorer: MarketScorer,
        regime_detector: RegimeDetector,
        universe_builder: UniverseBuilder,
        cost_model: CostModel,
        primary_timeframe: str = "5m",
        prefilter_multiple: int = 3,
        max_concurrent_requests: int = 8,
    ) -> None:
        self.config = config
        self.gateway = gateway
        self.candles = candles
        self.scorer = scorer
        self.regime_detector = regime_detector
        self.universe_builder = universe_builder
        self.cost_model = cost_model
        self.primary_timeframe = primary_timeframe
        # How many symbols to fetch klines for, as a multiple of top_markets.
        self.prefilter_multiple = prefilter_multiple
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.last_result: ScanResult | None = None
        self.scan_count = 0
        self.correlation_penalties: dict[str, float] = {}

    def set_correlation_penalties(self, penalties: dict[str, float]) -> None:
        """Fed by the correlation engine so held exposure lowers similar symbols."""
        self.correlation_penalties = dict(penalties)

    async def scan(self, protected: set[str] | None = None) -> ScanResult:
        """Run one full scan cycle."""
        started = time.time()
        errors: dict[str, str] = {}
        protected = protected or set()

        # -- 1. universe (2 cheap market-wide requests) ----------------------- #
        symbols: dict[str, SymbolInfo] = self.gateway.symbols
        if not symbols:
            symbols = await self.gateway.load_symbols()

        tickers: dict[str, Ticker24h] = await self.gateway.get_ticker_24h()
        books: dict[str, BookTicker] = await self.gateway.get_book_ticker()

        universe = self.universe_builder.build(symbols, tickers)
        if not universe.entries:
            log.warning("universe_empty", excluded=universe.exclusion_counts())
            result = ScanResult(
                (), {}, {}, universe, 0, time.time() - started, int(time.time() * 1000), errors
            )
            self.last_result = result
            return result

        # Funding rates: one market-wide request.
        try:
            marks = await self.gateway.get_mark_price()
        except Exception as exc:  # noqa: BLE001 - a scan must survive this
            log.warning("mark_price_fetch_failed", error=str(exc))
            marks = {}

        # -- 2. prefilter -------------------------------------------------- #
        shortlist = self._prefilter(universe, books, protected)

        # -- 3. fetch candles only for the shortlist ------------------------ #
        await self._ensure_candles(shortlist, errors)

        # -- 4. score and classify ------------------------------------------ #
        scores: dict[str, MarketScore] = {}
        regimes: dict[str, RegimeState] = {}
        now_ms = int(time.time() * 1000)

        for symbol in shortlist:
            series = self.candles.get(symbol, self.primary_timeframe)
            if series is None or series.is_empty:
                errors.setdefault(symbol, "NO_CANDLES")
                continue

            entry = next((e for e in universe.entries if e.symbol == symbol), None)
            mark = marks.get(symbol)

            inputs = ScoringInputs(
                symbol=symbol,
                series=series,
                liquidity=snapshot_from_book(
                    symbol,
                    books.get(symbol),
                    quote_volume_24h=entry.quote_volume_24h if entry else 0.0,
                ),
                funding_rate=mark.funding_rate if mark else 0.0,
                quote_volume_24h=entry.quote_volume_24h if entry else 0.0,
                price_change_24h=entry.price_change_24h if entry else 0.0,
                correlation_penalty=self.correlation_penalties.get(symbol, 0.0),
                timestamp=now_ms,
            )
            try:
                scores[symbol] = self.scorer.score(inputs)
                regimes[symbol] = self.regime_detector.detect(series)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not
                errors[symbol] = f"SCORING_FAILED: {exc}"  # abort the whole scan
                log.warning("symbol_scoring_failed", symbol=symbol, error=str(exc))

        # -- 5. rank ---------------------------------------------------------- #
        ranked = sorted(scores.values(), key=lambda s: s.total, reverse=True)
        top = ranked[: self.config.top_markets]

        candidates = tuple(
            Candidate(
                rank=index + 1,
                market_score=score,
                regime=regimes.get(score.symbol, _default_regime()).regime,
                risk_level=_risk_level(score),
            )
            for index, score in enumerate(top)
        )

        duration = time.time() - started
        self.scan_count += 1
        result = ScanResult(
            candidates=candidates,
            scores=scores,
            regimes=regimes,
            universe=universe,
            scanned=len(shortlist),
            duration_sec=duration,
            timestamp=now_ms,
            errors=errors,
        )
        self.last_result = result

        log.info(
            "scan_complete",
            universe=len(universe.entries),
            scanned=len(shortlist),
            ranked=len(candidates),
            duration_sec=round(duration, 2),
            errors=len(errors),
            top5=[(c.symbol, round(c.market_score.total, 1)) for c in candidates[:5]],
        )
        return result

    # ------------------------------------------------------------------ #
    def _prefilter(
        self, universe: UniverseReport, books: dict[str, BookTicker], protected: set[str]
    ) -> list[str]:
        """Shortlist symbols worth spending a kline request on.

        Ranks on data already in hand (24h volume and quoted spread), then takes
        a multiple of the final top-N so the expensive scoring has room to
        reorder things meaningfully.
        """
        cfg = self.config
        cheap_ranked: list[tuple[float, str]] = []

        for entry in universe.entries:
            book = books.get(entry.symbol)
            if book is None:
                continue
            spread_bps = book.spread_bps
            if spread_bps > cfg.max_spread_bps:
                continue
            # Cheap proxy: high volume and tight spread.
            proxy = entry.quote_volume_24h / max(spread_bps, 0.1)
            cheap_ranked.append((proxy, entry.symbol))

        cheap_ranked.sort(reverse=True)
        limit = max(cfg.top_markets, cfg.top_markets * self.prefilter_multiple)
        shortlist = [symbol for _, symbol in cheap_ranked[:limit]]

        # Symbols with open positions must always be scored, whatever their rank.
        for symbol in protected:
            if symbol not in shortlist:
                shortlist.append(symbol)
        return shortlist

    async def _ensure_candles(self, symbols: list[str], errors: dict[str, str]) -> None:
        """Fetch history for symbols we do not already have enough bars for."""
        needed = self.config.scoring_lookback_bars
        missing = [
            symbol
            for symbol in symbols
            if not self.candles.has(symbol, self.primary_timeframe, needed)
        ]
        if not missing:
            return

        async def fetch(symbol: str) -> None:
            async with self._semaphore:
                try:
                    candles = await self.gateway.get_klines(
                        symbol, self.primary_timeframe, limit=needed
                    )
                except Exception as exc:  # noqa: BLE001
                    errors[symbol] = f"KLINE_FETCH_FAILED: {exc}"
                    return
                series = self.candles.series(symbol, self.primary_timeframe)
                series.extend(candles)

        await asyncio.gather(*(fetch(symbol) for symbol in missing))
        log.debug(
            "candles_fetched",
            requested=len(missing),
            failed=len([s for s in missing if s in errors]),
        )


def _default_regime() -> RegimeState:
    from tradebot.market.regime import UNKNOWN

    return UNKNOWN


def _risk_level(score: MarketScore) -> str:
    """A coarse label for the dashboard, derived from volatility and spread."""
    volatility = score.volatility
    if volatility <= 0:
        return "UNKNOWN"
    if volatility > 0.02 or score.spread_bps > 8:
        return "HIGH"
    if volatility > 0.01 or score.spread_bps > 4:
        return "MEDIUM"
    return "LOW"


def enrich_candidates(
    candidates: tuple[Candidate, ...],
    best: dict[str, tuple[str, Direction, float, float | None, float | None]],
) -> tuple[Candidate, ...]:
    """Attach each symbol's best strategy/edge to its candidate row.

    Called after the strategy pass so the opportunity dashboard can show what the
    engine actually thinks, while keeping the scanner itself free of any strategy
    dependency.
    """
    from dataclasses import replace

    out: list[Candidate] = []
    for candidate in candidates:
        found = best.get(candidate.symbol)
        if found is None:
            out.append(candidate)
            continue
        strategy, direction, confidence, edge, opportunity = found
        out.append(
            replace(
                candidate,
                best_strategy=strategy,
                direction=direction,
                confidence=confidence,
                expected_net_edge=edge,
                opportunity_score=opportunity,
            )
        )
    return tuple(out)


__all__ = ["MarketScanner", "ScanResult", "enrich_candidates", "MarketRegime"]
