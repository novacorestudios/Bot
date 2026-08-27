"""Dynamic scanner, scoring, universe and regime detection.

The property that matters most: **no symbol is privileged**. The ranking is a
pure function of current data, so a symbol with dead volume ranks below an
active one regardless of its name or market cap.
"""

from __future__ import annotations

import pytest

from tradebot.core.config import (
    EdgeConfig,
    RegimeConfig,
    ScannerConfig,
)
from tradebot.core.types import Direction, MarketRegime
from tradebot.market.candles import CandleSeries, CandleStore
from tradebot.market.microstructure import CostModel, LiquiditySnapshot
from tradebot.market.regime import RegimeDetector
from tradebot.market.scanner import MarketScanner
from tradebot.market.scoring import MarketScorer, ScoringInputs
from tradebot.market.universe import UniverseBuilder

from ..conftest import flat_prices, make_candles, ranging_prices, trend_prices
from ..fakes import FakeGateway, book_for, make_symbol_info, mark_for, ticker_for


def series_from(prices, volumes=None, timeframe="5m") -> CandleSeries:
    series = CandleSeries("X", timeframe, 500)
    series.extend(make_candles(prices, volumes=volumes))
    return series


def scorer() -> MarketScorer:
    return MarketScorer(ScannerConfig(), CostModel(EdgeConfig()))


def inputs_for(
    prices,
    *,
    symbol="X",
    spread=1.0,
    depth=500_000.0,
    volume=5e8,
    funding=0.0001,
    volumes=None,
    correlation=0.0,
):
    return ScoringInputs(
        symbol=symbol,
        series=series_from(prices, volumes),
        liquidity=LiquiditySnapshot(symbol, spread, depth, depth, 0.1, volume),
        funding_rate=funding,
        quote_volume_24h=volume,
        correlation_penalty=correlation,
        timestamp=0,
    )


class TestScoringDiscrimination:
    def test_active_market_outranks_a_dead_one(self):
        active = scorer().score(inputs_for(trend_prices(200, drift=0.0008, noise=0.0015)))
        dead = scorer().score(inputs_for(flat_prices(200), spread=5.0, depth=2_000.0, volume=2e7))
        assert active.total > dead.total

    def test_no_symbol_name_influences_the_score(self):
        """The scanner must have no notion of a 'good' coin."""
        prices = trend_prices(200, drift=0.0008)
        btc = scorer().score(inputs_for(prices, symbol="BTCUSDT"))
        obscure = scorer().score(inputs_for(prices, symbol="ZZZWEIRDUSDT"))
        assert btc.total == pytest.approx(obscure.total)

    def test_tight_spread_scores_above_wide(self):
        prices = trend_prices(200)
        tight = scorer().score(inputs_for(prices, spread=0.5))
        wide = scorer().score(inputs_for(prices, spread=5.5))
        assert tight.components["spread"] > wide.components["spread"]
        assert tight.total > wide.total

    def test_deeper_book_scores_higher_liquidity(self):
        prices = trend_prices(200)
        deep = scorer().score(inputs_for(prices, depth=2_000_000.0))
        thin = scorer().score(inputs_for(prices, depth=5_000.0))
        assert deep.components["liquidity"] > thin.components["liquidity"]

    def test_volatility_is_scored_as_a_band_not_a_maximum(self):
        """Too calm and too wild are both wrong for a sub-hour strategy."""
        calm = scorer().score(inputs_for(flat_prices(200)))
        ideal = scorer().score(inputs_for(trend_prices(200, drift=0.0005, noise=0.003)))
        wild = scorer().score(inputs_for(trend_prices(200, drift=0.0, noise=0.06)))
        assert ideal.components["volatility"] > calm.components["volatility"]
        assert ideal.components["volatility"] > wild.components["volatility"]

    def test_extreme_funding_is_penalised(self):
        prices = trend_prices(200)
        cheap = scorer().score(inputs_for(prices, funding=0.00001))
        expensive = scorer().score(inputs_for(prices, funding=0.003))
        assert cheap.components["funding"] > expensive.components["funding"]

    def test_correlation_penalty_lowers_the_score(self):
        prices = trend_prices(200)
        alone = scorer().score(inputs_for(prices, correlation=0.0))
        crowded = scorer().score(inputs_for(prices, correlation=1.0))
        assert crowded.total < alone.total

    def test_cost_penalty_grows_when_costs_eat_the_move(self):
        """A symbol whose spread exceeds its typical move is nearly untradable."""
        quiet_expensive = scorer().score(inputs_for(flat_prices(200), spread=6.0))
        assert quiet_expensive.penalties["estimated_cost"] > 0

    def test_score_is_bounded_zero_to_hundred(self):
        for prices in (trend_prices(200), flat_prices(200), ranging_prices(200)):
            score = scorer().score(inputs_for(prices))
            assert 0.0 <= score.total <= 100.0

    def test_insufficient_history_scores_neutral_not_zero(self):
        """A newly listed symbol should not be pushed to the bottom by ignorance."""
        score = scorer().score(inputs_for(trend_prices(10)))
        assert score.components["recent_volume"] == pytest.approx(50.0)


class TestUniverse:
    def _symbols_and_tickers(self):
        symbols = {
            "GOODUSDT": make_symbol_info("GOODUSDT"),
            "ALSOUSDT": make_symbol_info("ALSOUSDT"),
            "THINUSDT": make_symbol_info("THINUSDT"),
            "HALTUSDT": make_symbol_info("HALTUSDT", status="BREAK"),
            "BUSDPAIR": make_symbol_info("BUSDPAIR", quote="BUSD"),
            "DATEDUSDT": make_symbol_info("DATEDUSDT", contract_type="CURRENT_QUARTER"),
            "EXPENSIVEUSDT": make_symbol_info("EXPENSIVEUSDT", min_notional=50_000.0),
        }
        tickers = {
            "GOODUSDT": ticker_for("GOODUSDT", 100.0, 5e8),
            "ALSOUSDT": ticker_for("ALSOUSDT", 2.0, 9e8),
            "THINUSDT": ticker_for("THINUSDT", 1.0, 1e6),
            "HALTUSDT": ticker_for("HALTUSDT", 1.0, 5e8),
            "BUSDPAIR": ticker_for("BUSDPAIR", 1.0, 5e8),
            "DATEDUSDT": ticker_for("DATEDUSDT", 1.0, 5e8),
            "EXPENSIVEUSDT": ticker_for("EXPENSIVEUSDT", 100.0, 5e8),
        }
        return symbols, tickers

    def test_only_tradable_usdt_perpetuals_survive(self):
        symbols, tickers = self._symbols_and_tickers()
        report = UniverseBuilder(ScannerConfig(), equity=75.0).build(symbols, tickers)
        assert set(report.symbols) == {"GOODUSDT", "ALSOUSDT"}

    @pytest.mark.parametrize(
        ("symbol", "reason"),
        [
            ("HALTUSDT", "STATUS_BREAK"),
            ("BUSDPAIR", "WRONG_QUOTE_ASSET"),
            ("DATEDUSDT", "WRONG_CONTRACT_TYPE"),
            ("THINUSDT", "INSUFFICIENT_24H_VOLUME"),
            ("EXPENSIVEUSDT", "MIN_NOTIONAL_TOO_LARGE_FOR_ACCOUNT"),
        ],
    )
    def test_every_exclusion_is_recorded_with_a_reason(self, symbol, reason):
        """Auditability: we must be able to say why a symbol was never considered."""
        symbols, tickers = self._symbols_and_tickers()
        report = UniverseBuilder(ScannerConfig(), equity=75.0).build(symbols, tickers)
        assert report.excluded[symbol] == reason

    def test_a_larger_account_can_afford_more_symbols(self):
        symbols, tickers = self._symbols_and_tickers()
        small = UniverseBuilder(ScannerConfig(), equity=75.0).build(symbols, tickers)
        large = UniverseBuilder(ScannerConfig(), equity=500_000.0).build(symbols, tickers)
        assert len(large.symbols) > len(small.symbols)

    def test_deny_list_is_honoured(self):
        symbols, tickers = self._symbols_and_tickers()
        config = ScannerConfig(deny_list=("GOODUSDT",))
        report = UniverseBuilder(config, equity=75.0).build(symbols, tickers)
        assert "GOODUSDT" not in report.symbols
        assert report.excluded["GOODUSDT"] == "DENY_LIST"

    def test_allow_list_restricts_to_itself(self):
        symbols, tickers = self._symbols_and_tickers()
        config = ScannerConfig(allow_list=("ALSOUSDT",))
        report = UniverseBuilder(config, equity=75.0).build(symbols, tickers)
        assert report.symbols == ("ALSOUSDT",)

    def test_symbol_without_ticker_data_is_excluded(self):
        symbols, _ = self._symbols_and_tickers()
        report = UniverseBuilder(ScannerConfig(), equity=75.0).build(symbols, {})
        assert report.symbols == ()


class TestRegimeDetection:
    def detector(self, **overrides) -> RegimeDetector:
        return RegimeDetector(RegimeConfig(**overrides))

    def classify(self, prices, volumes=None) -> MarketRegime:
        return self.detector().detect(series_from(prices, volumes)).regime

    def test_strong_trend(self):
        assert (
            self.classify(trend_prices(250, drift=0.002, noise=0.0004)) is MarketRegime.STRONG_TREND
        )

    def test_flat_market_is_not_trending(self):
        assert self.classify(flat_prices(250)) in {
            MarketRegime.SIDEWAYS,
            MarketRegime.LOW_VOLATILITY,
        }

    def test_crash_is_panic(self):
        prices = trend_prices(240, drift=0.0001, noise=0.0005)
        prices += [prices[-1] * 0.97, prices[-1] * 0.94, prices[-1] * 0.90]
        assert self.classify(prices) is MarketRegime.PANIC

    def test_panic_blocks_every_strategy(self):
        """This is the point of the regime: in panic, nothing runs."""
        assert self.detector().strategy_weights(MarketRegime.PANIC) == {}

    def test_moderate_move_alone_is_not_panic(self):
        """Otherwise every news candle would suspend trading."""
        prices = trend_prices(240, drift=0.0001, noise=0.0005)
        prices += [prices[-1] * 0.995, prices[-1] * 0.992, prices[-1] * 0.99]
        assert self.classify(prices) is not MarketRegime.PANIC

    def test_moderate_move_with_volume_explosion_is_panic(self):
        prices = trend_prices(240, drift=0.0001, noise=0.0005)
        prices += [prices[-1] * 0.99, prices[-1] * 0.985, prices[-1] * 0.978]
        volumes = [1000.0] * 240 + [9000.0] * 3
        assert self.classify(prices, volumes) is MarketRegime.PANIC

    def test_insufficient_history_is_reported_not_guessed(self):
        state = self.detector().detect(series_from(trend_prices(20)))
        assert "INSUFFICIENT_DATA" in state.reasons
        assert state.confidence == 0.0

    def test_trend_direction_is_reported(self):
        up = self.detector().detect(series_from(trend_prices(250, drift=0.002, noise=0.0003)))
        down = self.detector().detect(series_from(trend_prices(250, drift=-0.002, noise=0.0003)))
        assert up.direction is Direction.LONG
        assert down.direction is Direction.SHORT

    def test_regime_weights_gate_strategies(self):
        detector = self.detector()
        detector.config.strategy_weights.update(
            {"STRONG_TREND": {"momentum": 1.0}, "SIDEWAYS": {"mean_reversion": 1.0}}
        )
        assert detector.allows(MarketRegime.STRONG_TREND, "momentum")
        assert not detector.allows(MarketRegime.STRONG_TREND, "mean_reversion")
        assert not detector.allows(MarketRegime.PANIC, "momentum")

    def test_every_state_carries_its_evidence(self):
        """'Why did it think that?' must always be answerable."""
        state = self.detector().detect(series_from(trend_prices(250, drift=0.002)))
        assert state.reasons
        assert "adx" in state.as_dict()


class TestScanCycle:
    def build_gateway(self, specs) -> FakeGateway:
        """specs: {symbol: (prices, quote_volume, spread_bps)}"""
        gateway = FakeGateway()
        for symbol, (prices, volume, spread) in specs.items():
            gateway.symbols[symbol] = make_symbol_info(symbol)
            gateway.klines[(symbol, "5m")] = make_candles(prices)
            gateway.books[symbol] = book_for(symbol, prices[-1], spread)
            gateway.tickers[symbol] = ticker_for(symbol, prices[-1], volume)
            gateway.marks[symbol] = mark_for(symbol, prices[-1])
        return gateway

    def make_scanner(self, gateway, **config_overrides) -> MarketScanner:
        config = ScannerConfig(**config_overrides)
        cost_model = CostModel(EdgeConfig())
        return MarketScanner(
            config=config,
            gateway=gateway,
            candles=CandleStore(500),
            scorer=MarketScorer(config, cost_model),
            regime_detector=RegimeDetector(RegimeConfig()),
            universe_builder=UniverseBuilder(config, equity=75.0),
            cost_model=cost_model,
            primary_timeframe="5m",
        )

    async def test_ranking_is_driven_by_data_not_by_symbol(self):
        gateway = self.build_gateway(
            {
                "BTCUSDT": (flat_prices(250), 5e8, 4.0),  # dead
                "OBSCUREUSDT": (trend_prices(250, drift=0.0008, noise=0.002), 5e8, 0.5),
            }
        )
        result = await self.make_scanner(gateway, top_markets=5).scan()
        assert result.candidates
        assert result.candidates[0].symbol == "OBSCUREUSDT", (
            "an active obscure market must outrank a dead major one"
        )

    async def test_top_n_is_respected(self):
        specs = {
            f"SYM{i}USDT": (trend_prices(250, drift=0.0005, seed=i), 5e8, 1.0) for i in range(12)
        }
        result = await self.make_scanner(self.build_gateway(specs), top_markets=5).scan()
        assert len(result.candidates) == 5

    async def test_candidates_are_ranked_descending(self):
        specs = {
            f"SYM{i}USDT": (trend_prices(250, drift=0.0002 * i, seed=i), 5e8, 1.0) for i in range(8)
        }
        result = await self.make_scanner(self.build_gateway(specs), top_markets=8).scan()
        totals = [c.market_score.total for c in result.candidates]
        assert totals == sorted(totals, reverse=True)
        assert [c.rank for c in result.candidates] == list(range(1, len(totals) + 1))

    async def test_a_failing_symbol_does_not_abort_the_scan(self):
        specs = {
            "GOODUSDT": (trend_prices(250), 5e8, 1.0),
            "BROKENUSDT": (trend_prices(250), 5e8, 1.0),
        }
        gateway = self.build_gateway(specs)
        gateway.fail_symbols.add("BROKENUSDT")
        result = await self.make_scanner(gateway, top_markets=5).scan()
        assert "GOODUSDT" in result.symbols
        assert "BROKENUSDT" in result.errors

    async def test_empty_universe_returns_no_candidates_without_raising(self):
        result = await self.make_scanner(FakeGateway()).scan()
        assert result.candidates == ()

    async def test_prefilter_limits_expensive_kline_requests(self):
        """Fetching klines for the whole universe would consume the rate budget."""
        specs = {f"SYM{i}USDT": (trend_prices(250, seed=i), 5e8 - i * 1e6, 1.0) for i in range(60)}
        gateway = self.build_gateway(specs)
        await self.make_scanner(gateway, top_markets=5).scan()
        assert gateway.kline_calls <= 15 + 1, (
            f"prefilter should cap kline requests; made {gateway.kline_calls}"
        )

    async def test_symbols_with_positions_are_always_scored(self):
        """A held symbol that drops out of the ranking still needs its data."""
        specs = {
            f"SYM{i}USDT": (trend_prices(250, drift=0.001, seed=i), 9e8, 0.5) for i in range(30)
        }
        specs["HELDUSDT"] = (flat_prices(250), 3e7, 5.0)  # would never rank
        gateway = self.build_gateway(specs)
        result = await self.make_scanner(gateway, top_markets=5).scan(protected={"HELDUSDT"})
        assert "HELDUSDT" in result.scores

    async def test_scan_result_table_is_renderable(self):
        gateway = self.build_gateway({"AUSDT": (trend_prices(250), 5e8, 1.0)})
        rows = (await self.make_scanner(gateway, top_markets=5).scan()).table()
        assert rows and {"rank", "symbol", "market_score", "regime"} <= set(rows[0])

    async def test_correlation_penalties_reach_the_scorer(self):
        specs = {
            "AUSDT": (trend_prices(250, drift=0.0006, seed=1), 5e8, 1.0),
            "BUSDT": (trend_prices(250, drift=0.0006, seed=1), 5e8, 1.0),
        }
        scanner = self.make_scanner(self.build_gateway(specs), top_markets=5)
        baseline = await scanner.scan()
        scanner.set_correlation_penalties({"AUSDT": 1.0})
        penalised = await scanner.scan()
        assert penalised.scores["AUSDT"].total < baseline.scores["AUSDT"].total
