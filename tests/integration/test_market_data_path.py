"""The engine's market data path.

The V1 audit's first critical finding was that `MarketStream`, `UserStream` and
`MarketState` all existed, were unit-tested and worked — and nothing in the
engine ever constructed them. Every unit test passed while the engine ran on
15-second REST polling.

The lesson, and the reason this file exists: a test asserting *the engine uses*
a component is worth more than a test asserting the component works. Everything
here is about the wiring.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap

import pytest

from tradebot.app import runner as runner_module
from tradebot.app.runner import TradingEngine
from tradebot.core.clock import VirtualClock
from tradebot.core.config import AppConfig, Settings, load_tunables
from tradebot.core.types import (
    BookTicker,
    Candidate,
    Candle,
    MarketRegime,
    MarketScore,
    MarkPriceInfo,
    RejectionReason,
    TradingMode,
)
from tradebot.market.scanner import ScanResult
from tradebot.market.state import DataSource, Freshness
from tradebot.market.universe import UniverseReport

from ..conftest import REPO_ROOT

SRC = pathlib.Path(REPO_ROOT / "src" / "tradebot")


def _calls_in(source: str) -> set[str]:
    """Names of every function called in a block of source."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


# --------------------------------------------------------------------------- #
# Structural: the components are actually reachable from the orchestrator
# --------------------------------------------------------------------------- #
class TestStreamsAreWired:
    def test_engine_constructs_and_starts_the_market_stream(self) -> None:
        """C-1: this is the assertion whose absence let the bug ship."""
        calls = _calls_in(inspect.getsource(TradingEngine._start_streams))
        assert "MarketStream" in calls, "the engine never constructs a MarketStream"
        assert "start" in calls, "the market stream is constructed but never started"

    def test_engine_constructs_the_user_stream(self) -> None:
        calls = _calls_in(inspect.getsource(TradingEngine._start_streams))
        assert "UserStream" in calls

    def test_build_starts_the_streams(self) -> None:
        assert "_start_streams" in _calls_in(inspect.getsource(TradingEngine._build))

    def test_teardown_stops_both_streams(self) -> None:
        source = inspect.getsource(TradingEngine._teardown)
        assert "self.market_stream.stop()" in source
        assert "self.user_stream.stop()" in source

    def test_scan_repoints_the_stream_at_the_new_ranking(self) -> None:
        """The top-25 rotates; the subscription must rotate with it."""
        assert "_resubscribe" in _calls_in(inspect.getsource(TradingEngine._scan_once))
        assert "set_symbols" in _calls_in(inspect.getsource(TradingEngine._resubscribe))

    def test_a_reconnect_requests_reconciliation(self) -> None:
        """A gap in the feed is a gap in our knowledge of our own orders."""
        source = inspect.getsource(TradingEngine._on_market_stream_connected)
        assert "_reconcile_requested" in source
        assert "_reconcile_requested" in inspect.getsource(TradingEngine._reconcile_loop)


class TestReadsGoThroughMarketState:
    """No consumer may reach around MarketState to a raw price."""

    def test_the_old_polled_snapshots_are_gone(self) -> None:
        source = inspect.getsource(runner_module)
        assert "self.book_tickers" not in source
        assert "self.mark_prices" not in source

    @pytest.mark.parametrize(
        "method",
        ["_evaluate_candidates", "_manage_positions", "_prices", "open_positions_view"],
    )
    def test_consumers_read_from_market_state(self, method: str) -> None:
        source = inspect.getsource(getattr(TradingEngine, method))
        assert "self.candles.price(" not in source, (
            f"{method} reads a price that carries no freshness information"
        )

    def test_paper_broker_adapter_reads_from_market_state(self) -> None:
        source = inspect.getsource(runner_module._MarketAdapter)
        assert "self._engine.market.price(" in source
        assert "self._engine.market.book(" in source


# --------------------------------------------------------------------------- #
# Behavioural: an engine instance, no network
# --------------------------------------------------------------------------- #
@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> TradingEngine:
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    config = AppConfig(
        settings=Settings(),
        tunables=load_tunables(
            REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
        ),
    )
    assert config.mode is TradingMode.PAPER
    built = TradingEngine(config)
    # A virtual clock everywhere, so freshness is driven by the test.
    built.clock = VirtualClock(start_ms=1_700_000_000_000)
    built.market.clock = built.clock
    return built


def candle_at(clock: VirtualClock, close: float = 100.0) -> Candle:
    now = clock.now_ms()
    return Candle(
        open_time=now - 60_000,
        open=close,
        high=close * 1.002,
        low=close * 0.998,
        close=close,
        volume=100.0,
        close_time=now,
        closed=True,
    )


def scan_result(symbols: list[str], timestamp: int) -> ScanResult:
    candidates = tuple(
        Candidate(
            rank=i + 1,
            market_score=MarketScore(
                symbol=symbol,
                total=85.0,
                components={},
                penalties={},
                volatility=0.005,
                liquidity_usd=5e6,
                spread_bps=1.0,
                funding_rate=0.0001,
                timestamp=timestamp,
            ),
            regime=MarketRegime.STRONG_TREND,
        )
        for i, symbol in enumerate(symbols)
    )
    return ScanResult(
        candidates=candidates,
        scores={c.symbol: c.market_score for c in candidates},
        regimes={},
        universe=UniverseReport(entries=(), excluded={}),
        scanned=len(symbols),
        duration_sec=0.1,
        timestamp=timestamp,
    )


class TestStreamCallbacksFeedTheState:
    @pytest.mark.asyncio
    async def test_a_streamed_candle_becomes_engine_state(self, engine: TradingEngine) -> None:
        await engine._on_stream_candle("BTCUSDT", "5m", candle_at(engine.clock, 100.0))
        assert engine.market.price("BTCUSDT") == 100.0
        assert engine.market.freshness("BTCUSDT") is Freshness.LIVE
        assert engine.market.state_for("BTCUSDT").candle_source is DataSource.WEBSOCKET

    @pytest.mark.asyncio
    async def test_a_streamed_book_becomes_the_engine_price(self, engine: TradingEngine) -> None:
        await engine._on_stream_candle("BTCUSDT", "5m", candle_at(engine.clock, 100.0))
        await engine._on_stream_book(
            BookTicker(
                symbol="BTCUSDT",
                bid_price=101.0,
                bid_qty=10.0,
                ask_price=101.2,
                ask_qty=10.0,
                timestamp=engine.clock.now_ms(),
            )
        )
        assert engine.market.price("BTCUSDT") == pytest.approx(101.1)

    @pytest.mark.asyncio
    async def test_mark_prices_are_kept_only_for_followed_symbols(
        self, engine: TradingEngine
    ) -> None:
        """`!markPrice@arr@1s` carries every symbol on the exchange; storing
        them all would grow the state without bound."""
        engine.market.set_subscribed({"BTCUSDT"})
        for symbol in ("BTCUSDT", "DOGEUSDT"):
            await engine._on_stream_mark(
                MarkPriceInfo(
                    symbol=symbol,
                    mark_price=1.0,
                    index_price=1.0,
                    funding_rate=0.0002,
                    next_funding_time=engine.clock.now_ms() + 600_000,
                    timestamp=engine.clock.now_ms(),
                )
            )
        assert engine.market.mark("BTCUSDT") is not None
        assert engine.market.mark("DOGEUSDT") is None


class TestStaleSymbolsAreExcludedFromEntry:
    @pytest.mark.asyncio
    async def test_a_stale_symbol_never_reaches_the_pipeline(
        self, engine: TradingEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exit criterion for V2-3: stale data cannot become a trade."""
        clock = engine.clock
        primary = engine.tunables.timeframes.primary

        # Both symbols have candles; only FRESHUSDT keeps receiving them.
        for symbol in ("FRESHUSDT", "STALEUSDT"):
            for _ in range(3):
                await engine._on_stream_candle(symbol, primary, candle_at(clock, 100.0))
                clock.advance(1)

        clock.advance(engine.tunables.stream.stale_after_sec + 5)
        await engine._on_stream_candle("FRESHUSDT", primary, candle_at(clock, 100.0))

        assert engine.market.is_tradable("FRESHUSDT") is True
        assert engine.market.is_tradable("STALEUSDT") is False

        seen: list[str] = []

        def spy(view, *args, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(view.symbol)
            raise AssertionError("unreachable in this test")

        monkeypatch.setattr(engine.pipeline, "evaluate", spy)

        class _Scanner:
            last_result = scan_result(["STALEUSDT", "FRESHUSDT"], clock.now_ms())
            correlation_penalties: dict[str, float] = {}

        engine.scanner = _Scanner()  # type: ignore[assignment]

        with pytest.raises(AssertionError):
            await engine._evaluate_candidates()

        assert seen == ["FRESHUSDT"], (
            "a symbol with stale data was offered to the strategy pipeline"
        )
        assert engine.pipeline.rejections.get(RejectionReason.STALE_DATA.value, 0) == 1

    @pytest.mark.asyncio
    async def test_staleness_is_measured_per_symbol_not_globally(
        self, engine: TradingEngine
    ) -> None:
        """One live symbol used to make every symbol look live, because the
        age came from the scan-loop timestamp."""
        clock = engine.clock
        await engine._on_stream_candle("AUSDT", "5m", candle_at(clock))
        clock.advance(60)
        await engine._on_stream_candle("BUSDT", "5m", candle_at(clock))

        assert engine._data_age_sec("BUSDT") == pytest.approx(0.0, abs=0.001)
        assert engine._data_age_sec("AUSDT") == pytest.approx(60.0, abs=0.001)


class TestAccountUpdatesFromTheUserStream:
    @pytest.mark.asyncio
    async def test_account_update_refreshes_equity(self, engine: TradingEngine) -> None:
        await engine._on_user_event(
            "ACCOUNT_UPDATE",
            {
                "e": "ACCOUNT_UPDATE",
                "E": engine.clock.now_ms(),
                "a": {
                    "m": "ORDER",
                    "B": [{"a": "USDT", "wb": "80.0", "cw": "80.0", "bc": "5.0"}],
                    "P": [
                        {
                            "s": "BTCUSDT",
                            "pa": "0.01",
                            "ep": "60000",
                            "up": "1.5",
                            "mt": "isolated",
                            "ps": "BOTH",
                        }
                    ],
                },
            },
        )
        assert engine.equity == pytest.approx(81.5)
        assert engine._reconcile_requested.is_set()

    @pytest.mark.asyncio
    async def test_available_balance_is_not_guessed(self, engine: TradingEngine) -> None:
        """ACCOUNT_UPDATE carries no available balance. Sizing depends on it, so
        it stays at the last REST-confirmed value rather than being inferred."""
        before = engine.available_balance
        await engine._on_user_event(
            "ACCOUNT_UPDATE",
            {"a": {"m": "ORDER", "B": [{"a": "USDT", "wb": "80.0", "cw": "80.0"}], "P": []}},
        )
        assert engine.available_balance == before

    @pytest.mark.asyncio
    async def test_junk_user_events_do_not_raise(self, engine: TradingEngine) -> None:
        for payload in (None, [], {}, "text", [1, 2]):
            await engine._on_user_event("ACCOUNT_UPDATE", payload)
            await engine._on_user_event("ORDER_TRADE_UPDATE", payload)
