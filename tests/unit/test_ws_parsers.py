"""WebSocket payload parsing.

Every payload here is shaped like the ones in the official Binance USDⓈ-M
Futures WebSocket documentation. The array-shaped ``!markPrice@arr@1s`` payload
is the one that mattered: the previous handler called ``.get()`` on it and
raised ``AttributeError: 'list' object has no attribute 'get'``, which killed
the market feed within a second of connecting (AUDIT_REPORT.md C-2).
"""

from __future__ import annotations

import pytest

from tradebot.exchange.binance import parsers

# --------------------------------------------------------------------------- #
# Documented payloads
# --------------------------------------------------------------------------- #
KLINE = {
    "e": "kline",
    "E": 1_700_000_000_000,
    "s": "BTCUSDT",
    "k": {
        "t": 1_700_000_000_000,
        "T": 1_700_000_059_999,
        "s": "BTCUSDT",
        "i": "1m",
        "o": "100.0",
        "c": "100.5",
        "h": "101.0",
        "l": "99.5",
        "v": "12.5",
        "n": 42,
        "x": True,
        "q": "1250.0",
        "V": "6.0",
    },
}

BOOK_TICKER = {
    "e": "bookTicker",
    "u": 400900217,
    "E": 1_700_000_000_100,
    "T": 1_700_000_000_050,
    "s": "BTCUSDT",
    "b": "100.40",
    "B": "31.2",
    "a": "100.60",
    "A": "40.7",
}

MARK_PRICE_SINGLE = {
    "e": "markPriceUpdate",
    "E": 1_700_000_000_000,
    "s": "BTCUSDT",
    "p": "60000.0",
    "i": "59999.0",
    "P": "60001.0",
    "r": "0.0001",
    "T": 1_700_000_600_000,
}

# The array form. This is what `!markPrice@arr@1s` actually delivers.
MARK_PRICE_ARRAY = [
    MARK_PRICE_SINGLE,
    {**MARK_PRICE_SINGLE, "s": "ETHUSDT", "p": "3000.0", "r": "-0.00005"},
]

ORDER_TRADE_UPDATE = {
    "e": "ORDER_TRADE_UPDATE",
    "E": 1_700_000_000_000,
    "T": 1_700_000_000_000,
    "o": {
        "s": "BTCUSDT",
        "c": "tb-entry-1",
        "S": "BUY",
        "o": "LIMIT",
        "q": "0.01",
        "p": "60000",
        "ap": "60000.5",
        "sp": "0",
        "x": "TRADE",
        "X": "FILLED",
        "i": 8886774,
        "l": "0.01",
        "z": "0.01",
        "L": "60000.5",
        "n": "0.24",
        "N": "USDT",
        "T": 1_700_000_000_000,
        "m": False,
        "R": False,
        "rp": "0",
    },
}

ACCOUNT_UPDATE = {
    "e": "ACCOUNT_UPDATE",
    "E": 1_700_000_000_000,
    "T": 1_700_000_000_000,
    "a": {
        "m": "ORDER",
        "B": [{"a": "USDT", "wb": "75.5", "cw": "75.5", "bc": "0.5"}],
        "P": [
            {
                "s": "BTCUSDT",
                "pa": "0.01",
                "ep": "60000.0",
                "up": "1.25",
                "mt": "isolated",
                "ps": "BOTH",
            }
        ],
    },
}


# --------------------------------------------------------------------------- #
# event_type — the shape-first dispatch
# --------------------------------------------------------------------------- #
class TestEventType:
    def test_reads_mapping(self) -> None:
        assert parsers.event_type(KLINE) == "kline"

    def test_reads_array_from_its_first_element(self) -> None:
        """The regression: an array payload must not be treated as a mapping."""
        assert parsers.event_type(MARK_PRICE_ARRAY) == "markPriceUpdate"

    @pytest.mark.parametrize("payload", [None, [], "text", 42, [1, 2, 3], {}, [None], {"x": 1}])
    def test_never_raises_on_junk(self, payload: object) -> None:
        assert parsers.event_type(payload) == ""


class TestUnwrap:
    def test_combined_stream_envelope(self) -> None:
        stream, inner = parsers.unwrap({"stream": "btcusdt@kline_1m", "data": KLINE})
        assert stream == "btcusdt@kline_1m"
        assert inner == KLINE

    def test_raw_payload_passes_through(self) -> None:
        stream, inner = parsers.unwrap(KLINE)
        assert stream == ""
        assert inner == KLINE

    def test_array_payload_passes_through_intact(self) -> None:
        stream, inner = parsers.unwrap(MARK_PRICE_ARRAY)
        assert inner == MARK_PRICE_ARRAY

    def test_wrapped_array_payload(self) -> None:
        _, inner = parsers.unwrap({"stream": "!markPrice@arr@1s", "data": MARK_PRICE_ARRAY})
        assert isinstance(inner, list)
        assert len(inner) == 2


# --------------------------------------------------------------------------- #
# Mark price — both shapes
# --------------------------------------------------------------------------- #
class TestMarkPrice:
    def test_single_object(self) -> None:
        marks = parsers.parse_mark_price(MARK_PRICE_SINGLE)
        assert len(marks) == 1
        assert marks[0].symbol == "BTCUSDT"
        assert marks[0].mark_price == 60000.0
        assert marks[0].funding_rate == 0.0001
        assert marks[0].next_funding_time == 1_700_000_600_000

    def test_array_yields_every_symbol(self) -> None:
        marks = parsers.parse_mark_price(MARK_PRICE_ARRAY)
        assert [m.symbol for m in marks] == ["BTCUSDT", "ETHUSDT"]
        assert marks[1].funding_rate == -0.00005

    def test_always_returns_a_list(self) -> None:
        """Callers iterate unconditionally, so the empty case must be a list."""
        for junk in (None, "x", 7, [], {}, [{"bad": 1}]):
            assert parsers.parse_mark_price(junk) == []

    def test_one_bad_entry_does_not_lose_the_good_ones(self) -> None:
        marks = parsers.parse_mark_price([MARK_PRICE_SINGLE, {"e": "markPriceUpdate"}])
        assert [m.symbol for m in marks] == ["BTCUSDT"]


class TestKline:
    def test_parses_ohlcv(self) -> None:
        parsed = parsers.parse_kline(KLINE)
        assert parsed is not None
        symbol, interval, candle = parsed
        assert (symbol, interval) == ("BTCUSDT", "1m")
        assert (candle.open, candle.high, candle.low, candle.close) == (100.0, 101.0, 99.5, 100.5)
        assert candle.volume == 12.5
        assert candle.closed is True

    def test_forming_bar_is_marked_open(self) -> None:
        payload = {**KLINE, "k": {**KLINE["k"], "x": False}}
        parsed = parsers.parse_kline(payload)
        assert parsed is not None
        assert parsed[2].closed is False

    @pytest.mark.parametrize("payload", [None, [], {}, {"k": None}, {"k": []}, "x"])
    def test_junk_returns_none(self, payload: object) -> None:
        assert parsers.parse_kline(payload) is None


class TestBookTicker:
    def test_parses_quotes(self) -> None:
        book = parsers.parse_book_ticker(BOOK_TICKER)
        assert book is not None
        assert book.symbol == "BTCUSDT"
        assert book.bid_price == 100.40
        assert book.ask_price == 100.60
        assert book.mid == pytest.approx(100.50)

    @pytest.mark.parametrize("payload", [None, [], {}, {"s": "BTCUSDT"}, 5])
    def test_junk_returns_none(self, payload: object) -> None:
        assert parsers.parse_book_ticker(payload) is None


class TestUserEvents:
    def test_order_update(self) -> None:
        update = parsers.parse_order_update(ORDER_TRADE_UPDATE)
        assert update is not None
        assert update["symbol"] == "BTCUSDT"
        assert update["client_order_id"] == "tb-entry-1"
        assert update["status"] == "FILLED"
        assert update["filled_quantity"] == 0.01
        assert update["average_price"] == 60000.5
        assert update["commission"] == 0.24

    def test_account_update(self) -> None:
        account = parsers.parse_account_update(ACCOUNT_UPDATE)
        assert account is not None
        assert account["balances"][0]["wallet_balance"] == 75.5
        assert account["positions"][0]["symbol"] == "BTCUSDT"
        assert account["positions"][0]["unrealized_pnl"] == 1.25

    @pytest.mark.parametrize("payload", [None, [], {}, {"o": None}, "x"])
    def test_junk_order_update(self, payload: object) -> None:
        assert parsers.parse_order_update(payload) is None

    @pytest.mark.parametrize("payload", [None, [], {}, {"a": []}, "x"])
    def test_junk_account_update(self, payload: object) -> None:
        assert parsers.parse_account_update(payload) is None
