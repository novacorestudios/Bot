"""WebSocket payload parsing.

Separated from the stream machinery so the parsing can be tested against the
documented payload shapes without opening a socket — which is precisely the test
that was missing when AUDIT_REPORT.md C-2 shipped.

**The rule that fixes C-2: dispatch on payload SHAPE before content.**

The previous parser began with ``data.get("e", "")``, which is correct for every
stream except one. ``!markPrice@arr@1s`` delivers a JSON **array**, so the very
first message raised ``AttributeError`` — and the list-handling branch written
further down was unreachable, because control never got there.

The lesson generalises: a parser for an external feed must establish what it is
holding before it assumes anything about the contents. Every function here
returns ``None`` for a payload it does not recognise rather than raising,
because a malformed or unexpected message must not take down the stream that
carries every other symbol's data.
"""

from __future__ import annotations

from typing import Any

from tradebot.core.logging import get_logger
from tradebot.core.types import BookTicker, Candle, MarkPriceInfo

log = get_logger(__name__)


def unwrap(payload: Any) -> tuple[str, Any]:
    """Unwrap a combined-stream envelope into ``(stream_name, data)``.

    Combined streams (``/stream?streams=a/b``) wrap every message as
    ``{"stream": ..., "data": ...}``; single streams (``/ws/<name>``) do not.
    """
    if isinstance(payload, dict) and "stream" in payload and "data" in payload:
        return str(payload["stream"]), payload["data"]
    return "", payload


def event_type(data: Any) -> str:
    """The event type of a payload, whatever shape it arrives in.

    Returns ``""`` for anything unrecognised. Critically, this NEVER assumes the
    payload is a mapping — that assumption is exactly what caused C-2.
    """
    if isinstance(data, dict):
        return str(data.get("e", ""))
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            # An array stream is identified by the event type of its members.
            return str(first.get("e", ""))
    return ""


def is_array_payload(data: Any) -> bool:
    """True when the payload is an array of events (e.g. ``!markPrice@arr``)."""
    return isinstance(data, list)


# --------------------------------------------------------------------------- #
def parse_kline(data: Any, now_ms: int = 0) -> tuple[str, str, Candle] | None:
    """Parse a ``<symbol>@kline_<interval>`` payload.

    Returns ``(symbol, interval, candle)``. The ``x`` field marks whether the
    bar has closed, and that flag is what keeps strategies from reading a
    forming bar.
    """
    if not isinstance(data, dict):
        return None
    kline = data.get("k")
    if not isinstance(kline, dict):
        return None
    try:
        candle = Candle(
            open_time=int(kline["t"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            close_time=int(kline["T"]),
            quote_volume=float(kline.get("q", 0) or 0),
            trades=int(kline.get("n", 0) or 0),
            taker_buy_volume=float(kline.get("V", 0) or 0),
            closed=bool(kline.get("x", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("kline_parse_failed", error=str(exc))
        return None
    _ = now_ms
    return str(data.get("s", "")), str(kline.get("i", "")), candle


def parse_book_ticker(data: Any) -> BookTicker | None:
    """Parse a ``<symbol>@bookTicker`` payload."""
    if not isinstance(data, dict):
        return None
    try:
        return BookTicker(
            symbol=str(data["s"]),
            bid_price=float(data["b"]),
            bid_qty=float(data["B"]),
            ask_price=float(data["a"]),
            ask_qty=float(data["A"]),
            timestamp=int(data.get("E", 0) or 0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("book_ticker_parse_failed", error=str(exc))
        return None


def parse_mark_price(data: Any) -> list[MarkPriceInfo]:
    """Parse a mark-price payload, in EITHER of its two shapes.

    ``<symbol>@markPrice`` delivers a single object; ``!markPrice@arr@1s``
    delivers an array of them. Always returns a list — one element, many, or
    empty — so callers need no shape check of their own. This is the fix for
    C-2: the shape is resolved here, once, instead of at every call site.
    """
    entries = data if isinstance(data, list) else [data]
    out: list[MarkPriceInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                MarkPriceInfo(
                    symbol=str(entry["s"]),
                    mark_price=float(entry["p"]),
                    index_price=float(entry.get("i", 0) or 0),
                    funding_rate=float(entry.get("r", 0) or 0),
                    next_funding_time=int(entry.get("T", 0) or 0),
                    timestamp=int(entry.get("E", 0) or 0),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("mark_price_parse_failed", error=str(exc), symbol=entry.get("s"))
            continue
    return out


def parse_agg_trade(data: Any) -> dict[str, Any] | None:
    """Parse an ``<symbol>@aggTrade`` payload."""
    if not isinstance(data, dict):
        return None
    try:
        return {
            "symbol": str(data["s"]),
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "timestamp": int(data.get("T", 0) or 0),
            # 'm' is true when the BUYER is the maker, i.e. a seller took the bid.
            "buyer_is_maker": bool(data.get("m", False)),
        }
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# User data stream
# --------------------------------------------------------------------------- #
def parse_order_update(data: Any) -> dict[str, Any] | None:
    """Parse an ``ORDER_TRADE_UPDATE`` event.

    This is what lets fills be known in milliseconds instead of being discovered
    by the next reconciliation sweep.
    """
    if not isinstance(data, dict):
        return None
    order = data.get("o")
    if not isinstance(order, dict):
        return None
    try:
        return {
            "symbol": str(order["s"]),
            "client_order_id": str(order.get("c", "")),
            "exchange_order_id": str(order.get("i", "")),
            "side": str(order.get("S", "")),
            "order_type": str(order.get("o", "")),
            "status": str(order.get("X", "")),
            "execution_type": str(order.get("x", "")),
            "quantity": float(order.get("q", 0) or 0),
            "price": float(order.get("p", 0) or 0),
            "average_price": float(order.get("ap", 0) or 0),
            "stop_price": float(order.get("sp", 0) or 0),
            "filled_quantity": float(order.get("z", 0) or 0),
            "last_filled_quantity": float(order.get("l", 0) or 0),
            "last_filled_price": float(order.get("L", 0) or 0),
            "commission": float(order.get("n", 0) or 0),
            "commission_asset": str(order.get("N", "") or ""),
            "realized_pnl": float(order.get("rp", 0) or 0),
            "is_maker": bool(order.get("m", False)),
            "reduce_only": bool(order.get("R", False)),
            "timestamp": int(data.get("E", 0) or 0),
            "trade_time": int(order.get("T", 0) or 0),
        }
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("order_update_parse_failed", error=str(exc))
        return None


def parse_account_update(data: Any) -> dict[str, Any] | None:
    """Parse an ``ACCOUNT_UPDATE`` event: balance and position changes."""
    if not isinstance(data, dict):
        return None
    payload = data.get("a")
    if not isinstance(payload, dict):
        return None

    balances = []
    for entry in payload.get("B", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            balances.append(
                {
                    "asset": str(entry["a"]),
                    "wallet_balance": float(entry.get("wb", 0) or 0),
                    "cross_wallet_balance": float(entry.get("cw", 0) or 0),
                    "balance_change": float(entry.get("bc", 0) or 0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    positions = []
    for entry in payload.get("P", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            positions.append(
                {
                    "symbol": str(entry["s"]),
                    "position_amount": float(entry.get("pa", 0) or 0),
                    "entry_price": float(entry.get("ep", 0) or 0),
                    "unrealized_pnl": float(entry.get("up", 0) or 0),
                    "margin_type": str(entry.get("mt", "") or ""),
                    "position_side": str(entry.get("ps", "BOTH") or "BOTH"),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    return {
        "reason": str(payload.get("m", "")),
        "balances": balances,
        "positions": positions,
        "timestamp": int(data.get("E", 0) or 0),
    }
