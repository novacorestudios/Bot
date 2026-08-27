"""Binance error-code mapping.

Binance returns ``{"code": -2019, "msg": "Margin is insufficient."}`` with an
HTTP 400. Treating every 400 the same is how a bot ends up retrying an order
that can never succeed, or giving up on one that only needed a clock resync.
This module classifies each code so callers can react correctly.

Codes are from the official "Error codes for Binance USDⓈ-M Futures" page.
"""

from __future__ import annotations

from tradebot.core.errors import (
    AuthenticationError,
    ClockSkewError,
    ExchangeError,
    FilterViolationError,
    InsufficientMarginError,
    OrderRejectedError,
    RateLimitError,
    UnknownOrderError,
)

# -- classification ---------------------------------------------------------- #
# Retrying these can succeed unchanged.
RETRYABLE = {
    -1000,  # UNKNOWN — internal error
    -1001,  # DISCONNECTED
    -1007,  # TIMEOUT — status unknown, MUST query before re-sending
    -1016,  # SERVICE_SHUTTING_DOWN
}

# The clock drifted; resync and retry once.
CLOCK_SKEW = {-1021}

# Never retryable without operator action.
AUTH_ERRORS = {
    -1002,  # UNAUTHORIZED
    -1022,  # INVALID_SIGNATURE
    -2014,  # BAD_API_KEY_FMT
    -2015,  # REJECTED_MBX_KEY (bad key, IP or permissions)
    -2008,  # BAD_API_ID
}

# The order itself is wrong; retrying identically is pointless.
FILTER_ERRORS = {
    -1013,  # INVALID_MESSAGE / filter failure
    -1111,  # BAD_PRECISION
    -4003,  # QTY_LESS_THAN_ZERO
    -4004,  # QTY_LESS_THAN_MIN
    -4005,  # QTY_GREATER_THAN_MAX
    -4014,  # PRICE_NOT_INCREASED_BY_TICK
    -4015,  # INVALID_CL_ORD_ID_LEN
    -4055,  # AMOUNT_MUST_BE_POSITIVE
    -4164,  # MIN_NOTIONAL — order notional below the symbol minimum
    -1121,  # BAD_SYMBOL
}

MARGIN_ERRORS = {
    -2018,  # BALANCE_NOT_SUFFICIENT
    -2019,  # MARGIN_NOT_SUFFICIENT
    -4131,  # counterparty best price does not meet PERCENT_PRICE
    -2027,  # EXCEED_MAX_NOTIONAL_VALUE
    -2028,  # leverage reduction required
}

UNKNOWN_ORDER = {
    -2011,  # CANCEL_REJECTED / unknown order
    -2013,  # NO_SUCH_ORDER
}

RATE_LIMIT = {-1003, -1015}

# A duplicate clientOrderId means an earlier submission already landed. This is
# NOT a failure: the caller should query that order by its id and adopt it,
# which is exactly what makes retries safe.
DUPLICATE_CLIENT_ORDER_ID = {-4015, -2010}

REDUCE_ONLY_REJECTED = {-2022}  # ReduceOnly Order is rejected
POSITION_SIDE_MISMATCH = {-4061}
NO_NEED_TO_CHANGE_LEVERAGE = {-4028, -4046, -4047}


def is_retryable(code: int) -> bool:
    return code in RETRYABLE


def is_duplicate(code: int, message: str = "") -> bool:
    """True when the exchange is telling us this order already exists.

    Binance signals this with -4015/-2010 and a message mentioning a duplicate
    client order id; both are checked because the message wording has varied.
    """
    if code in DUPLICATE_CLIENT_ORDER_ID:
        lowered = message.lower()
        if "duplicate" in lowered or "already exist" in lowered:
            return True
    return "duplicate" in message.lower() and "clientorderid" in message.lower()


def raise_for_code(code: int, message: str, endpoint: str = "", status: int = 400) -> None:
    """Translate a Binance error payload into the right exception type."""
    context = {"code": code, "endpoint": endpoint, "http_status": status}
    # OrderRejectedError takes `code` as a named argument, so it must not also
    # arrive inside **context.
    order_context = {"endpoint": endpoint, "http_status": status}

    if code in CLOCK_SKEW:
        raise ClockSkewError(f"timestamp outside recvWindow: {message}", **context)
    if code in AUTH_ERRORS:
        raise AuthenticationError(
            f"authentication or permission failure: {message}. Check the API key, "
            f"its IP allow-list, and that futures trading is enabled.",
            **context,
        )
    if code in RATE_LIMIT:
        # order_context, not context: `context` carries a `code` key and
        # RateLimitError takes `banned` positionally after retry_after, so
        # splatting the full mapping risks a silent argument collision.
        raise RateLimitError(
            f"rate limited: {message}", retry_after=5.0, banned=False, **order_context
        )
    if code in MARGIN_ERRORS:
        raise InsufficientMarginError(f"insufficient margin: {message}", code=code, **order_context)
    if code in FILTER_ERRORS:
        raise FilterViolationError(
            f"order violates a symbol filter: {message}", code=code, **order_context
        )
    if code in UNKNOWN_ORDER:
        raise UnknownOrderError(f"order not found: {message}", **context)
    if code in RETRYABLE:
        error = ExchangeError(f"transient exchange error: {message}", **context)
        error.retryable = True
        raise error

    raise OrderRejectedError(f"binance error {code}: {message}", code=code, **order_context)


def describe(code: int) -> str:
    """Human-readable classification, used in risk-event logs."""
    if code in CLOCK_SKEW:
        return "clock skew"
    if code in AUTH_ERRORS:
        return "authentication/permission"
    if code in RATE_LIMIT:
        return "rate limit"
    if code in MARGIN_ERRORS:
        return "insufficient margin"
    if code in FILTER_ERRORS:
        return "symbol filter violation"
    if code in UNKNOWN_ORDER:
        return "unknown order"
    if code in RETRYABLE:
        return "transient"
    return "rejected"
