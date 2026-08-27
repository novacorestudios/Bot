"""Exception hierarchy.

Every exception carries enough context to be logged structurally, and none of
them ever carry an API secret.
"""

from __future__ import annotations

from typing import Any


class TradeBotError(Exception):
    """Base class for all errors raised by this package."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        extras = " ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({extras})"


class ConfigError(TradeBotError):
    """Configuration is missing, malformed or internally inconsistent."""


class SafetyError(ConfigError):
    """A safety precondition for the requested mode is not satisfied."""


# --------------------------------------------------------------------------- #
# Exchange errors
# --------------------------------------------------------------------------- #
class ExchangeError(TradeBotError):
    """Any failure interacting with the exchange."""

    retryable: bool = False


class NetworkError(ExchangeError):
    """Transport-level failure. The request may or may not have been applied."""

    retryable = True


class TimeoutError_(NetworkError):
    """Request timed out. State is INDETERMINATE — query before retrying."""


class RateLimitError(ExchangeError):
    """429/418. Carries the retry delay when the exchange supplies one."""

    retryable = True

    def __init__(
        self, message: str, retry_after: float = 1.0, banned: bool = False, **context: Any
    ) -> None:
        super().__init__(message, **context)
        self.retry_after = retry_after
        self.banned = banned


class AuthenticationError(ExchangeError):
    """Bad key, bad signature or insufficient permissions. Never retryable."""


class ClockSkewError(ExchangeError):
    """Binance -1021: timestamp outside recvWindow. Resync then retry once."""

    retryable = True


class OrderRejectedError(ExchangeError):
    """The exchange refused the order. Not retryable without changing it."""

    def __init__(self, message: str, code: int | None = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.code = code


class InsufficientMarginError(OrderRejectedError):
    """Binance -2019 and friends."""


class FilterViolationError(OrderRejectedError):
    """Order violates a symbol filter (tick size, step size, min notional...)."""


class UnknownOrderError(ExchangeError):
    """Binance -2013: order does not exist. Meaningful during reconciliation."""


# --------------------------------------------------------------------------- #
# Domain errors
# --------------------------------------------------------------------------- #
class RiskViolationError(TradeBotError):
    """An action was attempted that the risk rules forbid. Always a bug."""


class ReconciliationError(TradeBotError):
    """Local state could not be reconciled with the exchange."""


class DataError(TradeBotError):
    """Market data is missing, stale or malformed."""


class StrategyError(TradeBotError):
    """A strategy raised. The strategy is isolated, never the whole engine."""
