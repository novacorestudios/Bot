"""Exact arithmetic for exchange-facing values.

Binance rejects any price that is not an exact multiple of ``tickSize`` and any
quantity that is not an exact multiple of ``stepSize``. Binary floating point
cannot represent most of those steps exactly (0.1, 0.001, ...), so every value
that will be transmitted is rounded through ``Decimal`` and rendered as a plain
decimal string. Doing this with ``round()`` produces values Binance rejects with
error -1111 roughly at random, which is very hard to debug in production.
"""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal, InvalidOperation


def _dec(value: float | str | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    # str() of a float gives the shortest repr that round-trips, which is what
    # we want; Decimal(float) would drag in the binary representation error.
    return Decimal(str(value))


def round_to_step(value: float, step: float, mode: str = "down") -> float:
    """Round ``value`` to a multiple of ``step``.

    ``mode`` is ``down`` (default, safe for quantities), ``up`` or ``nearest``.
    A non-positive step returns the value unchanged.
    """
    if step <= 0:
        return value
    try:
        d_value, d_step = _dec(value), _dec(step)
        quotient = d_value / d_step
        rounding = {"down": ROUND_DOWN, "up": ROUND_UP, "nearest": ROUND_HALF_UP}[mode]
        return float((quotient.quantize(Decimal("1"), rounding=rounding)) * d_step)
    except (InvalidOperation, KeyError, ZeroDivisionError):
        return value


def round_price(price: float, tick_size: float, mode: str = "nearest") -> float:
    """Round a price to the symbol's tick size."""
    return round_to_step(price, tick_size, mode)


def round_quantity(quantity: float, step_size: float) -> float:
    """Round a quantity DOWN to the symbol's step size.

    Always down: rounding a quantity up can breach a risk limit or the available
    margin, while rounding down can only ever risk less than intended.
    """
    return round_to_step(quantity, step_size, "down")


def format_decimal(value: float, precision: int) -> str:
    """Render a value for transmission: fixed precision, no scientific notation.

    Binance rejects ``1e-05``; it wants ``0.00001``.
    """
    quant = Decimal(1).scaleb(-precision)
    d = _dec(value).quantize(quant, rounding=ROUND_DOWN)
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def decimals_for_step(step: float) -> int:
    """Number of decimal places implied by a step/tick size."""
    d = _dec(step).normalize()
    exponent = d.as_tuple().exponent
    return max(0, -int(exponent))


def clamp(value: float, low: float, high: float) -> float:
    """Constrain a value to [low, high]. Tolerates reversed bounds."""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns ``default`` instead of raising or producing inf/nan."""
    if denominator == 0 or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def pct_change(old: float, new: float) -> float:
    """Fractional change from ``old`` to ``new``; 0.0 when old is unusable."""
    return safe_div(new - old, abs(old), 0.0)


def bps(fraction: float) -> float:
    """Convert a fraction to basis points."""
    return fraction * 10_000.0


def from_bps(basis_points: float) -> float:
    """Convert basis points to a fraction."""
    return basis_points / 10_000.0


def normalise_score(value: float, low: float, high: float, invert: bool = False) -> float:
    """Map ``value`` onto 0..100 across [low, high], clamped.

    ``invert=True`` means lower input is better (spread, cost).
    """
    if high == low:
        return 50.0
    ratio = (value - low) / (high - low)
    ratio = clamp(ratio, 0.0, 1.0)
    if invert:
        ratio = 1.0 - ratio
    return ratio * 100.0


def band_score(value: float, low: float, high: float, falloff: float = 2.0) -> float:
    """Score a value by how well it sits inside a preferred band.

    100 inside [low, high], decaying outside. Used for volatility, where both
    too little and too much are bad.
    """
    if low > high:
        low, high = high, low
    if low <= value <= high:
        return 100.0
    width = high - low
    if width <= 0:
        return 0.0
    distance = (low - value) if value < low else (value - high)
    decay = distance / (width * falloff)
    return clamp(100.0 * (1.0 - decay), 0.0, 100.0)


def geometric_mean(values: list[float]) -> float:
    """Geometric mean of positive values; 0.0 if any value is non-positive."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))
