"""Symbol filter validation.

Binance rejects orders that violate a symbol's filters, and the rejections are
terse. Validating locally before transmission turns a production ``-1111`` into
a logged, explainable rejection with a reason code — and, crucially, prevents
the risk engine from believing it has a position it does not have.

Filters enforced (from ``exchangeInfo``):

* ``PRICE_FILTER``    — tickSize, minPrice, maxPrice
* ``LOT_SIZE``        — stepSize, minQty, maxQty (limit orders)
* ``MARKET_LOT_SIZE`` — the same, for market orders (often a lower maxQty)
* ``MIN_NOTIONAL``    — price × quantity floor
* ``PERCENT_PRICE``   — how far a limit price may sit from the mark price
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.core.mathutil import format_decimal, round_price, round_quantity
from tradebot.core.types import OrderType, SymbolInfo


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    quantity: float = 0.0
    price: float | None = None
    reason: str = ""
    detail: str = ""

    @classmethod
    def fail(cls, reason: str, detail: str = "") -> ValidationResult:
        return cls(ok=False, reason=reason, detail=detail)


def parse_symbol_info(entry: dict, brackets: list[dict] | None = None) -> SymbolInfo:
    """Build a :class:`SymbolInfo` from one ``exchangeInfo.symbols`` entry."""
    from tradebot.core.types import LeverageBracket

    filters = {f.get("filterType"): f for f in entry.get("filters", [])}
    price_filter = filters.get("PRICE_FILTER", {})
    lot = filters.get("LOT_SIZE", {})
    market_lot = filters.get("MARKET_LOT_SIZE", {})
    notional = filters.get("MIN_NOTIONAL", {})
    percent = filters.get("PERCENT_PRICE", {})

    parsed_brackets: tuple[LeverageBracket, ...] = ()
    if brackets:
        parsed_brackets = tuple(
            LeverageBracket(
                bracket=int(b.get("bracket", 0)),
                initial_leverage=int(b.get("initialLeverage", 1)),
                notional_cap=float(b.get("notionalCap", 0) or 0),
                notional_floor=float(b.get("notionalFloor", 0) or 0),
                maint_margin_ratio=float(b.get("maintMarginRatio", 0) or 0),
                cum=float(b.get("cum", 0) or 0),
            )
            for b in brackets
        )

    max_leverage = max((b.initial_leverage for b in parsed_brackets), default=20)

    return SymbolInfo(
        symbol=entry["symbol"],
        base_asset=entry.get("baseAsset", ""),
        quote_asset=entry.get("quoteAsset", ""),
        status=entry.get("status", entry.get("contractStatus", "")),
        contract_type=entry.get("contractType", ""),
        price_precision=int(entry.get("pricePrecision", 8)),
        quantity_precision=int(entry.get("quantityPrecision", 8)),
        tick_size=float(price_filter.get("tickSize", 0) or 0),
        step_size=float(lot.get("stepSize", 0) or 0),
        min_qty=float(lot.get("minQty", 0) or 0),
        max_qty=float(lot.get("maxQty", 0) or 0),
        min_notional=float(notional.get("notional", 0) or 0),
        market_min_qty=float(market_lot.get("minQty", 0) or 0),
        market_max_qty=float(market_lot.get("maxQty", 0) or 0),
        multiplier_up=float(percent.get("multiplierUp", 5) or 5),
        multiplier_down=float(percent.get("multiplierDown", 0.2) or 0.2),
        max_leverage=max_leverage,
        brackets=parsed_brackets,
        onboard_date=int(entry.get("onboardDate", 0) or 0),
    )


def validate_order(
    info: SymbolInfo,
    quantity: float,
    price: float | None,
    order_type: OrderType,
    reference_price: float | None = None,
) -> ValidationResult:
    """Round and validate an order against every relevant filter.

    Returns the ADJUSTED quantity and price on success. Callers must use the
    returned values, not the ones they passed in.
    """
    if quantity <= 0:
        return ValidationResult.fail("NON_POSITIVE_QUANTITY", f"quantity={quantity}")

    # -- quantity: always round DOWN so rounding can never enlarge risk ----- #
    adjusted_qty = round_quantity(quantity, info.step_size) if info.step_size > 0 else quantity
    if adjusted_qty <= 0:
        return ValidationResult.fail(
            "QUANTITY_ROUNDS_TO_ZERO",
            f"quantity={quantity} step_size={info.step_size}",
        )

    is_market = order_type in {
        OrderType.MARKET,
        OrderType.STOP_MARKET,
        OrderType.TAKE_PROFIT_MARKET,
        OrderType.TRAILING_STOP_MARKET,
    }
    min_qty = info.market_min_qty if (is_market and info.market_min_qty > 0) else info.min_qty
    max_qty = info.market_max_qty if (is_market and info.market_max_qty > 0) else info.max_qty

    if min_qty > 0 and adjusted_qty < min_qty:
        return ValidationResult.fail("BELOW_MIN_QTY", f"quantity={adjusted_qty} min_qty={min_qty}")
    if max_qty > 0 and adjusted_qty > max_qty:
        return ValidationResult.fail("ABOVE_MAX_QTY", f"quantity={adjusted_qty} max_qty={max_qty}")

    # -- price -------------------------------------------------------------- #
    adjusted_price = price
    if price is not None:
        if price <= 0:
            return ValidationResult.fail("NON_POSITIVE_PRICE", f"price={price}")
        adjusted_price = round_price(price, info.tick_size) if info.tick_size > 0 else price
        if adjusted_price <= 0:
            return ValidationResult.fail(
                "PRICE_ROUNDS_TO_ZERO", f"price={price} tick_size={info.tick_size}"
            )
        if reference_price and reference_price > 0:
            upper = reference_price * info.multiplier_up
            lower = reference_price * info.multiplier_down
            if not lower <= adjusted_price <= upper:
                return ValidationResult.fail(
                    "PERCENT_PRICE",
                    f"price={adjusted_price} allowed=[{lower:.10g},{upper:.10g}]",
                )

    # -- notional ----------------------------------------------------------- #
    notional_price = adjusted_price or reference_price
    if info.min_notional > 0:
        if notional_price is None:
            return ValidationResult.fail(
                "NO_PRICE_FOR_NOTIONAL",
                "a market order needs a reference price to check MIN_NOTIONAL",
            )
        notional = adjusted_qty * notional_price
        if notional < info.min_notional:
            return ValidationResult.fail(
                "BELOW_MIN_NOTIONAL",
                f"notional={notional:.8f} min_notional={info.min_notional}",
            )

    return ValidationResult(ok=True, quantity=adjusted_qty, price=adjusted_price)


def min_quantity_for_notional(info: SymbolInfo, price: float) -> float:
    """Smallest step-aligned quantity that satisfies MIN_NOTIONAL and minQty.

    This is what makes a 75 USDT account hard: on some symbols this quantity
    already represents more notional than the account may risk, and the correct
    answer is to skip the symbol rather than to oversize.
    """
    if price <= 0:
        return 0.0
    candidates = [info.min_qty]
    if info.min_notional > 0:
        candidates.append(info.min_notional / price)
    required = max(candidates)
    if info.step_size <= 0:
        return required
    # Round UP to the next step, then verify — rounding down would fall short.
    steps = required / info.step_size
    rounded = round_quantity(required, info.step_size)
    if rounded < required:
        rounded = round_quantity(required + info.step_size, info.step_size)
    _ = steps
    return rounded


def format_order_params(info: SymbolInfo, quantity: float, price: float | None) -> dict[str, str]:
    """Render quantity/price as the plain decimal strings Binance expects."""
    params = {"quantity": format_decimal(quantity, info.quantity_precision)}
    if price is not None:
        params["price"] = format_decimal(price, info.price_precision)
    return params
