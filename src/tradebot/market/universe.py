"""Tradable universe discovery.

Builds the set of symbols the scanner is allowed to consider, from
``exchangeInfo`` plus hard filters. Nothing here is hardcoded to a particular
coin: the universe is whatever Binance currently lists as a tradable USDT
perpetual, minus symbols the account demonstrably cannot trade.

That last part matters more than it sounds for a 75 USDT account. A symbol whose
``MIN_NOTIONAL`` is 20 USDT forces a position size that, at the configured
leverage, may represent more risk than the account permits. Discovering that at
order-placement time wastes a cycle and an order-rate slot; discovering it here
removes the symbol from consideration entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.core.config import ScannerConfig
from tradebot.core.logging import get_logger
from tradebot.core.types import SymbolInfo, Ticker24h
from tradebot.exchange.binance.filters import min_quantity_for_notional

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    symbol: str
    info: SymbolInfo
    quote_volume_24h: float
    last_price: float
    price_change_24h: float


@dataclass(frozen=True, slots=True)
class UniverseReport:
    """The universe plus why everything else was excluded — this is auditable."""

    entries: tuple[UniverseEntry, ...]
    excluded: dict[str, str]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(entry.symbol for entry in self.entries)

    def exclusion_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason in self.excluded.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts


class UniverseBuilder:
    """Applies hard filters to produce the scannable symbol set."""

    def __init__(
        self, config: ScannerConfig, equity: float = 75.0, max_min_notional_ratio: float = 0.5
    ) -> None:
        self.config = config
        self.equity = equity
        self.max_min_notional_ratio = max_min_notional_ratio

    def build(
        self, symbols: dict[str, SymbolInfo], tickers: dict[str, Ticker24h]
    ) -> UniverseReport:
        cfg = self.config
        entries: list[UniverseEntry] = []
        excluded: dict[str, str] = {}

        allow = {s.upper() for s in cfg.allow_list}
        deny = {s.upper() for s in cfg.deny_list}

        for symbol, info in symbols.items():
            if deny and symbol in deny:
                excluded[symbol] = "DENY_LIST"
                continue
            if allow and symbol not in allow:
                excluded[symbol] = "NOT_IN_ALLOW_LIST"
                continue
            if info.quote_asset != cfg.quote_asset:
                excluded[symbol] = "WRONG_QUOTE_ASSET"
                continue
            if info.contract_type != cfg.contract_type:
                excluded[symbol] = "WRONG_CONTRACT_TYPE"
                continue
            if info.status != "TRADING":
                excluded[symbol] = f"STATUS_{info.status or 'UNKNOWN'}"
                continue

            ticker = tickers.get(symbol)
            if ticker is None:
                excluded[symbol] = "NO_TICKER_DATA"
                continue
            if ticker.last_price < cfg.min_price:
                excluded[symbol] = "PRICE_BELOW_MINIMUM"
                continue
            if ticker.quote_volume < cfg.min_24h_quote_volume:
                excluded[symbol] = "INSUFFICIENT_24H_VOLUME"
                continue

            # Can this account form a valid position at all?
            reason = self._affordability_problem(info, ticker.last_price)
            if reason:
                excluded[symbol] = reason
                continue

            entries.append(
                UniverseEntry(
                    symbol=symbol,
                    info=info,
                    quote_volume_24h=ticker.quote_volume,
                    last_price=ticker.last_price,
                    price_change_24h=ticker.price_change_pct,
                )
            )

        entries.sort(key=lambda e: e.quote_volume_24h, reverse=True)

        report = UniverseReport(tuple(entries), excluded)
        log.info(
            "universe_built",
            tradable=len(entries),
            excluded=len(excluded),
            reasons=report.exclusion_counts(),
        )
        return report

    def _affordability_problem(self, info: SymbolInfo, price: float) -> str | None:
        """Reject symbols whose smallest valid position is too big for this account.

        Returns a reason string, or None when the symbol is usable.
        """
        if price <= 0:
            return "NO_PRICE"

        min_qty = min_quantity_for_notional(info, price)
        if min_qty <= 0:
            return "NO_VALID_QUANTITY"

        min_notional = min_qty * price
        # The minimum position must fit inside the account's notional budget.
        # Compared against equity, not margin: leverage does not make a position
        # smaller, only cheaper to hold.
        if min_notional > self.equity * self.max_min_notional_ratio * max(1, info.max_leverage):
            return "MIN_NOTIONAL_TOO_LARGE_FOR_ACCOUNT"
        return None

    def update_equity(self, equity: float) -> None:
        """Equity changes as the account grows or shrinks; the universe follows."""
        self.equity = equity
