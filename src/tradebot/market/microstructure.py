"""Execution-cost modelling.

The whole system turns on one question: *after everything it costs to get in and
out, is this trade still worth taking?* This module answers the cost half.

Costs of a round trip, as fractions of notional:

* **Fees** — taker or maker, per side. Verify your actual rate: the default
  0.04 % taker assumes no VIP tier and no BNB discount.
* **Spread** — a marketable order crosses it. We charge half the quoted spread
  per side by default, which is what a marketable order pays against the mid.
* **Slippage** — a base component plus a market-impact term that grows with
  order size relative to available depth. Small accounts have small impact; the
  base term dominates.
* **Funding** — charged every 8 hours on perpetuals. A 10-minute trade usually
  avoids it entirely; a trade held across the timestamp pays in full, and a
  short receives it when funding is positive.

Every number here is an *estimate*. Realised slippage is recorded per trade and
compared against these estimates, so the model can be corrected with evidence
rather than guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.core.config import EdgeConfig
from tradebot.core.mathutil import from_bps, safe_div
from tradebot.core.types import BookTicker, CostEstimate, Direction, OrderBook


@dataclass(frozen=True, slots=True)
class LiquiditySnapshot:
    """What we know about a symbol's tradability right now."""

    symbol: str
    spread_bps: float
    bid_notional: float
    ask_notional: float
    book_imbalance: float
    quote_volume_24h: float

    @property
    def depth_notional(self) -> float:
        """Notional resting near the touch, both sides averaged."""
        return (self.bid_notional + self.ask_notional) / 2.0

    def depth_for(self, direction: Direction) -> float:
        """Depth on the side we would consume."""
        if direction is Direction.LONG:
            return self.ask_notional  # a buy lifts asks
        if direction is Direction.SHORT:
            return self.bid_notional
        return self.depth_notional


def snapshot_from_book(
    symbol: str,
    book: BookTicker | None,
    depth: OrderBook | None = None,
    quote_volume_24h: float = 0.0,
) -> LiquiditySnapshot:
    """Build a snapshot from whatever market data is available."""
    if book is None:
        return LiquiditySnapshot(symbol, float("inf"), 0.0, 0.0, 0.0, quote_volume_24h)

    if depth is not None and depth.bids and depth.asks:
        bid_notional, ask_notional = depth.notional_within(0.002)
        imbalance = depth.imbalance
    else:
        # Best-bid/ask only: the touch quantity is all we can see.
        bid_notional = book.bid_price * book.bid_qty
        ask_notional = book.ask_price * book.ask_qty
        imbalance = book.imbalance

    return LiquiditySnapshot(
        symbol=symbol,
        spread_bps=book.spread_bps,
        bid_notional=bid_notional,
        ask_notional=ask_notional,
        book_imbalance=imbalance,
        quote_volume_24h=quote_volume_24h,
    )


class CostModel:
    """Estimates the cost of a round trip before it is taken."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config

    # -- components --------------------------------------------------------- #
    def fee_rate(self, taker: bool) -> float:
        return self.config.taker_fee if taker else self.config.maker_fee

    def spread_cost(self, spread_bps: float, sides: int = 2) -> float:
        """Cost of crossing the spread, as a fraction of notional.

        A marketable order pays roughly half the spread relative to the mid on
        each side, so a round trip pays about one full spread.
        """
        if spread_bps <= 0 or spread_bps == float("inf"):
            return 0.0
        return from_bps(spread_bps) * self.config.spread_cost_fraction * sides

    def slippage(self, notional: float, depth_notional: float) -> float:
        """Base slippage plus a size-dependent impact term.

        Impact grows with the fraction of visible depth the order consumes. With
        a 75 USDT account on a liquid perpetual this term is negligible, which
        is the one genuine advantage of trading small.
        """
        base = from_bps(self.config.base_slippage_bps)
        if depth_notional <= 0:
            # No depth information: assume the order is significant. Being
            # pessimistic here rejects trades; being optimistic loses money.
            return base * 3.0
        participation = safe_div(notional, depth_notional, 1.0)
        return base + self.config.impact_coefficient * base * participation

    def funding_cost(
        self,
        direction: Direction,
        funding_rate: float,
        expected_duration_sec: float,
        seconds_to_funding: float,
    ) -> float:
        """Funding paid (positive) or received (negative) over the holding period.

        A long pays when funding is positive; a short receives it. If the trade
        is expected to close before the next funding timestamp, it is zero —
        which is the usual case for a sub-hour scalp, and a real advantage of
        short holding periods.
        """
        if funding_rate == 0.0 or direction is Direction.WAIT:
            return 0.0
        if seconds_to_funding > 0 and expected_duration_sec < seconds_to_funding:
            return 0.0
        interval_sec = self.config.funding_interval_hours * 3600
        periods = max(1.0, expected_duration_sec / interval_sec)
        return funding_rate * periods * direction.sign

    # -- assembly ----------------------------------------------------------- #
    def estimate(
        self,
        direction: Direction,
        notional: float,
        liquidity: LiquiditySnapshot,
        funding_rate: float = 0.0,
        expected_duration_sec: float = 600.0,
        seconds_to_funding: float = float("inf"),
        taker_entry: bool | None = None,
        taker_exit: bool | None = None,
    ) -> CostEstimate:
        """Full round-trip cost as fractions of notional."""
        entry_taker = self.config.assume_taker_entry if taker_entry is None else taker_entry
        exit_taker = self.config.assume_taker_exit if taker_exit is None else taker_exit

        depth = liquidity.depth_for(direction)

        return CostEstimate(
            entry_fee=self.fee_rate(entry_taker),
            exit_fee=self.fee_rate(exit_taker),
            # Only marketable sides pay the spread.
            spread_cost=self.spread_cost(
                liquidity.spread_bps, sides=int(entry_taker) + int(exit_taker)
            ),
            slippage=self.slippage(notional, depth) * 2,  # entry and exit
            funding=self.funding_cost(
                direction, funding_rate, expected_duration_sec, seconds_to_funding
            ),
        )

    def breakeven_move(self, liquidity: LiquiditySnapshot, notional: float = 0.0) -> float:
        """Minimum favourable price move, as a fraction, just to break even.

        Surfaced on the dashboard because it makes the scalping problem concrete:
        if this is 0.15 % and the strategy's average winner is 0.12 %, the
        strategy cannot work no matter how good its hit rate looks.
        """
        estimate = self.estimate(
            Direction.LONG,
            notional or 1000.0,
            liquidity,
            funding_rate=0.0,
            expected_duration_sec=0.0,
        )
        return estimate.total
