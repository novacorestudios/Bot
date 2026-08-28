"""The order state machine.

Every order the engine submits is tracked here from creation to a terminal
state. Before this existed, an order's fate was inferred from whatever the
`place_order` call happened to return, and anything that arrived afterwards —
a partial fill, a cancellation by the exchange, an expiry — was discovered only
by the next reconciliation sweep, up to a minute later (AUDIT_REPORT.md M-5).

The legal transitions::

                          ┌──────────────┐
                          │   CREATED    │
                          └──────┬───────┘
                                 │ submit
                          ┌──────▼───────┐
              ┌───────────│  SUBMITTED   │───────────┐
              │           └──────┬───────┘           │
              │ reject/fail      │ ack               │ timeout
      ┌───────▼───────┐   ┌──────▼───────┐    ┌──────▼───────┐
      │REJECTED/FAILED│   │ ACKNOWLEDGED │    │ INDETERMINATE│
      └───────────────┘   └──────┬───────┘    └──────┬───────┘
                                 │                   │ resolved by
                    ┌────────────┼────────────┐      │ reconciliation
                    │            │            │      │
          ┌─────────▼──┐  ┌──────▼─────┐ ┌────▼──────▼──┐
          │PARTIALLY_  │─►│   FILLED   │ │  CANCELLED / │
          │  FILLED    │  └────────────┘ │   EXPIRED    │
          └─────┬──────┘                 └──────▲───────┘
                │      cancel requested          │
                └────────────────────────────────┘

Three rules the machine enforces, each of which corresponds to a real failure:

1. **A terminal state is final.** A late ``NEW`` arriving after a ``FILLED`` —
   which happens, because REST polling and the user stream race — must not
   resurrect the order.
2. **Filled quantity never decreases.** Out-of-order updates would otherwise
   walk a fill backwards and make the engine think it holds less than it does.
3. **INDETERMINATE is a real state, not an error.** A submission that timed out
   may or may not have reached the exchange. Guessing either way risks a
   duplicate position or an unprotected one, so it is recorded as unknown and
   entries stay blocked until reconciliation resolves it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tradebot.core.clock import Clock, SystemClock
from tradebot.core.logging import get_logger
from tradebot.core.types import OrderStatus

log = get_logger(__name__)


class OrderState(StrEnum):
    """Where an order is in its life, from our point of view."""

    CREATED = "CREATED"  # built locally, not yet sent
    SUBMITTED = "SUBMITTED"  # sent, no response yet
    ACKNOWLEDGED = "ACKNOWLEDGED"  # the exchange has it, resting
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"  # never reached the exchange
    INDETERMINATE = "INDETERMINATE"  # sent, outcome unknown

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def is_open(self) -> bool:
        """Still capable of filling — i.e. still exposes us to the market."""
        return self in _OPEN

    @property
    def needs_resolution(self) -> bool:
        """Blocks new entries until reconciliation settles it."""
        return self is OrderState.INDETERMINATE


_TERMINAL = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.FAILED,
    }
)

_OPEN = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_REQUESTED,
    }
)

#: What may follow what. Anything not listed is rejected and logged rather than
#: applied, because an illegal transition means our model of the order is wrong
#: and acting on a wrong model is how positions get lost.
TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset(
        {
            OrderState.SUBMITTED,
            OrderState.FAILED,  # rejected locally, e.g. by a filter check
        }
    ),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,  # a market order can fill before any ack
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.CANCELLED,
            OrderState.FAILED,
            OrderState.INDETERMINATE,
        }
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.REJECTED,
            OrderState.INDETERMINATE,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,  # each additional partial fill
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,  # cancelled with a partial fill standing
            OrderState.EXPIRED,
            OrderState.INDETERMINATE,
        }
    ),
    OrderState.CANCEL_REQUESTED: frozenset(
        {
            OrderState.CANCELLED,
            OrderState.FILLED,  # it filled before the cancel landed
            OrderState.PARTIALLY_FILLED,
            OrderState.EXPIRED,
            OrderState.INDETERMINATE,
        }
    ),
    # Reconciliation is the only thing that can move an order out of
    # INDETERMINATE, and it may resolve to any outcome.
    OrderState.INDETERMINATE: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
    OrderState.FAILED: frozenset(),
}

#: Binance order statuses mapped to our states. ``NEW`` is an acknowledgement,
#: not a creation: by the time we see it the exchange already has the order.
FROM_EXCHANGE: dict[str, OrderState] = {
    "NEW": OrderState.ACKNOWLEDGED,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
    "FILLED": OrderState.FILLED,
    "CANCELED": OrderState.CANCELLED,
    "CANCELLED": OrderState.CANCELLED,
    "REJECTED": OrderState.REJECTED,
    "EXPIRED": OrderState.EXPIRED,
    "EXPIRED_IN_MATCH": OrderState.EXPIRED,
    "PENDING": OrderState.SUBMITTED,
    "UNKNOWN": OrderState.INDETERMINATE,
    "NEW_INSURANCE": OrderState.ACKNOWLEDGED,
    "NEW_ADL": OrderState.ACKNOWLEDGED,
}


def state_for_status(status: OrderStatus | str) -> OrderState | None:
    """Translate an exchange status. Returns None for anything unrecognised."""
    key = status.value if isinstance(status, OrderStatus) else str(status)
    return FROM_EXCHANGE.get(key.upper())


@dataclass(slots=True)
class Transition:
    """One recorded state change, for the audit trail."""

    at_ms: int
    from_state: OrderState
    to_state: OrderState
    source: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "at_ms": self.at_ms,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(slots=True)
class TrackedOrder:
    """One order and everything we know about its progress."""

    client_order_id: str
    symbol: str
    side: str
    quantity: float
    purpose: str = "ENTRY"  # ENTRY | EXIT | STOP | TARGET
    state: OrderState = OrderState.CREATED
    exchange_order_id: str = ""
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    realized_pnl: float = 0.0
    created_ms: int = 0
    updated_ms: int = 0
    last_detail: str = ""
    history: list[Transition] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def fill_fraction(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    def age_sec(self, now_ms: int) -> float:
        return max(0.0, (now_ms - self.created_ms) / 1000.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "purpose": self.purpose,
            "state": self.state.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "fill_fraction": round(self.fill_fraction, 4),
            "average_price": self.average_price,
            "commission": self.commission,
            "detail": self.last_detail,
            "transitions": [t.as_dict() for t in self.history],
        }


class OrderTracker:
    """Owns the lifecycle of every order the engine has submitted.

    Updates arrive from three places that do not agree on ordering: the
    `place_order` response, the user data stream, and reconciliation. The
    tracker is what makes the result of interleaving them deterministic.
    """

    def __init__(self, clock: Clock | None = None, keep_terminal: int = 500) -> None:
        self.clock = clock or SystemClock()
        self.keep_terminal = keep_terminal
        self._orders: dict[str, TrackedOrder] = {}
        self._terminal_order: list[str] = []

        self.illegal_transitions = 0
        self.late_updates = 0
        self.unknown_orders = 0

    # ------------------------------------------------------------------ #
    def __contains__(self, client_order_id: object) -> bool:
        return client_order_id in self._orders

    def __len__(self) -> int:
        return len(self._orders)

    def get(self, client_order_id: str) -> TrackedOrder | None:
        return self._orders.get(client_order_id)

    @property
    def orders(self) -> dict[str, TrackedOrder]:
        return dict(self._orders)

    def open_orders(self) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.state.is_open]

    def open_for(self, symbol: str) -> list[TrackedOrder]:
        return [o for o in self.open_orders() if o.symbol == symbol]

    def indeterminate(self) -> list[TrackedOrder]:
        """Orders whose outcome we do not know. Entries stay blocked while any
        of these exist: we may or may not have a position."""
        return [o for o in self._orders.values() if o.state.needs_resolution]

    # ------------------------------------------------------------------ #
    def create(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        purpose: str = "ENTRY",
    ) -> TrackedOrder:
        """Register an order before it is sent.

        Registering first is deliberate: if the process dies between here and
        the exchange acknowledging, the id still exists to reconcile against.
        """
        existing = self._orders.get(client_order_id)
        if existing is not None:
            return existing

        now = self.clock.now_ms()
        order = TrackedOrder(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            purpose=purpose,
            created_ms=now,
            updated_ms=now,
        )
        self._orders[client_order_id] = order
        return order

    def transition(
        self,
        client_order_id: str,
        to_state: OrderState,
        source: str = "local",
        detail: str = "",
    ) -> bool:
        """Move an order to a new state. Returns False when refused."""
        order = self._orders.get(client_order_id)
        if order is None:
            self.unknown_orders += 1
            log.warning(
                "order_update_for_unknown_id",
                client_order_id=client_order_id,
                state=to_state.value,
                source=source,
                message="an order we never submitted, or one lost across a restart",
            )
            return False

        if order.state is to_state and to_state is not OrderState.PARTIALLY_FILLED:
            return True  # idempotent; duplicate updates are normal

        if order.state.is_terminal:
            # Rule 1: terminal is final. REST polling and the user stream race,
            # so a stale NEW arriving after a FILLED is expected, not an error.
            self.late_updates += 1
            log.debug(
                "late_update_after_terminal_state",
                client_order_id=client_order_id,
                current=order.state.value,
                attempted=to_state.value,
                source=source,
            )
            return False

        if to_state not in TRANSITIONS[order.state]:
            self.illegal_transitions += 1
            log.warning(
                "illegal_order_transition",
                client_order_id=client_order_id,
                symbol=order.symbol,
                current=order.state.value,
                attempted=to_state.value,
                source=source,
                detail=detail,
            )
            return False

        now = self.clock.now_ms()
        order.history.append(
            Transition(
                at_ms=now,
                from_state=order.state,
                to_state=to_state,
                source=source,
                detail=detail,
            )
        )
        order.state = to_state
        order.updated_ms = now
        if detail:
            order.last_detail = detail

        if to_state.is_terminal:
            self._record_terminal(client_order_id)

        log.debug(
            "order_state_changed",
            client_order_id=client_order_id,
            symbol=order.symbol,
            state=to_state.value,
            source=source,
        )
        return True

    # ------------------------------------------------------------------ #
    def apply_exchange_update(self, update: dict[str, Any], source: str = "user_stream") -> bool:
        """Apply a parsed ``ORDER_TRADE_UPDATE`` (or an equivalent REST poll).

        This is what makes fills known in milliseconds rather than at the next
        reconciliation sweep.
        """
        client_order_id = str(update.get("client_order_id") or "")
        if not client_order_id:
            return False

        order = self._orders.get(client_order_id)
        if order is None:
            self.unknown_orders += 1
            return False

        # Rule 2: quantities only ever move forward. Two updates for the same
        # order can arrive out of order; taking the max keeps our view of the
        # position from shrinking beneath us.
        filled = float(update.get("filled_quantity", 0.0) or 0.0)
        if filled >= order.filled_quantity:
            order.filled_quantity = filled
            average = float(update.get("average_price", 0.0) or 0.0)
            if average > 0:
                order.average_price = average
        else:
            self.late_updates += 1

        commission = float(update.get("commission", 0.0) or 0.0)
        if commission > 0:
            order.commission += commission
        order.realized_pnl += float(update.get("realized_pnl", 0.0) or 0.0)

        exchange_id = str(update.get("exchange_order_id") or "")
        if exchange_id:
            order.exchange_order_id = exchange_id

        status = str(update.get("status") or "")
        target = state_for_status(status)
        if target is None:
            log.debug("unmapped_order_status", status=status, client_order_id=client_order_id)
            return False

        return self.transition(client_order_id, target, source, detail=status)

    def resolve_indeterminate(
        self, client_order_id: str, to_state: OrderState, detail: str = ""
    ) -> bool:
        """Reconciliation settling an order whose outcome we did not know."""
        return self.transition(client_order_id, to_state, "reconciliation", detail)

    # ------------------------------------------------------------------ #
    def _record_terminal(self, client_order_id: str) -> None:
        """Keep a bounded history of completed orders.

        A long-running bot submits thousands of orders; holding them all is a
        slow memory leak that only shows up in production.
        """
        self._terminal_order.append(client_order_id)
        while len(self._terminal_order) > self.keep_terminal:
            oldest = self._terminal_order.pop(0)
            self._orders.pop(oldest, None)

    def stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for order in self._orders.values():
            by_state[order.state.value] = by_state.get(order.state.value, 0) + 1
        return {
            "tracked": len(self._orders),
            "open": len(self.open_orders()),
            "indeterminate": len(self.indeterminate()),
            "by_state": by_state,
            "illegal_transitions": self.illegal_transitions,
            "late_updates": self.late_updates,
            "unknown_orders": self.unknown_orders,
        }

    def report(self, limit: int = 50) -> list[dict[str, Any]]:
        """Open orders first, newest first — what an operator wants to see."""
        rows = sorted(
            self._orders.values(),
            key=lambda o: (o.state.is_terminal, -o.updated_ms),
        )
        return [o.as_dict() for o in rows[:limit]]
