from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_CEILING
from enum import StrEnum
from typing import Iterable

from kaishi_bot.domain import Side
from kaishi_bot.fees import FeeSchedule


ZERO = Decimal("0")
CENTICENT = Decimal("0.0001")


class ReplayAction(StrEnum):
    WAIT = "wait"
    BUY_UP = "buy_up"
    BUY_DOWN = "buy_down"
    HOLD = "hold"
    ADD = "add"
    EXIT_HALF = "exit_half"
    EXIT_ALL = "exit_all"


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    available_at: datetime
    stream_priority: int
    sequence: int
    stable_row_id: str
    kind: str
    payload: object


class PointInTimeEventStream:
    def __init__(self, events: Iterable[ReplayEvent]) -> None:
        self.events = sorted(events, key=lambda item: (
            item.available_at, item.stream_priority, item.sequence,
            item.stable_row_id,
        ))
        self.cursor = 0

    def reset(self) -> None:
        self.cursor = 0

    def visible_until(self, timestamp: datetime) -> tuple[ReplayEvent, ...]:
        visible: list[ReplayEvent] = []
        while (
            self.cursor < len(self.events)
            and self.events[self.cursor].available_at <= timestamp
        ):
            visible.append(self.events[self.cursor])
            self.cursor += 1
        return tuple(visible)


@dataclass(frozen=True, slots=True)
class ExecutionLevel:
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True, slots=True)
class ExecutableBook:
    ticker: str
    available_at: datetime
    up_asks: tuple[ExecutionLevel, ...] = ()
    up_bids: tuple[ExecutionLevel, ...] = ()
    down_asks: tuple[ExecutionLevel, ...] = ()
    down_bids: tuple[ExecutionLevel, ...] = ()
    complete: bool = True


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    decision_ms: int = 0
    network_ms: int = 0
    exchange_ms: int = 0

    @property
    def total_ms(self) -> int:
        return self.decision_ms + self.network_ms + self.exchange_ms


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    ticker: str
    side: Side
    operation: str
    decision_at: datetime
    arrival_at: datetime
    requested_quantity: Decimal
    filled_quantity: Decimal
    unfilled_quantity: Decimal
    vwap: Decimal | None
    fee: Decimal
    gross: Decimal
    levels: tuple[ExecutionLevel, ...]
    reason: str

    @property
    def fill_ratio(self) -> Decimal:
        return (
            self.filled_quantity / self.requested_quantity
            if self.requested_quantity > 0 else ZERO
        )


def _order_fee(
    schedule: FeeSchedule, fills: Iterable[ExecutionLevel], *, multiplier: Decimal,
) -> Decimal:
    if schedule.fee_type != "quadratic":
        raise ValueError(f"unsupported fee type: {schedule.fee_type}")
    raw = sum((
        schedule.taker_rate * schedule.multiplier * multiplier
        * item.quantity * item.price * (Decimal("1") - item.price)
        for item in fills
    ), ZERO)
    return raw.quantize(CENTICENT, rounding=ROUND_CEILING)


class ExecutionSimulator:
    def __init__(self, books: Iterable[ExecutableBook]) -> None:
        self.books = sorted(books, key=lambda item: item.available_at)

    def book_at(self, ticker: str, arrival_at: datetime) -> ExecutableBook | None:
        candidates = [
            book for book in self.books
            if book.ticker == ticker and book.available_at <= arrival_at
        ]
        return candidates[-1] if candidates else None

    def execute(
        self, *, ticker: str, side: Side, operation: str,
        quantity: Decimal, decision_at: datetime, latency: LatencyProfile,
        fee_schedule: FeeSchedule, limit_price: Decimal | None = None,
        fee_stress_multiplier: Decimal = Decimal("1"),
        visible_depth_fraction: Decimal = Decimal("1"),
    ) -> SimulatedFill:
        if operation not in {"buy", "sell"}:
            raise ValueError("operation must be buy or sell")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not ZERO < visible_depth_fraction <= Decimal("1"):
            raise ValueError("visible depth fraction must be in (0,1]")
        arrival = decision_at.astimezone(UTC) + timedelta(milliseconds=latency.total_ms)
        book = self.book_at(ticker, arrival)
        if book is None or not book.complete:
            return SimulatedFill(
                ticker, side, operation, decision_at, arrival, quantity, ZERO,
                quantity, None, ZERO, ZERO, (), "no_valid_book",
            )
        if operation == "buy":
            levels = book.up_asks if side is Side.UP else book.down_asks
            ordered = sorted(levels, key=lambda item: item.price)
        else:
            levels = book.up_bids if side is Side.UP else book.down_bids
            ordered = sorted(levels, key=lambda item: item.price, reverse=True)

        remaining = quantity
        consumed: list[ExecutionLevel] = []
        for level in ordered:
            if limit_price is not None:
                if operation == "buy" and level.price > limit_price:
                    continue
                if operation == "sell" and level.price < limit_price:
                    continue
            available = (level.quantity * visible_depth_fraction).quantize(Decimal("0.01"))
            take = min(remaining, available)
            if take <= 0:
                continue
            consumed.append(ExecutionLevel(level.price, take))
            remaining -= take
            if remaining <= 0:
                break
        filled = quantity - remaining
        gross = sum((item.price * item.quantity for item in consumed), ZERO)
        vwap = gross / filled if filled > 0 else None
        fee = _order_fee(
            fee_schedule, consumed, multiplier=fee_stress_multiplier
        ) if consumed else ZERO
        reason = "filled" if remaining == 0 else ("partial_fill" if filled else "no_liquidity")
        return SimulatedFill(
            ticker, side, operation, decision_at, arrival, quantity, filled,
            remaining, vwap, fee, gross, tuple(consumed), reason,
        )


@dataclass(slots=True)
class ReplayPosition:
    side: Side
    quantity: Decimal
    average_cost: Decimal


@dataclass(slots=True)
class ReplayPortfolio:
    cash: Decimal
    position: ReplayPosition | None = None
    realized_pnl: Decimal = ZERO
    fees: Decimal = ZERO
    settled_keys: set[tuple[str, str, str]] = field(default_factory=set)

    def apply_fill(self, fill: SimulatedFill) -> None:
        if fill.filled_quantity <= 0:
            return
        if fill.operation == "buy":
            if self.position is not None and self.position.side is not fill.side:
                raise ValueError("cannot hold UP and DOWN simultaneously")
            previous_qty = self.position.quantity if self.position else ZERO
            previous_cost = (
                self.position.average_cost * previous_qty if self.position else ZERO
            )
            outlay = fill.gross + fill.fee
            self.cash -= outlay
            total_qty = previous_qty + fill.filled_quantity
            self.position = ReplayPosition(
                fill.side, total_qty, (previous_cost + outlay) / total_qty,
            )
        else:
            if self.position is None or self.position.side is not fill.side:
                raise ValueError("sell requires same-side position")
            quantity = min(fill.filled_quantity, self.position.quantity)
            proceeds = fill.gross - fill.fee
            cost = self.position.average_cost * quantity
            self.cash += proceeds
            self.realized_pnl += proceeds - cost
            remaining = self.position.quantity - quantity
            self.position = (
                ReplayPosition(fill.side, remaining, self.position.average_cost)
                if remaining > 0 else None
            )
        self.fees += fill.fee

    def settle(
        self, *, episode_id: str, ticker: str, settlement_version: str,
        winning_side: Side,
    ) -> bool:
        key = (episode_id, ticker, settlement_version)
        if key in self.settled_keys:
            return False
        self.settled_keys.add(key)
        if self.position is not None:
            payout = self.position.quantity if self.position.side is winning_side else ZERO
            cost = self.position.average_cost * self.position.quantity
            self.cash += payout
            self.realized_pnl += payout - cost
            self.position = None
        return True


def action_mask(
    *, position: ReplayPosition | None, data_complete: bool,
    stale: bool, market_open: bool, entry_allowed: bool,
    add_count: int = 0, max_add_count: int = 1,
) -> frozenset[ReplayAction]:
    allowed = {ReplayAction.WAIT}
    if position is None:
        if data_complete and not stale and market_open and entry_allowed:
            allowed |= {ReplayAction.BUY_UP, ReplayAction.BUY_DOWN}
    else:
        allowed |= {ReplayAction.HOLD, ReplayAction.EXIT_HALF, ReplayAction.EXIT_ALL}
        if (
            data_complete and not stale and market_open and entry_allowed
            and add_count < max_add_count
        ):
            allowed.add(ReplayAction.ADD)
    return frozenset(allowed)


class TrajectoryLedger:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rows: list[dict[str, object]] = []

    def append(self, **row: object) -> None:
        self.rows.append(row)

    def digest(self) -> str:
        raw = json.dumps(
            {"seed": self.seed, "rows": self.rows}, sort_keys=True,
            separators=(",", ":"), default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()
