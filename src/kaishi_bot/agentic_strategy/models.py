from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from kaishi_bot.domain import Side


ZERO = Decimal("0")
ONE = Decimal("1")


class Action(StrEnum):
    WAIT = "wait"
    BUY = "buy"
    HOLD = "hold"
    ADD = "add"
    EXIT_HALF = "exit_half"
    EXIT_ALL = "exit_all"


@dataclass(frozen=True, slots=True)
class PositionState:
    side: Side
    quantity: Decimal
    average_entry_price: Decimal
    seconds_held: int = 0
    add_count: int = 0
    partial_exit_taken: bool = False

    def __post_init__(self) -> None:
        if self.quantity <= ZERO:
            raise ValueError("position quantity must be positive")
        if not ZERO < self.average_entry_price < ONE:
            raise ValueError("average entry price must be between zero and one")
        if self.seconds_held < 0 or self.add_count < 0:
            raise ValueError("position counters cannot be negative")


@dataclass(frozen=True, slots=True)
class MarketObservation:
    ticker: str
    observed_at: datetime
    seconds_remaining: int
    up_bid: Decimal
    up_ask: Decimal
    down_bid: Decimal
    down_ask: Decimal
    target_price: Decimal
    brti_price: Decimal
    brti_sigma_per_sqrt_second: Decimal
    locked_sample_count: int = 0
    locked_sample_sum: Decimal = ZERO
    entry_fee_up: Decimal = ZERO
    entry_fee_down: Decimal = ZERO
    position: PositionState | None = None
    data_stale: bool = False
    has_gap: bool = False

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("ticker is required")
        if self.seconds_remaining < 0:
            raise ValueError("seconds remaining cannot be negative")
        for name in ("up_bid", "up_ask", "down_bid", "down_ask"):
            value = getattr(self, name)
            if not ZERO <= value <= ONE:
                raise ValueError(f"{name} must be between zero and one")
        if self.up_bid > self.up_ask or self.down_bid > self.down_ask:
            raise ValueError("bid cannot exceed ask")
        if self.target_price <= ZERO or self.brti_price <= ZERO:
            raise ValueError("target and BRTI price must be positive")
        if self.brti_sigma_per_sqrt_second < ZERO:
            raise ValueError("BRTI volatility cannot be negative")
        if not 0 <= self.locked_sample_count <= 60:
            raise ValueError("locked sample count must be between zero and 60")
        if self.locked_sample_count == 0 and self.locked_sample_sum != ZERO:
            raise ValueError("locked sum requires locked samples")
        if self.entry_fee_up < ZERO or self.entry_fee_down < ZERO:
            raise ValueError("entry fees cannot be negative")

    @property
    def is_final_minute(self) -> bool:
        return self.seconds_remaining <= 60

    def ask(self, side: Side) -> Decimal:
        return self.up_ask if side is Side.UP else self.down_ask

    def bid(self, side: Side) -> Decimal:
        return self.up_bid if side is Side.UP else self.down_bid

    def entry_fee(self, side: Side) -> Decimal:
        return self.entry_fee_up if side is Side.UP else self.entry_fee_down

    def required_remaining_average(self) -> Decimal | None:
        remaining = 60 - self.locked_sample_count
        if self.locked_sample_count <= 0 or remaining <= 0:
            return None
        return (
            Decimal(60) * self.target_price - self.locked_sample_sum
        ) / Decimal(remaining)


@dataclass(frozen=True, slots=True)
class ProbabilityEstimate:
    yes: Decimal
    no: Decimal
    uncertainty: Decimal
    required_remaining_average: Decimal | None

    def __post_init__(self) -> None:
        if not ZERO <= self.yes <= ONE or not ZERO <= self.no <= ONE:
            raise ValueError("probabilities must be between zero and one")
        if abs((self.yes + self.no) - ONE) > Decimal("0.000001"):
            raise ValueError("yes and no probabilities must sum to one")
        if self.uncertainty < ZERO:
            raise ValueError("uncertainty cannot be negative")

    def for_side(self, side: Side) -> Decimal:
        return self.yes if side is Side.UP else self.no


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    side: Side | None
    fraction: Decimal
    probability: Decimal
    edge: Decimal
    reason: str


@dataclass(frozen=True, slots=True)
class AgentConfig:
    entry_min: Decimal = Decimal("0.68")
    entry_max: Decimal = Decimal("0.77")
    entry_probability: Decimal = Decimal("0.82")
    entry_edge: Decimal = Decimal("0.08")
    final_entry_probability: Decimal = Decimal("0.88")
    final_entry_edge: Decimal = Decimal("0.07")
    add_probability: Decimal = Decimal("0.93")
    exit_probability: Decimal = Decimal("0.58")
    emergency_bid: Decimal = Decimal("0.42")
    emergency_probability: Decimal = Decimal("0.65")
    first_take_profit: Decimal = Decimal("0.89")
    final_take_profit: Decimal = Decimal("0.94")
    hold_to_settlement_probability: Decimal = Decimal("0.97")
    no_entry_last_seconds: int = 8
    confirmation_ticks: int = 3
    exploratory_fraction: Decimal = Decimal("0.25")
    final_minute_fraction: Decimal = Decimal("0.50")
    add_fraction: Decimal = Decimal("0.25")
    max_add_count: int = 1
    minimum_sigma_ratio: Decimal = Decimal("0.00001")

    def __post_init__(self) -> None:
        probabilities = (
            self.entry_min,
            self.entry_max,
            self.entry_probability,
            self.final_entry_probability,
            self.add_probability,
            self.exit_probability,
            self.emergency_bid,
            self.emergency_probability,
            self.first_take_profit,
            self.final_take_profit,
            self.hold_to_settlement_probability,
        )
        if any(not ZERO < value < ONE for value in probabilities):
            raise ValueError("price and probability settings must be between zero and one")
        if self.entry_min > self.entry_max:
            raise ValueError("entry minimum cannot exceed entry maximum")
        if self.first_take_profit > self.final_take_profit:
            raise ValueError("first take profit cannot exceed final take profit")
        if self.confirmation_ticks < 1 or self.no_entry_last_seconds < 0:
            raise ValueError("invalid timing settings")
        if self.max_add_count < 0:
            raise ValueError("max add count cannot be negative")
