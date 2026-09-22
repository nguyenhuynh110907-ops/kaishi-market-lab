from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR


CENT = Decimal("0.01")
CENTICENT = Decimal("0.0001")
FRACTIONAL_CONTRACT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    fee_type: str
    multiplier: Decimal
    taker_rate: Decimal
    version: str


def taker_fee(
    schedule: FeeSchedule, contracts: Decimal, price: Decimal
) -> Decimal:
    if schedule.fee_type != "quadratic":
        raise ValueError(f"unsupported fee type: {schedule.fee_type}")
    if contracts < 0 or not Decimal("0") <= price <= Decimal("1"):
        raise ValueError("contracts and price must be valid")
    raw = (
        schedule.taker_rate
        * schedule.multiplier
        * contracts
        * price
        * (Decimal("1") - price)
    )
    return raw.quantize(CENT, rounding=ROUND_CEILING)


def whole_contract_size(
    schedule: FeeSchedule, price: Decimal, all_in_budget: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    quantity = (
        Decimal(int(all_in_budget / price))
        if price > 0 and all_in_budget > 0
        else Decimal("0")
    )
    while quantity > 0:
        premium = quantity * price
        fee = taker_fee(schedule, quantity, price)
        if premium + fee <= all_in_budget:
            return quantity, premium, fee
        quantity -= 1
    return Decimal("0"), Decimal("0"), Decimal("0")


def fractional_entry_cost(
    schedule: FeeSchedule, contracts: Decimal, price: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Return premium, effective fee, and cent-aligned cash debit for a buy."""
    if schedule.fee_type != "quadratic":
        raise ValueError(f"unsupported fee type: {schedule.fee_type}")
    if contracts < 0 or not Decimal("0") <= price <= Decimal("1"):
        raise ValueError("contracts and price must be valid")
    if contracts == 0:
        return Decimal("0"), Decimal("0"), Decimal("0")
    premium = contracts * price
    raw_fee = (
        schedule.taker_rate
        * schedule.multiplier
        * contracts
        * price
        * (Decimal("1") - price)
    ).quantize(CENTICENT, rounding=ROUND_CEILING)
    cash_debit = (premium + raw_fee).quantize(CENT, rounding=ROUND_CEILING)
    return premium, cash_debit - premium, cash_debit


def fractional_exit_value(
    schedule: FeeSchedule, contracts: Decimal, price: Decimal
) -> tuple[Decimal, Decimal]:
    """Return effective fee and cent-aligned cash proceeds for a sell."""
    if schedule.fee_type != "quadratic":
        raise ValueError(f"unsupported fee type: {schedule.fee_type}")
    if contracts < 0 or not Decimal("0") <= price <= Decimal("1"):
        raise ValueError("contracts and price must be valid")
    if contracts == 0:
        return Decimal("0"), Decimal("0")
    gross = contracts * price
    raw_fee = (
        schedule.taker_rate
        * schedule.multiplier
        * contracts
        * price
        * (Decimal("1") - price)
    ).quantize(CENTICENT, rounding=ROUND_CEILING)
    proceeds = max(Decimal("0"), gross - raw_fee).quantize(CENT, rounding=ROUND_FLOOR)
    return gross - proceeds, proceeds


def fractional_contract_size(
    schedule: FeeSchedule, price: Decimal, all_in_budget: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Size a Live order in 0.01-contract increments within its cash budget."""
    if price <= 0 or all_in_budget < CENT:
        return Decimal("0"), Decimal("0"), Decimal("0")
    quantity = (
        (all_in_budget / price / FRACTIONAL_CONTRACT).to_integral_value(
            rounding=ROUND_FLOOR
        )
        * FRACTIONAL_CONTRACT
    )
    while quantity > 0:
        premium, fee, cash_debit = fractional_entry_cost(schedule, quantity, price)
        if cash_debit <= all_in_budget:
            return quantity, premium, fee
        quantity -= FRACTIONAL_CONTRACT
    return Decimal("0"), Decimal("0"), Decimal("0")
