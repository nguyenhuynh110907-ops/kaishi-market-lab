from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from kaishi_bot.fees import (
    FeeSchedule,
    fractional_contract_size,
    fractional_exit_value,
    taker_fee,
    whole_contract_size,
)


class EntryGuardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_floor_ratio: Decimal = Field(default=Decimal("0.60"), gt=0, lt=1)
    stop_loss_buffer: Decimal = Field(default=Decimal("0.02"), ge=0, lt=1)
    max_spread: Decimal = Field(default=Decimal("0.01"), gt=0, lt=1)
    max_spread_ratio: Decimal = Field(default=Decimal("0.15"), gt=0, le=1)
    confirmation_ticks: int = Field(default=2, ge=1, le=2)
    minimum_reward_risk: Decimal = Field(default=Decimal("1.50"), gt=0)
    reentry_cooldown_seconds: int = Field(default=10, ge=0, le=3600)


class GuardReason(StrEnum):
    BELOW_FLOOR = "below_floor"
    BID_IN_SL_BUFFER = "bid_in_sl_buffer"
    SPREAD_TOO_WIDE = "spread_too_wide"
    CONFIRMATION_PENDING = "confirmation_pending"
    REWARD_RISK_TOO_LOW = "reward_risk_too_low"
    COOLDOWN_ACTIVE = "cooldown_active"
    DEPTH_UNAVAILABLE = "depth_unavailable"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    INSUFFICIENT_BUDGET = "insufficient_budget"
    ELIGIBLE = "eligible"


@dataclass(frozen=True, slots=True)
class GuardQuote:
    bid: Decimal
    ask: Decimal
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class GuardDecision:
    eligible: bool
    reason: GuardReason
    quantity: Decimal
    premium: Decimal
    entry_fee: Decimal
    entry_outlay: Decimal
    reward_risk: Decimal | None
    effective_floor: Decimal
    effective_max_spread: Decimal


def _price_reason(
    quote: GuardQuote,
    *,
    entry_min: Decimal,
    entry_price: Decimal,
    settings: EntryGuardSettings,
) -> GuardReason | None:
    if quote.ask < entry_min:
        return GuardReason.BELOW_FLOOR
    if quote.ask > entry_price:
        return GuardReason.CONFIRMATION_PENDING
    maximum = min(settings.max_spread, settings.max_spread_ratio * quote.ask)
    if quote.ask - quote.bid > maximum:
        return GuardReason.SPREAD_TOO_WIDE
    return None


def evaluate_entry(
    *,
    previous: GuardQuote | None,
    current: GuardQuote,
    entry_min: Decimal,
    entry_price: Decimal,
    stop_loss: Decimal,
    take_profit: Decimal,
    budget: Decimal,
    settings: EntryGuardSettings,
    fee_schedule: FeeSchedule,
    cooldown_until: datetime | None,
    fractional: bool = False,
) -> GuardDecision:
    effective_floor = entry_min
    effective_max_spread = min(
        settings.max_spread,
        settings.max_spread_ratio * current.ask,
    )

    reason = _price_reason(
        current,
        entry_min=entry_min,
        entry_price=entry_price,
        settings=settings,
    )
    if reason is None and cooldown_until is not None and current.observed_at < cooldown_until:
        reason = GuardReason.COOLDOWN_ACTIVE

    quantity = premium = entry_fee = entry_outlay = Decimal("0")
    reward_risk: Decimal | None = None
    if reason is None:
        size = fractional_contract_size if fractional else whole_contract_size
        quantity, premium, entry_fee = size(fee_schedule, current.ask, budget)
        entry_outlay = premium + entry_fee
        if quantity > 0:
            if fractional:
                _, tp_net = fractional_exit_value(fee_schedule, quantity, take_profit)
                _, sl_net = fractional_exit_value(fee_schedule, quantity, stop_loss)
            else:
                tp_net = quantity * take_profit - taker_fee(
                    fee_schedule, quantity, take_profit
                )
                sl_net = quantity * stop_loss - taker_fee(
                    fee_schedule, quantity, stop_loss
                )
            reward = tp_net - entry_outlay
            risk = entry_outlay - sl_net
            if reward > 0 and risk > 0:
                reward_risk = reward / risk
        else:
            reason = GuardReason.INSUFFICIENT_BUDGET
    return GuardDecision(
        eligible=reason is None,
        reason=reason or GuardReason.ELIGIBLE,
        quantity=quantity,
        premium=premium,
        entry_fee=entry_fee,
        entry_outlay=entry_outlay,
        reward_risk=reward_risk,
        effective_floor=effective_floor,
        effective_max_spread=effective_max_spread,
    )
