from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from kaishi_bot.entry_guard import (
    EntryGuardSettings,
    GuardQuote,
    GuardReason,
    evaluate_entry,
)
from kaishi_bot.fees import FeeSchedule


NOW = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
SCHEDULE = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")


def side_quote(bid: str, ask: str, at: datetime) -> GuardQuote:
    return GuardQuote(bid=Decimal(bid), ask=Decimal(ask), observed_at=at)


def decide(
    previous: GuardQuote | None,
    current: GuardQuote,
    **changes: object,
):
    values = {
        "previous": previous,
        "current": current,
        "entry_price": Decimal("0.25"),
        "entry_min": Decimal("0.17"),
        "stop_loss": Decimal("0.15"),
        "take_profit": Decimal("0.40"),
        "budget": Decimal("10"),
        "settings": EntryGuardSettings(),
        "fee_schedule": SCHEDULE,
        "cooldown_until": None,
    }
    values.update(changes)
    return evaluate_entry(**values)


def test_rejects_jump_below_effective_floor_before_confirmation() -> None:
    decision = decide(
        side_quote("0.29", "0.30", NOW),
        side_quote("0.13", "0.14", NOW + timedelta(seconds=1)),
    )
    assert decision.reason is GuardReason.BELOW_FLOOR
    assert decision.effective_floor == Decimal("0.17")


def test_explicit_entry_min_is_the_only_lower_entry_boundary() -> None:
    rejected = decide(
        side_quote("0.18", "0.19", NOW),
        side_quote("0.15", "0.16", NOW + timedelta(seconds=1)),
        entry_min=Decimal("0.18"),
    )
    accepted = decide(
        side_quote("0.17", "0.18", NOW),
        side_quote("0.17", "0.18", NOW + timedelta(seconds=1)),
        entry_min=Decimal("0.18"),
        stop_loss=Decimal("0.15"),
        settings=EntryGuardSettings(max_spread_ratio="1"),
    )
    assert rejected.reason is GuardReason.BELOW_FLOOR
    assert accepted.reason is not GuardReason.BID_IN_SL_BUFFER


def test_rejects_absolute_spread() -> None:
    previous = side_quote("0.18", "0.20", NOW)
    current = side_quote("0.18", "0.20", NOW + timedelta(seconds=1))
    assert decide(previous, current).reason is GuardReason.SPREAD_TOO_WIDE


def test_rejects_relative_spread() -> None:
    previous = side_quote("0.15", "0.18", NOW)
    current = side_quote("0.15", "0.18", NOW + timedelta(seconds=1))
    settings = EntryGuardSettings(
        stop_loss_buffer="0", max_spread="0.10", max_spread_ratio="0.15"
    )
    assert decide(
        previous,
        current,
        stop_loss=Decimal("0.10"),
        settings=settings,
    ).reason is GuardReason.SPREAD_TOO_WIDE


@pytest.mark.parametrize("previous", [None, side_quote("0.19", "0.20", NOW)])
def test_single_valid_quote_enters_without_timing_confirmation(
    previous: GuardQuote | None,
) -> None:
    current = side_quote("0.19", "0.20", NOW + timedelta(seconds=10))
    decision = decide(previous, current)
    assert decision.eligible is True
    assert decision.reason is GuardReason.ELIGIBLE


def test_accepts_two_valid_ticks_and_reports_fee_aware_reward_risk() -> None:
    decision = decide(
        side_quote("0.19", "0.20", NOW),
        side_quote("0.19", "0.20", NOW + timedelta(seconds=1)),
    )
    assert decision.eligible is True
    assert decision.reason is GuardReason.ELIGIBLE
    assert decision.quantity > 0
    assert decision.entry_outlay <= Decimal("10")
    assert decision.reward_risk >= Decimal("1.50")


def test_active_cooldown_blocks_otherwise_valid_entry() -> None:
    decision = decide(
        side_quote("0.19", "0.20", NOW),
        side_quote("0.19", "0.20", NOW + timedelta(seconds=1)),
        cooldown_until=NOW + timedelta(seconds=5),
    )
    assert decision.reason is GuardReason.COOLDOWN_ACTIVE


def test_reward_risk_is_diagnostic_and_does_not_reject_entry() -> None:
    decision = decide(
        side_quote("0.71", "0.72", NOW),
        side_quote("0.71", "0.72", NOW + timedelta(seconds=1)),
        entry_min=Decimal("0.69"),
        entry_price=Decimal("0.74"),
        stop_loss=Decimal("0.29"),
        take_profit=Decimal("0.97"),
        budget=Decimal("5"),
        settings=EntryGuardSettings(minimum_reward_risk="0.96"),
    )
    assert decision.reward_risk < Decimal("0.96")
    assert decision.eligible is True
    assert decision.reason is GuardReason.ELIGIBLE


def test_live_fractional_entry_sizes_one_cent_budget() -> None:
    decision = decide(
        None,
        side_quote("0.80", "0.81", NOW),
        entry_min=Decimal("0.77"),
        entry_price=Decimal("0.82"),
        stop_loss=Decimal("0.49"),
        take_profit=Decimal("0.94"),
        budget=Decimal("0.01"),
        fractional=True,
    )
    assert decision.eligible is True
    assert decision.quantity == Decimal("0.01")
    assert decision.entry_outlay == Decimal("0.01")


def test_zero_sized_entry_fails_closed() -> None:
    decision = decide(
        None,
        side_quote("0.19", "0.20", NOW),
        budget=Decimal("0.001"),
        fractional=True,
    )
    assert decision.eligible is False
    assert decision.reason is GuardReason.INSUFFICIENT_BUDGET
    assert decision.quantity == 0
