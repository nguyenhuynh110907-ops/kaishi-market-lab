from decimal import Decimal

import pytest

from kaishi_bot.fees import (
    FeeSchedule,
    fractional_contract_size,
    fractional_entry_cost,
    fractional_exit_value,
    taker_fee,
    whole_contract_size,
)


SCHEDULE = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "2026-02-05")


def test_quadratic_taker_fee_rounds_up_to_cent() -> None:
    assert taker_fee(SCHEDULE, Decimal("3"), Decimal("0.25")) == Decimal("0.04")


def test_whole_contract_size_keeps_premium_plus_fee_under_one_dollar() -> None:
    quantity, premium, fee = whole_contract_size(
        SCHEDULE, Decimal("0.25"), Decimal("1.00")
    )
    assert (quantity, premium, fee) == (
        Decimal("3"), Decimal("0.75"), Decimal("0.04")
    )
    assert premium + fee <= Decimal("1.00")


def test_unknown_fee_type_fails_closed() -> None:
    bad = FeeSchedule("flat", Decimal("1"), Decimal("0.07"), "test")
    with pytest.raises(ValueError, match="unsupported fee type"):
        taker_fee(bad, Decimal("1"), Decimal("0.25"))


def test_zero_quantity_has_zero_fee() -> None:
    assert taker_fee(SCHEDULE, Decimal("0"), Decimal("0.25")) == Decimal("0.00")


def test_one_cent_live_budget_buys_one_hundredth_contract() -> None:
    quantity, premium, fee = fractional_contract_size(
        SCHEDULE, Decimal("0.81"), Decimal("0.01")
    )
    assert quantity == Decimal("0.01")
    assert premium == Decimal("0.0081")
    assert premium + fee == Decimal("0.01")


def test_fractional_cost_and_exit_are_cent_aligned() -> None:
    premium, fee, debit = fractional_entry_cost(
        SCHEDULE, Decimal("0.01"), Decimal("0.81")
    )
    exit_fee, proceeds = fractional_exit_value(
        SCHEDULE, Decimal("0.01"), Decimal("0.94")
    )
    assert debit == Decimal("0.01")
    assert premium + fee == debit
    assert proceeds == Decimal("0.00")
    assert exit_fee == Decimal("0.0094")
