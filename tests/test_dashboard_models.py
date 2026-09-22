from decimal import Decimal

import pytest
from pydantic import ValidationError

from kaishi_bot.dashboard_models import DashboardSettings, RuntimeMode


def test_dashboard_defaults_are_safe_and_cover_five_crypto_series() -> None:
    settings = DashboardSettings()

    assert settings.mode is RuntimeMode.PAPER
    assert settings.stop_loss == Decimal("0.15")
    assert settings.entry_price == Decimal("0.25")
    assert settings.entry_min == Decimal("0.17")
    assert settings.take_profit == Decimal("0.40")
    assert settings.entry_amount == Decimal("10.00")
    assert settings.daily_cap == Decimal("1000.00")
    assert {item.series for item in settings.assets.values()} == {
        "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"
    }


def test_paper_and_live_daily_limits_are_separate() -> None:
    settings = DashboardSettings()

    assert settings.paper_daily_cap == Decimal("1000.00")
    assert settings.daily_cap == Decimal("1000.00")


def test_entry_guard_defaults_are_safe() -> None:
    settings = DashboardSettings()

    assert settings.entry_guard.entry_floor_ratio == Decimal("0.60")
    assert settings.entry_guard.stop_loss_buffer == Decimal("0.02")
    assert settings.entry_guard.max_spread == Decimal("0.01")
    assert settings.entry_guard.max_spread_ratio == Decimal("0.15")
    assert settings.entry_guard.confirmation_ticks == 2
    assert settings.entry_guard.minimum_reward_risk == Decimal("1.50")
    assert settings.entry_guard.reentry_cooldown_seconds == 10


def test_entry_window_defaults_to_the_full_fifteen_minutes() -> None:
    settings = DashboardSettings()
    assert settings.entry_start_seconds == 0
    assert settings.entry_end_seconds == 900


@pytest.mark.parametrize(
    "payload",
    [
        {"entry_start_seconds": 300, "entry_end_seconds": 300},
        {"entry_start_seconds": 600, "entry_end_seconds": 300},
        {"entry_start_seconds": -1},
        {"entry_end_seconds": 901},
    ],
)
def test_entry_window_must_be_ordered_inside_the_cycle(payload) -> None:
    with pytest.raises(ValueError):
        DashboardSettings.model_validate(payload)


def test_entry_band_must_stay_between_stop_loss_and_take_profit() -> None:
    with pytest.raises(ValidationError, match="entry_min"):
        DashboardSettings(
            stop_loss="0.24",
            entry_min="0.24",
            entry_price="0.25",
            take_profit="0.40",
        )

    with pytest.raises(ValidationError, match="entry_min"):
        DashboardSettings(entry_min="0.26", entry_price="0.25")


@pytest.mark.parametrize(
    "changes",
    [
        {"stop_loss": "0.25"},
        {"entry_price": "0.40"},
        {"take_profit": "1.00"},
        {"entry_amount": "200.01"},
        {"paper_daily_cap": "10000.01"},
        {"daily_cap": "10000.01"},
    ],
)
def test_dashboard_rejects_unsafe_thresholds_and_caps(changes: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        DashboardSettings(**changes)


def test_dashboard_allows_two_hundred_dollars_per_entry() -> None:
    assert DashboardSettings(entry_amount="200").entry_amount == Decimal("200")


def test_dashboard_allows_ten_thousand_dollars_per_day() -> None:
    settings = DashboardSettings(paper_daily_cap="10000", daily_cap="10000")
    assert settings.paper_daily_cap == Decimal("10000")
    assert settings.daily_cap == Decimal("10000")


def test_dashboard_rejects_unknown_assets() -> None:
    settings = DashboardSettings().model_dump(mode="json")
    settings["assets"]["ADA"] = {
        "series": "KXADA15M", "enabled": True, "trade_up": True, "trade_down": True
    }

    with pytest.raises(ValidationError, match="assets must be exactly"):
        DashboardSettings.model_validate(settings)
