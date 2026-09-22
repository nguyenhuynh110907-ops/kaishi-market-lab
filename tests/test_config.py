from decimal import Decimal
from pathlib import Path

import pytest

from kaishi_bot.config import BotConfig, Credentials


def test_defaults_are_safe(tmp_path: Path) -> None:
    path = tmp_path / "bot.yaml"
    path.write_text("series: KXBTC15M\n", encoding="utf-8")

    config = BotConfig.load(path)

    assert config.entry_price == Decimal("0.25")
    assert config.take_profit_price == Decimal("0.40")
    assert config.contracts == Decimal("1")
    assert config.trade_up is True
    assert config.trade_down is True
    assert config.min_seconds_before_close == 60


@pytest.mark.parametrize("value", [0, 1, 9])
def test_close_guard_cannot_be_unsafe(tmp_path: Path, value: int) -> None:
    path = tmp_path / "bot.yaml"
    path.write_text(f"min_seconds_before_close: {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="greater than or equal to 10"):
        BotConfig.load(path)


def test_take_profit_must_exceed_entry(tmp_path: Path) -> None:
    path = tmp_path / "bot.yaml"
    path.write_text(
        'entry_price: "0.25"\ntake_profit_price: "0.25"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="greater than entry"):
        BotConfig.load(path)


def test_at_least_one_side_must_be_enabled(tmp_path: Path) -> None:
    path = tmp_path / "bot.yaml"
    path.write_text("trade_up: false\ntrade_down: false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one side"):
        BotConfig.load(path)


def test_unknown_configuration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bot.yaml"
    path.write_text("production_url: https://example.invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        BotConfig.load(path)


def test_credentials_require_existing_key_file(tmp_path: Path) -> None:
    key = tmp_path / "demo.key"
    key.write_text("private", encoding="utf-8")

    credentials = Credentials.load(
        {
            "KALSHI_KEY_ID": "demo-id",
            "KALSHI_PRIVATE_KEY_PATH": str(key),
        }
    )

    assert credentials.key_id == "demo-id"
    assert credentials.private_key_path == key


def test_credentials_reject_missing_key_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="readable file"):
        Credentials.load(
            {
                "KALSHI_KEY_ID": "demo-id",
                "KALSHI_PRIVATE_KEY_PATH": str(tmp_path / "missing.key"),
            }
        )
