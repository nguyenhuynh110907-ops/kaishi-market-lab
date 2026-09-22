import inspect

from kaishi_bot.config import BotConfig
from kaishi_bot.exchange import KalshiDemoAdapter


def test_adapter_creation_has_no_production_configuration_path() -> None:
    source = inspect.getsource(KalshiDemoAdapter.create)

    assert "demo_config" in source
    assert "production" not in source.lower()
    assert "base_url" not in source


def test_strategy_configuration_cannot_accept_exchange_url() -> None:
    assert "base_url" not in BotConfig.model_fields
    assert "environment" not in BotConfig.model_fields
