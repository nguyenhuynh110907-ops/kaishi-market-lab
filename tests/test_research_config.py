from pathlib import Path

import pytest

from kaishi_bot.research_config import ResearchCaptureConfig


def test_missing_research_config_is_disabled() -> None:
    config = ResearchCaptureConfig.load(Path("does-not-exist.yaml"))
    assert config.enabled is False
    assert config.assets == ("BTC", "ETH", "SOL", "XRP", "DOGE")


def test_nested_research_config_resolves_root_from_config_directory(tmp_path) -> None:
    path = tmp_path / "research.yaml"
    path.write_text(
        "research_capture:\n  enabled: true\n  root: capture\n  assets: [BTC]\n",
        encoding="utf-8",
    )
    config = ResearchCaptureConfig.load(path)
    assert config.enabled is True
    assert config.root == tmp_path / "capture"
    assert config.index_ids["BTC"] == "BRTI"


def test_research_config_rejects_unknown_assets() -> None:
    with pytest.raises(ValueError, match="unsupported research assets"):
        ResearchCaptureConfig(assets=("NOT_A_COIN",))
