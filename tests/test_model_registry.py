import json

import pytest

from kaishi_bot.model_registry import ModelRegistry, ModelRegistryError


def _artifact(root, model_id="btc_model"):
    directory = root / model_id
    directory.mkdir(parents=True)
    (directory / "model.ubj").write_bytes(b"not-a-real-model-and-never-loaded")
    (directory / "metadata.json").write_text(json.dumps({
        "artifact_version": 1,
        "feature_names": ["seconds_remaining"],
        "config": {"seed": 17},
        "split": {
            "train_tickers": ["train"],
            "validation_tickers": ["validation"],
            "test_tickers": ["test"],
        },
        "inference_rounds": 1,
        "test_was_used_for_selection": False,
    }))
    (directory / "smoke_report.json").write_text(json.dumps({
        "purpose": "technical_smoke_test_only",
        "warning": "Never use for live trading.",
        "dataset": {"asset": "BTC", "observations": 100},
        "test": {"accuracy": 0.5, "brier": 0.25, "market_count": 1},
    }))
    return directory


def test_registry_lists_json_metadata_without_loading_model(tmp_path) -> None:
    root = tmp_path / "artifacts"
    _artifact(root)

    item = ModelRegistry(root).list()[0]

    assert item["model_id"] == "btc_model"
    assert item["status"] == "technical_only"
    assert item["ready_for_paper_shadow"] is True
    assert item["ready_for_live"] is False
    assert item["artifact"]["model_bytes"] > 0
    assert len(item["artifact"]["model_sha256"]) == 64


@pytest.mark.parametrize("model_id", ["../escape", "a/b", ".", "..", ""])
def test_registry_rejects_path_traversal(tmp_path, model_id) -> None:
    registry = ModelRegistry(tmp_path / "artifacts")
    with pytest.raises(ModelRegistryError):
        registry.get(model_id)


def test_registry_rejects_symlinked_artifact_directory(tmp_path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    _artifact(tmp_path, "outside")
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)

    assert ModelRegistry(root).list() == []


def test_registry_surfaces_incomplete_direct_child_as_invalid(tmp_path) -> None:
    root = tmp_path / "artifacts"
    (root / "incomplete").mkdir(parents=True)

    item = ModelRegistry(root).list()[0]

    assert item["status"] == "invalid"
    assert item["ready_for_live"] is False
