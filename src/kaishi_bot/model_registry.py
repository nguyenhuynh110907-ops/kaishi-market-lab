from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ModelRegistryError(ValueError):
    pass


class ModelRegistry:
    """Read-only registry for direct children of a fixed artifact root."""

    JSON_LIMIT = 2 * 1024 * 1024

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._cache: dict[str, tuple[tuple[tuple[int, int], ...], dict[str, object]]] = {}

    def list(self) -> list[dict[str, object]]:
        if not self.root.is_dir():
            return []
        models: list[dict[str, object]] = []
        for directory in sorted(self.root.iterdir(), key=lambda item: item.name):
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                models.append(self._read(directory.name))
            except ModelRegistryError as error:
                models.append({
                    "model_id": directory.name,
                    "status": "invalid",
                    "ready_for_paper_shadow": False,
                    "ready_for_live": False,
                    "errors": [str(error)],
                })
        return sorted(
            models,
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )

    def get(self, model_id: str) -> dict[str, object]:
        return self._read(model_id)

    def directory(self, model_id: str) -> Path:
        if not model_id or model_id in {".", ".."} or Path(model_id).name != model_id:
            raise ModelRegistryError("model ID không hợp lệ")
        candidate = self.root / model_id
        if candidate.is_symlink() or not candidate.is_dir():
            raise ModelRegistryError("không tìm thấy model")
        resolved = candidate.resolve()
        if resolved.parent != self.root:
            raise ModelRegistryError("model nằm ngoài artifact root")
        return resolved

    def _read(self, model_id: str) -> dict[str, object]:
        directory = self.directory(model_id)
        paths = {
            name: directory / name
            for name in ("model.ubj", "metadata.json", "smoke_report.json")
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ModelRegistryError("thiếu " + ", ".join(missing))
        if any(path.is_symlink() for path in paths.values()):
            raise ModelRegistryError("artifact symlink không được phép")
        fingerprint = tuple(
            (path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths.values()
        )
        cached = self._cache.get(model_id)
        if cached is not None and cached[0] == fingerprint:
            return deepcopy(cached[1])
        metadata = self._json(paths["metadata.json"])
        report = self._json(paths["smoke_report.json"])
        if not isinstance(metadata, dict) or not isinstance(report, dict):
            raise ModelRegistryError("metadata và report phải là JSON object")
        split = metadata.get("split") if isinstance(metadata.get("split"), dict) else {}
        report_split = report.get("split") if isinstance(report.get("split"), dict) else {}
        test = report.get("test") if isinstance(report.get("test"), dict) else {}
        dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
        config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
        split = split or report_split
        test_used = metadata.get("test_was_used_for_selection")
        errors: list[str] = []
        if metadata.get("artifact_version") != 1:
            errors.append("artifact_version không được hỗ trợ")
        if test_used is not False:
            errors.append("không chứng minh được test set chưa dùng để chọn model")
        if not metadata.get("feature_names"):
            errors.append("thiếu feature schema")
        if not test:
            errors.append("thiếu test metrics")
        purpose = str(report.get("purpose") or "unknown")
        technical_only = purpose == "technical_smoke_test_only"
        model_stat = paths["model.ubj"].stat()
        updated = max(path.stat().st_mtime for path in paths.values())
        result: dict[str, object] = {
            "model_id": model_id,
            "model_type": "xgboost",
            "artifact_version": metadata.get("artifact_version"),
            "status": "technical_only" if technical_only and not errors else (
                "ready" if not errors else "invalid"
            ),
            "ready_for_paper_shadow": not errors,
            "ready_for_live": False,
            "purpose": purpose,
            "warning": report.get("warning"),
            "asset": dataset.get("asset"),
            "dataset": dataset,
            "test": test,
            "split_counts": {
                key: len(split.get(f"{key}_tickers", []))
                if isinstance(split.get(f"{key}_tickers", []), list) else 0
                for key in ("train", "validation", "test")
            },
            "config": config,
            "feature_count": len(metadata.get("feature_names", [])),
            "feature_names": metadata.get("feature_names", []),
            "inference_rounds": metadata.get("inference_rounds"),
            "test_was_used_for_selection": test_used,
            "artifact": {
                "model_file": "model.ubj",
                "model_bytes": model_stat.st_size,
                "model_sha256": self._sha256(paths["model.ubj"]),
                "metadata_file": "metadata.json",
                "report_file": "smoke_report.json",
            },
            "updated_at": datetime.fromtimestamp(updated, tz=UTC).isoformat(),
            "errors": errors,
        }
        self._cache[model_id] = (fingerprint, deepcopy(result))
        return result

    def _json(self, path: Path) -> Any:
        if path.stat().st_size > self.JSON_LIMIT:
            raise ModelRegistryError(f"{path.name} vượt giới hạn 2 MB")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ModelRegistryError(f"{path.name} không đọc được") from error

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
