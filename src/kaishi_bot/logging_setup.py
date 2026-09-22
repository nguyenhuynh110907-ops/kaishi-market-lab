from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in root.handlers:
        existing.close()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
