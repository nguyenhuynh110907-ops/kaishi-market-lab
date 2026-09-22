import json
import logging
from pathlib import Path

from kaishi_bot.logging_setup import configure_logging


def test_logging_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "bot.jsonl"
    configure_logging(path)

    logging.getLogger("kaishi_bot.test").info("connected to DEMO")

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["level"] == "INFO"
    assert payload["logger"] == "kaishi_bot.test"
    assert payload["message"] == "connected to DEMO"
