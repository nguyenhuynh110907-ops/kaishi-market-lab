import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kaishi_bot.research_models import RtiEvent
from kaishi_bot.research_parquet import ParquetRtiSink
from kaishi_bot.research_store import ResearchStore


pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")


def event(source_ms: int, asset: str = "BTC", index_id: str = "BRTI") -> RtiEvent:
    source_time = datetime.fromtimestamp(source_ms / 1000, tz=UTC)
    raw = json.dumps({"id": index_id, "time": source_ms, "value": "63310.36000000"})
    return RtiEvent(
        asset=asset, index_id=index_id, source_timestamp_ms=source_ms,
        source_time_utc=source_time,
        kalshi_received_at=datetime.fromtimestamp((source_ms + 50) / 1000, tz=UTC),
        collector_received_at=datetime.fromtimestamp((source_ms + 80) / 1000, tz=UTC),
        price=Decimal("63310.36000000"), seq=7, session_id="session",
        upstream_latency_ms=50, transport_latency_ms=30, collector_latency_ms=80,
        avg_60s=Decimal("63300.12345678"), final_15m_avg=None,
        avg_60s_window_size=60, avg_60s_window_start_ms=source_ms - 60_000,
        avg_60s_window_end_exclusive_ms=source_ms, is_stale=False,
        gap_detected=False, missing_sample_count=0, raw_data_json=raw,
        event_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


def test_parquet_sink_keeps_exact_decimal_and_commits_manifest_checkpoint(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    root = tmp_path / "research"
    try:
        source_ms = 1_786_476_600_000
        assert ParquetRtiSink(
            root, store, config_sha256="config-hash"
        ).write([event(source_ms)]) == 1
        manifest = store.connection.execute("SELECT * FROM research_files").fetchone()
        checkpoint = store.checkpoint("rti_events", "BRTI")
        path = root / manifest["path"]
        table = parquet.read_table(path)
        assert table.column("price")[0].as_py() == Decimal("63310.36000000")
        assert manifest["status"] == "committed"
        assert manifest["row_count"] == 1
        assert manifest["byte_size"] == path.stat().st_size
        assert manifest["config_sha256"] == "config-hash"
        assert json.loads(manifest["partition_json"]) == {
            "asset": "BTC", "date": "2026-08-11",
        }
        assert checkpoint["last_source_timestamp_ms"] == source_ms
        assert not list(root.rglob("*.tmp"))
    finally:
        store.close()
