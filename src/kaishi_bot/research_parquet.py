from __future__ import annotations

import hashlib
import os
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kaishi_bot.research_models import RtiEvent
from kaishi_bot.research_store import ResearchStore


RTI_SCHEMA_VERSION = 1
EIGHT_DP = Decimal("0.00000001")


def _exact_8dp(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    quantized = value.quantize(EIGHT_DP)
    if quantized != value:
        raise ValueError(f"RTI value has more than 8 decimal places: {value}")
    return quantized


class ParquetUnavailableError(RuntimeError):
    pass


class ParquetRtiSink:
    """Write immutable, partitioned RTI batches and commit their manifests."""

    def __init__(
        self, root: Path, store: ResearchStore, compression: str = "zstd",
        config_sha256: str | None = None,
    ) -> None:
        self.root = root
        self.store = store
        self.compression = None if compression == "none" else compression
        self.config_sha256 = config_sha256

    @staticmethod
    def available() -> bool:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            return False
        return True

    def write(self, events: list[RtiEvent]) -> int:
        if not events:
            return 0
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ParquetUnavailableError(
                "research capture requires the 'research' optional dependency"
            ) from error

        partitions: dict[tuple[str, str], list[RtiEvent]] = defaultdict(list)
        for event in events:
            day = event.source_time_utc.astimezone(UTC).date().isoformat()
            partitions[(day, event.asset)].append(event)

        total = 0
        for (day, asset), rows in partitions.items():
            rows.sort(key=lambda event: (event.source_timestamp_ms, event.seq or -1))
            directory = self.root / "rti_events" / f"date={day}" / f"asset={asset}"
            directory.mkdir(parents=True, exist_ok=True)
            file_id = str(uuid.uuid4())
            final_path = directory / f"part-{file_id}.parquet"
            temporary_path = directory / f".{file_id}.parquet.tmp"
            schema = pa.schema([
                # ``asset`` is represented by the Hive partition to avoid a
                # duplicate partition/file column conflict in datasets.
                ("index_id", pa.string()),
                ("source_timestamp_ms", pa.int64()),
                ("source_time_utc", pa.timestamp("us", tz="UTC")),
                ("kalshi_received_at", pa.timestamp("us", tz="UTC")),
                ("collector_received_at", pa.timestamp("us", tz="UTC")),
                ("price", pa.decimal128(24, 8)), ("seq", pa.int64()),
                ("session_id", pa.string()), ("upstream_latency_ms", pa.int64()),
                ("transport_latency_ms", pa.int64()),
                ("collector_latency_ms", pa.int64()),
                ("avg_60s", pa.decimal128(24, 8)),
                ("final_15m_avg", pa.decimal128(24, 8)),
                ("avg_60s_window_size", pa.int32()),
                ("avg_60s_window_start_ms", pa.int64()),
                ("avg_60s_window_end_exclusive_ms", pa.int64()),
                ("final_15m_window_size", pa.int32()),
                ("final_15m_window_start_ms", pa.int64()),
                ("final_15m_window_end_exclusive_ms", pa.int64()),
                ("is_stale", pa.bool_()), ("gap_detected", pa.bool_()),
                ("missing_sample_count", pa.int32()),
                ("raw_data_json", pa.string()), ("event_sha256", pa.string()),
            ])
            data = {
                "index_id": [row.index_id for row in rows],
                "source_timestamp_ms": [row.source_timestamp_ms for row in rows],
                "source_time_utc": [row.source_time_utc for row in rows],
                "kalshi_received_at": [row.kalshi_received_at for row in rows],
                "collector_received_at": [row.collector_received_at for row in rows],
                "price": [_exact_8dp(row.price) for row in rows],
                "seq": [row.seq for row in rows],
                "session_id": [row.session_id for row in rows],
                "upstream_latency_ms": [row.upstream_latency_ms for row in rows],
                "transport_latency_ms": [row.transport_latency_ms for row in rows],
                "collector_latency_ms": [row.collector_latency_ms for row in rows],
                "avg_60s": [_exact_8dp(row.avg_60s) for row in rows],
                "final_15m_avg": [_exact_8dp(row.final_15m_avg) for row in rows],
                "avg_60s_window_size": [row.avg_60s_window_size for row in rows],
                "avg_60s_window_start_ms": [row.avg_60s_window_start_ms for row in rows],
                "avg_60s_window_end_exclusive_ms": [
                    row.avg_60s_window_end_exclusive_ms for row in rows
                ],
                "final_15m_window_size": [row.final_15m_window_size for row in rows],
                "final_15m_window_start_ms": [row.final_15m_window_start_ms for row in rows],
                "final_15m_window_end_exclusive_ms": [
                    row.final_15m_window_end_exclusive_ms for row in rows
                ],
                "is_stale": [row.is_stale for row in rows],
                "gap_detected": [row.gap_detected for row in rows],
                "missing_sample_count": [row.missing_sample_count for row in rows],
                "raw_data_json": [row.raw_data_json for row in rows],
                "event_sha256": [row.event_sha256 for row in rows],
            }
            table = pa.Table.from_pydict(data, schema=schema)
            try:
                pq.write_table(table, temporary_path, compression=self.compression)
                with temporary_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_path, final_path)
                digest = hashlib.sha256()
                with final_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                by_index: dict[str, RtiEvent] = {}
                for row in rows:
                    previous = by_index.get(row.index_id)
                    if previous is None or row.source_timestamp_ms > previous.source_timestamp_ms:
                        by_index[row.index_id] = row
                seqs = [row.seq for row in rows if row.seq is not None]
                self.store.register_file_and_checkpoints(
                    file_id=file_id, dataset="rti_events",
                    path=str(final_path.relative_to(self.root)), sha256=digest.hexdigest(),
                    row_count=len(rows), min_event_time=rows[0].source_time_utc,
                    max_event_time=rows[-1].source_time_utc,
                    min_seq=min(seqs) if seqs else None,
                    max_seq=max(seqs) if seqs else None,
                    session_id=rows[-1].session_id,
                    checkpoints=[
                        (index_id, row.source_timestamp_ms, row.seq, row.event_sha256)
                        for index_id, row in by_index.items()
                    ],
                    byte_size=final_path.stat().st_size,
                    config_sha256=self.config_sha256,
                    partition={"date": day, "asset": asset},
                )
            finally:
                temporary_path.unlink(missing_ok=True)
            total += len(rows)
        return total
