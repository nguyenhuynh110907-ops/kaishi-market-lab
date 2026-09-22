from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from kaishi_bot.research_features import FeatureObservation
from kaishi_bot.research_parquet import ParquetUnavailableError
from kaishi_bot.research_store import ResearchStore


class ParquetFeatureSink:
    def __init__(
        self, root: Path, store: ResearchStore, compression: str = "zstd",
        config_sha256: str | None = None,
    ) -> None:
        self.root = root
        self.store = store
        self.compression = None if compression == "none" else compression
        self.config_sha256 = config_sha256

    def write(self, rows: list[FeatureObservation], *, split: str) -> int:
        if split not in {"train", "validation", "test", "unassigned"}:
            raise ValueError("invalid feature split")
        if not rows:
            return 0
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ParquetUnavailableError("feature export requires research extra") from error
        groups: dict[tuple[str, str, str], list[FeatureObservation]] = defaultdict(list)
        for row in rows:
            day = row.observation_time.date().isoformat()
            groups[(row.dataset_version, day, row.asset)].append(row)
        for (version, day, asset), items in groups.items():
            directory = (
                self.root / "feature_observations" / f"dataset_version={version}"
                / f"split={split}" / f"date={day}" / f"asset={asset}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            file_id = str(uuid.uuid4())
            temporary = directory / f".{file_id}.parquet.tmp"
            final = directory / f"part-{file_id}.parquet"
            records = []
            for row in sorted(items, key=lambda item: (item.available_at, item.ticker)):
                record = asdict(row)
                record["contract_history_json"] = json.dumps(
                    {key: str(value) if value is not None else None
                     for key, value in row.contract_history}, sort_keys=True,
                    separators=(",", ":"),
                )
                record.pop("contract_history")
                record["incomplete_reasons"] = list(row.incomplete_reasons)
                records.append(record)
            table = pa.Table.from_pylist(records)
            try:
                pq.write_table(table, temporary, compression=self.compression)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, final)
                content = final.read_bytes()
                parent_ids = sorted({
                    manifest for row in items for manifest in (
                        row.market_manifest_id, row.rti_manifest_id,
                        row.quote_manifest_id, row.orderbook_manifest_id,
                    ) if manifest
                })
                latest = max(items, key=lambda item: item.available_at)
                self.store.register_file_and_checkpoints(
                    file_id=file_id, dataset="feature_observations",
                    path=str(final.relative_to(self.root)),
                    sha256=hashlib.sha256(content).hexdigest(), row_count=len(items),
                    min_event_time=min(item.available_at for item in items),
                    max_event_time=max(item.available_at for item in items),
                    min_seq=None, max_seq=None, session_id="feature-builder",
                    checkpoints=[(
                        latest.ticker, int(latest.available_at.timestamp() * 1000),
                        None, hashlib.sha256(latest.ticker.encode()).hexdigest(),
                    )],
                    schema_version=1, byte_size=len(content),
                    config_sha256=self.config_sha256,
                    partition={"dataset_version": version, "split": split,
                               "date": day, "asset": asset},
                    parent_manifest_ids=parent_ids,
                )
            finally:
                temporary.unlink(missing_ok=True)
        return len(rows)
