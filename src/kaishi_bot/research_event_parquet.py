from __future__ import annotations

import hashlib
import os
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from kaishi_bot.research_models import (
    BookCheckpoint,
    ContractQuoteEvent,
    OrderBookEvent,
)
from kaishi_bot.research_parquet import ParquetUnavailableError
from kaishi_bot.research_store import ResearchStore


PRICE = Decimal("0.0001")
QUANTITY = Decimal("0.01")


def _price(value: Decimal | None) -> Decimal | None:
    return value.quantize(PRICE) if value is not None else None


def _quantity(value: Decimal | None) -> Decimal | None:
    return value.quantize(QUANTITY) if value is not None else None


class ParquetCaptureSink:
    """Atomic immutable writer for quote, raw-book, and checkpoint datasets."""

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

    def write(
        self, *, quotes: Iterable[ContractQuoteEvent] = (),
        books: Iterable[OrderBookEvent] = (),
        checkpoints: Iterable[BookCheckpoint] = (),
    ) -> int:
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ParquetUnavailableError(
                "research capture requires the 'research' optional dependency"
            ) from error
        total = 0
        total += self._partition_write(
            "contract_quote_events", list(quotes), self._quote_schema(pa), self._quote_row
        )
        total += self._partition_write(
            "orderbook_events", list(books), self._book_schema(pa), self._book_row
        )
        total += self._partition_write(
            "book_checkpoints", list(checkpoints), self._checkpoint_schema(pa),
            self._checkpoint_row,
        )
        return total

    def _partition_write(self, dataset: str, rows: list[Any], schema, convert) -> int:
        if not rows:
            return 0
        groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for row in rows:
            day = row.available_at.astimezone(UTC).date().isoformat()
            groups[(day, row.asset)].append(row)
        for (day, asset), items in groups.items():
            items.sort(key=lambda row: (
                row.available_at,
                getattr(row, "sequence", getattr(row, "book_sequence", -1)),
                getattr(row, "stable_row_id", getattr(row, "state_sha256", "")),
            ))
            self._write_file(dataset, day, asset, items, schema, convert)
        return len(rows)

    def _write_file(self, dataset, day, asset, rows, schema, convert) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        directory = self.root / dataset / f"date={day}" / f"asset={asset}"
        directory.mkdir(parents=True, exist_ok=True)
        file_id = str(uuid.uuid4())
        final_path = directory / f"part-{file_id}.parquet"
        temporary = directory / f".{file_id}.parquet.tmp"
        table = pa.Table.from_pylist([convert(row) for row in rows], schema=schema)
        try:
            pq.write_table(table, temporary, compression=self.compression)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, final_path)
            content = final_path.read_bytes()
            sequences = [
                int(getattr(row, "sequence", getattr(row, "book_sequence", 0)))
                for row in rows
            ]
            latest = rows[-1]
            last_ms = int(latest.available_at.timestamp() * 1000)
            last_hash = str(
                getattr(latest, "event_sha256", getattr(latest, "state_sha256", ""))
            )
            self.store.register_file_and_checkpoints(
                file_id=file_id, dataset=dataset,
                path=str(final_path.relative_to(self.root)),
                sha256=hashlib.sha256(content).hexdigest(), row_count=len(rows),
                min_event_time=rows[0].available_at,
                max_event_time=rows[-1].available_at,
                min_seq=min(sequences), max_seq=max(sequences),
                session_id=str(getattr(latest, "session_id", getattr(latest, "collector_session_id", ""))),
                checkpoints=[(latest.ticker, last_ms, max(sequences), last_hash)],
                schema_version=1, byte_size=len(content),
                config_sha256=self.config_sha256,
                partition={"date": day, "asset": asset},
            )
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _quote_schema(pa):
        return pa.schema([
            ("ticker", pa.string()), ("series_ticker", pa.string()),
            ("source_timestamp", pa.timestamp("us", tz="UTC")),
            ("collector_received_at", pa.timestamp("us", tz="UTC")),
            ("available_at", pa.timestamp("us", tz="UTC")),
            ("up_bid", pa.decimal128(12, 4)), ("up_ask", pa.decimal128(12, 4)),
            ("down_bid", pa.decimal128(12, 4)), ("down_ask", pa.decimal128(12, 4)),
            ("up_spread", pa.decimal128(12, 4)), ("down_spread", pa.decimal128(12, 4)),
            ("book_sequence", pa.int64()), ("collector_session_id", pa.string()),
            ("event_kind", pa.string()), ("is_stale", pa.bool_()),
            ("gap_detected", pa.bool_()), ("book_valid", pa.bool_()),
            ("price_convention", pa.string()), ("event_sha256", pa.string()),
            ("stable_row_id", pa.string()), ("raw_payload_json", pa.string()),
        ])

    @staticmethod
    def _quote_row(row):
        data = row.model_dump()
        data.pop("asset")
        for name in ("up_bid", "up_ask", "down_bid", "down_ask", "up_spread", "down_spread"):
            data[name] = _price(data[name])
        return data

    @staticmethod
    def _book_schema(pa):
        level = pa.list_(pa.struct([
            ("price", pa.decimal128(12, 4)),
            ("quantity", pa.decimal128(20, 2)),
        ]))
        return pa.schema([
            ("ticker", pa.string()), ("event_type", pa.string()),
            ("source_timestamp", pa.timestamp("us", tz="UTC")),
            ("collector_received_at", pa.timestamp("us", tz="UTC")),
            ("available_at", pa.timestamp("us", tz="UTC")),
            ("session_id", pa.string()), ("sequence", pa.int64()),
            ("side", pa.string()), ("price", pa.decimal128(12, 4)),
            ("quantity", pa.decimal128(20, 2)),
            ("signed_quantity_delta", pa.decimal128(20, 2)),
            ("yes_levels", level), ("no_levels", level),
            ("gap_detected", pa.bool_()), ("book_valid_after_event", pa.bool_()),
            ("price_convention", pa.string()), ("raw_payload_json", pa.string()),
            ("event_sha256", pa.string()), ("stable_row_id", pa.string()),
        ])

    @staticmethod
    def _book_row(row):
        data = row.model_dump()
        data.pop("asset")
        data["price"] = _price(data["price"])
        data["quantity"] = _quantity(data["quantity"])
        data["signed_quantity_delta"] = _quantity(data["signed_quantity_delta"])
        for name in ("yes_levels", "no_levels"):
            data[name] = [
                {"price": _price(item["price"]), "quantity": _quantity(item["quantity"])}
                for item in data[name]
            ]
        return data

    @staticmethod
    def _checkpoint_schema(pa):
        level = pa.list_(pa.struct([
            ("price", pa.decimal128(12, 4)),
            ("quantity", pa.decimal128(20, 2)),
        ]))
        return pa.schema([
            ("ticker", pa.string()), ("available_at", pa.timestamp("us", tz="UTC")),
            ("session_id", pa.string()), ("last_sequence", pa.int64()),
            ("yes_levels", level), ("no_levels", level),
            ("up_best_bid", pa.decimal128(12, 4)),
            ("up_best_ask", pa.decimal128(12, 4)),
            ("down_best_bid", pa.decimal128(12, 4)),
            ("down_best_ask", pa.decimal128(12, 4)),
            ("depth_1c", pa.decimal128(20, 2)),
            ("depth_3c", pa.decimal128(20, 2)),
            ("depth_5c", pa.decimal128(20, 2)),
            ("total_visible_depth", pa.decimal128(20, 2)),
            ("book_imbalance", pa.decimal128(12, 8)),
            ("is_complete", pa.bool_()),
            ("incomplete_reasons", pa.list_(pa.string())),
            ("state_sha256", pa.string()),
        ])

    @staticmethod
    def _checkpoint_row(row):
        data = row.model_dump()
        data.pop("asset")
        for name in ("up_best_bid", "up_best_ask", "down_best_bid", "down_best_ask"):
            data[name] = _price(data[name])
        for name in ("depth_1c", "depth_3c", "depth_5c", "total_visible_depth"):
            data[name] = _quantity(data[name])
        if data["book_imbalance"] is not None:
            data["book_imbalance"] = data["book_imbalance"].quantize(Decimal("0.00000001"))
        for name in ("yes_levels", "no_levels"):
            data[name] = [
                {"price": _price(item["price"]), "quantity": _quantity(item["quantity"])}
                for item in data[name]
            ]
        return data
