from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


def utc_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class ResearchMarket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset: str
    series_ticker: str
    ticker: str
    title: str = ""
    open_time: datetime
    close_time: datetime
    target_price: Decimal | None = None
    target_source_field: str | None = None
    rules_primary: str | None = None
    rules_secondary: str | None = None
    official_result: str | None = None
    expiration_value: Decimal | None = None
    settlement_value: Decimal | None = None
    settlement_ts: datetime | None = None
    discovered_at: datetime
    refreshed_at: datetime
    result_observed_at: datetime | None = None
    raw_payload: dict[str, Any]
    payload_sha256: str

    @model_validator(mode="after")
    def normalize_times(self) -> "ResearchMarket":
        for field in (
            "open_time", "close_time", "discovered_at", "refreshed_at",
            "settlement_ts", "result_observed_at",
        ):
            value = getattr(self, field)
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.close_time <= self.open_time:
            raise ValueError("market close must follow open")
        return self

    @property
    def target_window_start(self) -> datetime:
        return utc_time(self.open_time) - timedelta(seconds=60)

    @property
    def target_window_end(self) -> datetime:
        return utc_time(self.open_time)

    @property
    def settlement_window_start(self) -> datetime:
        return utc_time(self.close_time) - timedelta(seconds=60)

    @property
    def settlement_window_end(self) -> datetime:
        return utc_time(self.close_time)

    @field_serializer(
        "target_price", "expiration_value", "settlement_value", when_used="json"
    )
    def serialize_decimal(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None


class RtiEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset: str
    index_id: str
    source_timestamp_ms: int = Field(gt=0)
    source_time_utc: datetime
    kalshi_received_at: datetime
    collector_received_at: datetime
    price: Decimal
    seq: int | None = None
    session_id: str
    upstream_latency_ms: int
    transport_latency_ms: int
    collector_latency_ms: int
    avg_60s: Decimal | None = None
    final_15m_avg: Decimal | None = None
    avg_60s_window_size: int | None = None
    avg_60s_window_start_ms: int | None = None
    avg_60s_window_end_exclusive_ms: int | None = None
    final_15m_window_size: int | None = None
    final_15m_window_start_ms: int | None = None
    final_15m_window_end_exclusive_ms: int | None = None
    is_stale: bool = False
    gap_detected: bool = False
    missing_sample_count: int = 0
    raw_data_json: str
    event_sha256: str

    @model_validator(mode="after")
    def validate_event(self) -> "RtiEvent":
        for field in ("source_time_utc", "kalshi_received_at", "collector_received_at"):
            if getattr(self, field).utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.missing_sample_count < 0:
            raise ValueError("missing sample count cannot be negative")
        return self

    @field_serializer("price", "avg_60s", "final_15m_avg", when_used="json")
    def serialize_decimal(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None


class BookLevel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    price: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    quantity: Decimal = Field(gt=Decimal("0"))


class OrderBookEvent(BaseModel):
    """One immutable Kalshi snapshot/delta envelope.

    Snapshots preserve both level arrays in one row. Deltas use side/price and
    signed_quantity_delta. ``available_at`` is the replay visibility time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    asset: str
    event_type: Literal["snapshot", "delta"]
    source_timestamp: datetime | None = None
    collector_received_at: datetime
    available_at: datetime
    session_id: str
    sequence: int
    side: Literal["yes", "no"] | None = None
    price: Decimal | None = None
    quantity: Decimal | None = None
    signed_quantity_delta: Decimal | None = None
    yes_levels: tuple[BookLevel, ...] = ()
    no_levels: tuple[BookLevel, ...] = ()
    gap_detected: bool = False
    book_valid_after_event: bool = False
    price_convention: str = "legacy_side_price_v1"
    raw_payload_json: str
    event_sha256: str
    stable_row_id: str

    @model_validator(mode="after")
    def validate_shape(self) -> "OrderBookEvent":
        for field in ("collector_received_at", "available_at", "source_timestamp"):
            value = getattr(self, field)
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.event_type == "delta" and (
            self.side is None or self.price is None
            or self.signed_quantity_delta is None
        ):
            raise ValueError("delta requires side, price, and signed quantity")
        if self.event_type == "snapshot" and self.side is not None:
            raise ValueError("snapshot cannot have a delta side")
        return self


class ContractQuoteEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    asset: str
    series_ticker: str
    source_timestamp: datetime | None = None
    collector_received_at: datetime
    available_at: datetime
    up_bid: Decimal | None = None
    up_ask: Decimal | None = None
    down_bid: Decimal | None = None
    down_ask: Decimal | None = None
    up_spread: Decimal | None = None
    down_spread: Decimal | None = None
    book_sequence: int
    collector_session_id: str
    event_kind: Literal["top_change", "heartbeat", "resnapshot", "ticker"] = "top_change"
    is_stale: bool = False
    gap_detected: bool = False
    book_valid: bool = False
    price_convention: str = "legacy_side_price_v1"
    event_sha256: str
    stable_row_id: str
    raw_payload_json: str | None = None

    @model_validator(mode="after")
    def validate_quote(self) -> "ContractQuoteEvent":
        for field in ("collector_received_at", "available_at", "source_timestamp"):
            value = getattr(self, field)
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        return self


class BookCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    asset: str
    available_at: datetime
    session_id: str
    last_sequence: int
    yes_levels: tuple[BookLevel, ...]
    no_levels: tuple[BookLevel, ...]
    up_best_bid: Decimal | None = None
    up_best_ask: Decimal | None = None
    down_best_bid: Decimal | None = None
    down_best_ask: Decimal | None = None
    depth_1c: Decimal = Decimal("0")
    depth_3c: Decimal = Decimal("0")
    depth_5c: Decimal = Decimal("0")
    total_visible_depth: Decimal = Decimal("0")
    book_imbalance: Decimal | None = None
    is_complete: bool
    incomplete_reasons: tuple[str, ...] = ()
    state_sha256: str


class FeeMetadataVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fee_version: str
    series_ticker: str
    fee_type: str
    fee_multiplier: Decimal
    taker_rate: Decimal
    maker_rate: Decimal | None = None
    effective_from: datetime
    effective_to: datetime | None = None
    observed_at: datetime
    source_kind: str
    source_change_id: str | None = None
    source_payload_hash: str
    raw_payload_json: str

    @model_validator(mode="after")
    def validate_version(self) -> "FeeMetadataVersion":
        for field in ("effective_from", "effective_to", "observed_at"):
            value = getattr(self, field)
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("fee effective_to must follow effective_from")
        if self.fee_multiplier <= 0 or self.taker_rate < 0:
            raise ValueError("fee rates must be valid")
        return self
