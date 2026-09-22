from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_INDEX_IDS = {
    "BTC": "BRTI",
    "ETH": "ETHUSD_RTI",
    "SOL": "SOLUSD_RTI",
    "XRP": "XRPUSD_RTI",
    "DOGE": "DOGEUSD_RTI",
}

DEFAULT_SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "DOGE": "KXDOGE15M",
}


class ResearchCaptureConfig(BaseModel):
    """Configuration for the research data plane only.

    This model deliberately contains no trading mode, arm state, or order
    setting. A missing file is equivalent to a disabled collector.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    root: Path = Path("data/research")
    assets: tuple[str, ...] = tuple(DEFAULT_INDEX_IDS)
    index_ids: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_INDEX_IDS))
    series: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SERIES))
    persist_market_metadata: bool = True
    persist_rti: bool = True
    persist_contract_quotes: bool = True
    persist_orderbook: bool = False
    orderbook_depth: int = Field(default=10, ge=10, le=100)
    compression: str = "zstd"
    writer_batch_rows: int = Field(default=50_000, ge=1, le=1_000_000)
    flush_interval_seconds: float = Field(default=300, gt=0, le=900)
    metadata_refresh_seconds: float = Field(default=15, ge=5, le=900)
    stale_after_seconds: float = Field(default=3, gt=0, le=60)
    queue_maxsize: int = Field(default=10_000, ge=100, le=1_000_000)

    @model_validator(mode="after")
    def validate_assets(self) -> "ResearchCaptureConfig":
        if not self.assets:
            raise ValueError("research assets cannot be empty")
        if len(self.assets) != len(set(self.assets)):
            raise ValueError("research assets must be unique")
        unknown = set(self.assets) - set(DEFAULT_INDEX_IDS)
        if unknown:
            raise ValueError(f"unsupported research assets: {', '.join(sorted(unknown))}")
        missing_indexes = set(self.assets) - set(self.index_ids)
        missing_series = set(self.assets) - set(self.series)
        if missing_indexes or missing_series:
            raise ValueError("each research asset requires an index id and series")
        if self.compression not in {"zstd", "snappy", "gzip", "none"}:
            raise ValueError("unsupported research compression")
        return self

    @classmethod
    def load(cls, path: Path | None) -> "ResearchCaptureConfig":
        if path is None or not path.is_file():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "research_capture" in raw:
            raw = raw["research_capture"] or {}
        config = cls.model_validate(raw)
        if not config.root.is_absolute():
            config = config.model_copy(update={"root": path.parent / config.root})
        return config
