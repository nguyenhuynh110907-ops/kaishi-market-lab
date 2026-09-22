from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BotConfig(BaseModel):
    """Validated strategy settings with intentionally safe defaults."""

    model_config = ConfigDict(extra="forbid")

    series: str = "KXBTC15M"
    entry_price: Decimal = Field(default=Decimal("0.25"), gt=0, lt=1)
    take_profit_price: Decimal = Field(default=Decimal("0.40"), gt=0, lt=1)
    contracts: Decimal = Field(default=Decimal("1"), gt=0)
    trade_up: bool = True
    trade_down: bool = True
    min_seconds_before_close: int = Field(default=60, ge=10)

    @model_validator(mode="after")
    def validate_strategy(self) -> "BotConfig":
        if not (self.trade_up or self.trade_down):
            raise ValueError("at least one side must be enabled")
        if self.take_profit_price <= self.entry_price:
            raise ValueError("take profit must be greater than entry")
        return self

    @classmethod
    def load(cls, path: Path) -> "BotConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)


class Credentials(BaseModel):
    """Kalshi Demo credentials loaded independently from strategy config."""

    key_id: str
    private_key_path: Path

    @classmethod
    def load(cls, environ: Mapping[str, str]) -> "Credentials":
        key_id = environ.get("KALSHI_KEY_ID", "")
        raw_key_path = environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        key_path = Path(raw_key_path).expanduser() if raw_key_path else Path()
        if not key_id:
            raise ValueError("KALSHI_KEY_ID is required")
        if not raw_key_path or not key_path.is_file():
            raise ValueError("KALSHI_PRIVATE_KEY_PATH must point to a readable file")
        return cls(key_id=key_id, private_key_path=key_path)
