from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaishi_bot.entry_guard import EntryGuardSettings


class RuntimeMode(StrEnum):
    PAPER = "paper"
    READ_ONLY = "read_only"
    LIVE = "live"


class AssetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    series: str
    enabled: bool = True
    trade_up: bool = True
    trade_down: bool = True


def default_assets() -> dict[str, AssetSettings]:
    return {
        "BTC": AssetSettings(series="KXBTC15M"),
        "ETH": AssetSettings(series="KXETH15M"),
        "SOL": AssetSettings(series="KXSOL15M"),
        "XRP": AssetSettings(series="KXXRP15M"),
        "DOGE": AssetSettings(series="KXDOGE15M"),
    }


class DashboardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: RuntimeMode = RuntimeMode.PAPER
    bot_enabled: bool = False
    log_quotes: bool = True
    stop_loss: Decimal = Field(default=Decimal("0.15"), gt=0, lt=1)
    entry_min: Decimal | None = Field(default=None, gt=0, lt=1)
    entry_price: Decimal = Field(default=Decimal("0.25"), gt=0, lt=1)
    take_profit: Decimal = Field(default=Decimal("0.40"), gt=0, lt=1)
    entry_amount: Decimal = Field(default=Decimal("10.00"), gt=0, le=200)
    paper_daily_cap: Decimal = Field(default=Decimal("1000.00"), gt=0, le=10000)
    daily_cap: Decimal = Field(default=Decimal("1000.00"), gt=0, le=10000)
    min_seconds_before_close: int = Field(default=60, ge=10, le=840)
    entry_start_seconds: int = Field(default=0, ge=0, le=885)
    entry_end_seconds: int = Field(default=900, ge=15, le=900)
    entry_guard: EntryGuardSettings = Field(default_factory=EntryGuardSettings)
    assets: dict[str, AssetSettings] = Field(default_factory=default_assets)

    @model_validator(mode="after")
    def validate_strategy(self) -> "DashboardSettings":
        if self.entry_start_seconds >= self.entry_end_seconds:
            raise ValueError("entry start must be before entry end")
        if self.entry_min is None:
            self.entry_min = max(
                self.entry_price * self.entry_guard.entry_floor_ratio,
                self.stop_loss + self.entry_guard.stop_loss_buffer,
            )
        if not self.stop_loss < self.entry_min <= self.entry_price < self.take_profit:
            raise ValueError(
                "require stop_loss < entry_min <= entry_price < take_profit"
            )
        expected = {"BTC", "ETH", "SOL", "XRP", "DOGE"}
        if set(self.assets) != expected:
            raise ValueError("assets must be exactly BTC, ETH, SOL, XRP, DOGE")
        return self


class QuotePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed_at: datetime
    up_bid: Decimal = Field(ge=0, le=1)
    up_ask: Decimal = Field(ge=0, le=1)
    down_bid: Decimal = Field(ge=0, le=1)
    down_ask: Decimal = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_books(self) -> "QuotePoint":
        if self.up_bid > self.up_ask or self.down_bid > self.down_ask:
            raise ValueError("bid cannot exceed ask")
        return self


class AssetMarket(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: str
    series: str
    ticker: str
    title: str = ""
    open_time: datetime
    close_time: datetime
    target: str | None = None


class AssetSnapshot(BaseModel):
    asset: str
    market: AssetMarket | None = None
    quote: QuotePoint | None = None
    chart: list[QuotePoint] = Field(default_factory=list)
    status: str = "discovering"
    error: str | None = None
