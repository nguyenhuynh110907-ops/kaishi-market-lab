from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

from kaishi_bot.dashboard_models import AssetMarket, QuotePoint
from kaishi_bot.fees import FeeSchedule


PRODUCTION_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class MarketDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DepthLevel:
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    ticker: str
    observed_at: datetime
    up_asks: tuple[DepthLevel, ...]
    down_asks: tuple[DepthLevel, ...]

    def available(self, side: str, limit_price: Decimal) -> Decimal:
        levels = self.up_asks if str(getattr(side, "value", side)) == "up" else self.down_asks
        return sum(
            (level.quantity for level in levels if level.price <= limit_price),
            Decimal("0"),
        )


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _price(payload: dict[str, Any], dollars: str, cents: str) -> Decimal | None:
    value = payload.get(dollars)
    if value is not None and value != "":
        return Decimal(str(value))
    value = payload.get(cents)
    if value is not None:
        return Decimal(str(value)) / Decimal("100")
    return None


class PublicKalshiMarketData:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self._fee_schedules: dict[str, FeeSchedule] = {}

    @classmethod
    def production(cls) -> "PublicKalshiMarketData":
        return cls(httpx.AsyncClient(base_url=PRODUCTION_BASE_URL, timeout=5.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def discover(
        self, asset: str, series: str, now: datetime | None = None
    ) -> AssetMarket:
        observed_at = now or datetime.now(UTC)
        response = await self.client.get(
            "/markets", params={"status": "open", "series_ticker": series, "limit": 200}
        )
        response.raise_for_status()
        candidates = []
        for item in response.json().get("markets", []):
            if item.get("status") not in {"open", "active"}:
                continue
            open_time = _time(item["open_time"])
            close_time = _time(item["close_time"])
            if open_time <= observed_at < close_time:
                candidates.append((item, open_time, close_time))
        if not candidates:
            raise MarketDiscoveryError(f"no active {series} market")
        if len(candidates) != 1:
            tickers = ", ".join(sorted(str(item[0]["ticker"]) for item in candidates))
            raise MarketDiscoveryError(f"ambiguous active {series} markets: {tickers}")
        item, open_time, close_time = candidates[0]
        target = item.get("floor_strike_dollars") or item.get("subtitle")
        return AssetMarket(
            asset=asset, series=series, ticker=str(item["ticker"]),
            title=str(item.get("title", "")), open_time=open_time,
            close_time=close_time, target=str(target) if target is not None else None,
        )

    async def quotes(self, tickers: list[str]) -> dict[str, QuotePoint]:
        if not tickers:
            return {}
        response = await self.client.get(
            "/markets", params={"tickers": ",".join(tickers), "limit": len(tickers)}
        )
        response.raise_for_status()
        observed_at = datetime.now(UTC)
        result: dict[str, QuotePoint] = {}
        for item in response.json().get("markets", []):
            yes_bid = _price(item, "yes_bid_dollars", "yes_bid")
            yes_ask = _price(item, "yes_ask_dollars", "yes_ask")
            if yes_bid is None or yes_ask is None:
                raise ValueError(f"missing executable quote for {item.get('ticker', 'unknown')}")
            no_bid = _price(item, "no_bid_dollars", "no_bid")
            no_ask = _price(item, "no_ask_dollars", "no_ask")
            result[str(item["ticker"])] = QuotePoint(
                observed_at=observed_at,
                up_bid=yes_bid,
                up_ask=yes_ask,
                down_bid=no_bid if no_bid is not None else Decimal("1") - yes_ask,
                down_ask=no_ask if no_ask is not None else Decimal("1") - yes_bid,
            )
        return result

    async def result(self, ticker: str) -> str | None:
        response = await self.client.get(f"/markets/{ticker}")
        response.raise_for_status()
        market = response.json().get("market", {})
        result = str(market.get("result", "")).lower()
        if result == "yes":
            return "up"
        if result == "no":
            return "down"
        return None

    async def orderbook(self, ticker: str, depth: int = 100) -> OrderBookSnapshot:
        response = await self.client.get(
            f"/markets/{ticker}/orderbook", params={"depth": depth}
        )
        response.raise_for_status()
        payload = response.json()
        book = payload.get("orderbook_fp") or payload.get("orderbook") or {}

        def levels(name: str) -> list[tuple[Decimal, Decimal]]:
            result: list[tuple[Decimal, Decimal]] = []
            for raw_price, raw_quantity in book.get(name, []):
                price = Decimal(str(raw_price))
                quantity = Decimal(str(raw_quantity))
                if "dollars" not in name:
                    price /= Decimal("100")
                if not Decimal("0") <= price <= Decimal("1") or quantity <= 0:
                    raise ValueError("invalid orderbook level")
                result.append((price, quantity))
            return result

        yes_bids = levels("yes_dollars" if "yes_dollars" in book else "yes")
        no_bids = levels("no_dollars" if "no_dollars" in book else "no")
        up_asks = tuple(sorted(
            (DepthLevel(Decimal("1") - price, quantity) for price, quantity in no_bids),
            key=lambda item: item.price,
        ))
        down_asks = tuple(sorted(
            (DepthLevel(Decimal("1") - price, quantity) for price, quantity in yes_bids),
            key=lambda item: item.price,
        ))
        return OrderBookSnapshot(
            ticker=ticker,
            observed_at=datetime.now(UTC),
            up_asks=up_asks,
            down_asks=down_asks,
        )

    async def fee_schedule(self, series: str, *, refresh: bool = False) -> FeeSchedule:
        cached = self._fee_schedules.get(series)
        if cached is not None and not refresh:
            return cached
        response = await self.client.get(f"/series/{series}")
        response.raise_for_status()
        payload = response.json().get("series", {})
        fee_type = str(payload.get("fee_type", ""))
        if fee_type != "quadratic":
            raise ValueError(f"unsupported fee type: {fee_type or 'missing'}")
        multiplier = payload.get("fee_multiplier")
        version = payload.get("last_updated_ts")
        if multiplier is None or not version:
            raise ValueError(f"incomplete fee metadata for {series}")
        schedule = FeeSchedule(
            fee_type=fee_type,
            multiplier=Decimal(str(multiplier)),
            taker_rate=Decimal("0.07"),
            version=str(version),
        )
        self._fee_schedules[series] = schedule
        return schedule
