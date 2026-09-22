from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from kaishi_bot.market_data import PRODUCTION_BASE_URL
from kaishi_bot.research_models import FeeMetadataVersion, ResearchMarket


def _time(value: object | None) -> datetime | None:
    if value in {None, ""}:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _decimal(payload: dict[str, Any], *fields: str) -> tuple[Decimal | None, str | None]:
    for field in fields:
        value = payload.get(field)
        if value not in {None, ""}:
            return Decimal(str(value)), field
    return None, None


def canonical_payload(payload: dict[str, Any]) -> tuple[str, str]:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_market_payload(
    asset: str, series: str, payload: dict[str, Any], observed_at: datetime,
) -> ResearchMarket:
    """Create an exact, auditable market model from an official API payload."""
    raw_json, payload_hash = canonical_payload(payload)
    del raw_json  # ResearchMarket retains the structured payload; store canonicalizes it.
    target, target_field = _decimal(payload, "floor_strike", "floor_strike_dollars")
    expiration, _ = _decimal(payload, "expiration_value")
    settlement, _ = _decimal(payload, "settlement_value_dollars", "settlement_value")
    result = str(payload.get("result") or "").strip().lower() or None
    settlement_ts = _time(payload.get("settlement_ts"))
    result_observed_at = observed_at if result or expiration is not None else None
    open_time = _time(payload.get("open_time"))
    close_time = _time(payload.get("close_time"))
    if open_time is None or close_time is None:
        raise ValueError("official market payload is missing open_time or close_time")
    return ResearchMarket(
        asset=asset,
        series_ticker=str(payload.get("series_ticker") or series),
        ticker=str(payload["ticker"]),
        title=str(payload.get("title") or ""),
        open_time=open_time,
        close_time=close_time,
        target_price=target,
        target_source_field=target_field,
        rules_primary=str(payload.get("rules_primary") or "") or None,
        rules_secondary=str(payload.get("rules_secondary") or "") or None,
        official_result=result,
        expiration_value=expiration,
        settlement_value=settlement,
        settlement_ts=settlement_ts,
        discovered_at=observed_at,
        refreshed_at=observed_at,
        result_observed_at=result_observed_at,
        raw_payload=payload,
        payload_sha256=payload_hash,
    )


def parse_fee_history(
    series: str, payloads: list[dict[str, Any]], observed_at: datetime,
    *, taker_rate: Decimal = Decimal("0.07"), source_kind: str = "fee_changes",
) -> list[FeeMetadataVersion]:
    """Create provable effective intervals from official fee-change records."""
    ordered = sorted(
        payloads,
        key=lambda item: _time(item.get("scheduled_ts") or item.get("last_updated_ts"))
        or observed_at,
    )
    result: list[FeeMetadataVersion] = []
    for index, payload in enumerate(ordered):
        effective_from = _time(
            payload.get("scheduled_ts") or payload.get("last_updated_ts")
        )
        if effective_from is None:
            # A current series observation is not evidence about earlier dates.
            effective_from = observed_at
        next_from = (
            _time(ordered[index + 1].get("scheduled_ts") or ordered[index + 1].get("last_updated_ts"))
            if index + 1 < len(ordered) else None
        )
        raw, digest = canonical_payload(payload)
        fee_type = str(payload.get("fee_type") or "")
        multiplier = payload.get("fee_multiplier")
        if fee_type != "quadratic" or multiplier in {None, ""}:
            continue
        change_id = str(payload.get("id") or "") or None
        version_material = f"{series}\0{effective_from.isoformat()}\0{digest}"
        version_id = hashlib.sha256(version_material.encode()).hexdigest()
        result.append(FeeMetadataVersion(
            fee_version=version_id, series_ticker=series, fee_type=fee_type,
            fee_multiplier=Decimal(str(multiplier)), taker_rate=taker_rate,
            maker_rate=None, effective_from=effective_from, effective_to=next_from,
            observed_at=observed_at, source_kind=source_kind,
            source_change_id=change_id, source_payload_hash=digest,
            raw_payload_json=raw,
        ))
    return result


class ResearchMarketClient:
    """Public REST client isolated from the trading market-data client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    @classmethod
    def production(cls) -> "ResearchMarketClient":
        return cls(httpx.AsyncClient(base_url=PRODUCTION_BASE_URL, timeout=10.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def active(self, asset: str, series: str) -> ResearchMarket | None:
        observed_at = datetime.now(UTC)
        response = await self.client.get(
            "/markets", params={"status": "open", "series_ticker": series, "limit": 200}
        )
        response.raise_for_status()
        document = json.loads(response.content, parse_float=Decimal)
        active: list[dict[str, Any]] = []
        for payload in document.get("markets", []):
            open_time = _time(payload.get("open_time"))
            close_time = _time(payload.get("close_time"))
            if (
                payload.get("status") in {"open", "active"}
                and open_time is not None and close_time is not None
                and open_time <= observed_at < close_time
            ):
                active.append(payload)
        if not active:
            return None
        if len(active) != 1:
            raise RuntimeError(f"ambiguous active research markets for {series}")
        return parse_market_payload(asset, series, active[0], observed_at)

    async def detail(self, asset: str, series: str, ticker: str) -> ResearchMarket:
        observed_at = datetime.now(UTC)
        response = await self.client.get(f"/markets/{ticker}")
        response.raise_for_status()
        document = json.loads(response.content, parse_float=Decimal)
        return parse_market_payload(asset, series, document["market"], observed_at)

    async def fee_versions(self, series: str) -> list[FeeMetadataVersion]:
        observed_at = datetime.now(UTC)
        response = await self.client.get(
            "/series/fee_changes",
            params={"series_ticker": series, "show_historical": "true"},
        )
        response.raise_for_status()
        document = json.loads(response.content, parse_float=Decimal)
        changes = list(document.get("series_fee_change_arr") or [])
        if changes:
            return parse_fee_history(series, changes, observed_at)
        current = await self.client.get(f"/series/{series}")
        current.raise_for_status()
        current_document = json.loads(current.content, parse_float=Decimal)
        payload = dict(current_document.get("series") or {})
        return parse_fee_history(
            series, [payload], observed_at, source_kind="series_current",
        )
