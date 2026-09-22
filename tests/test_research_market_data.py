import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from kaishi_bot.research_market_data import (
    ResearchMarketClient,
    parse_fee_history,
    parse_market_payload,
)


def market_payload(now: datetime) -> dict[str, object]:
    return {
        "ticker": "KXBTC15M-TEST",
        "series_ticker": "KXBTC15M",
        "title": "BTC 15 minute",
        "status": "open",
        "open_time": now.isoformat(),
        "close_time": (now + timedelta(minutes=15)).isoformat(),
        "floor_strike": Decimal("63310.36000000"),
        "rules_primary": "official rule",
        "rules_secondary": "official settlement detail",
    }


def test_market_parser_keeps_exact_target_and_official_window_boundaries() -> None:
    now = datetime(2026, 8, 11, 19, 0, tzinfo=UTC)
    market = parse_market_payload("BTC", "KXBTC15M", market_payload(now), now)
    assert market.target_price == Decimal("63310.36000000")
    assert market.target_source_field == "floor_strike"
    assert market.target_window_start == now - timedelta(seconds=60)
    assert market.target_window_end == now
    assert market.settlement_window_start == now + timedelta(minutes=14)
    assert market.settlement_window_end == now + timedelta(minutes=15)


@pytest.mark.asyncio
async def test_market_client_parses_json_numbers_as_decimal_without_float_roundtrip() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    raw = {
        **market_payload(now),
        "floor_strike": 63310.36000001,
    }
    body = json.dumps(raw, default=str).replace('"63310.36000000"', "63310.36000000")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"markets": []}))

    exact_body = (
        '{"markets":[{"ticker":"KXBTC15M-TEST","series_ticker":"KXBTC15M",'
        '"title":"BTC","status":"open","open_time":"'
        + now.isoformat()
        + '","close_time":"'
        + (now + timedelta(minutes=15)).isoformat()
        + '","floor_strike":63310.36000001}]}'
    )

    def exact_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=exact_body)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(exact_handler), base_url="https://example.test"
    ) as client:
        market = await ResearchMarketClient(client).active("BTC", "KXBTC15M")
    assert market is not None
    assert market.target_price == Decimal("63310.36000001")
def test_fee_history_builds_non_overlapping_effective_versions() -> None:
    observed = datetime(2026, 8, 11, 12, tzinfo=UTC)
    versions = parse_fee_history("KXBTC15M", [
        {"id": "new", "scheduled_ts": "2026-08-10T00:00:00Z",
         "fee_type": "quadratic", "fee_multiplier": 2},
        {"id": "old", "scheduled_ts": "2026-08-01T00:00:00Z",
         "fee_type": "quadratic", "fee_multiplier": 1},
    ], observed)
    assert [item.source_change_id for item in versions] == ["old", "new"]
    assert versions[0].effective_to == versions[1].effective_from
    assert versions[1].effective_to is None
    assert versions[0].taker_rate == Decimal("0.07")


def test_current_fee_without_timestamp_does_not_backdate_history() -> None:
    observed = datetime(2026, 8, 11, 12, tzinfo=UTC)
    version = parse_fee_history("KXBTC15M", [
        {"fee_type": "quadratic", "fee_multiplier": 1},
    ], observed, source_kind="series_current")[0]
    assert version.effective_from == observed
