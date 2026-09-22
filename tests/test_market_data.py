from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from kaishi_bot.market_data import MarketDiscoveryError, PublicKalshiMarketData


def client_for(payloads: dict[str, dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path
        return httpx.Response(200, json=payloads[key])
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://example.test")


@pytest.mark.asyncio
async def test_discovers_the_single_market_open_at_observed_time() -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    payload = {"markets": [{
        "ticker": "BTC-OPEN", "title": "BTC up?", "status": "active",
        "open_time": (now - timedelta(minutes=1)).isoformat(),
        "close_time": (now + timedelta(minutes=14)).isoformat(),
        "yes_bid_dollars": "0.5100", "yes_ask_dollars": "0.5200",
    }]}
    async with client_for({"/markets": payload}) as client:
        data = PublicKalshiMarketData(client)
        found = await data.discover("BTC", "KXBTC15M", now)
    assert found.ticker == "BTC-OPEN"
    assert found.asset == "BTC"


@pytest.mark.asyncio
async def test_ambiguous_active_markets_fail_closed() -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    item = {
        "ticker": "ONE", "status": "active",
        "open_time": (now - timedelta(minutes=1)).isoformat(),
        "close_time": (now + timedelta(minutes=14)).isoformat(),
    }
    async with client_for({"/markets": {"markets": [item, {**item, "ticker": "TWO"}]}}) as client:
        with pytest.raises(MarketDiscoveryError, match="ambiguous"):
            await PublicKalshiMarketData(client).discover("BTC", "KXBTC15M", now)


@pytest.mark.asyncio
async def test_quotes_prefer_fixed_point_and_derive_no_book() -> None:
    payload = {"markets": [{
        "ticker": "BTC-OPEN", "yes_bid_dollars": "0.5100", "yes_ask_dollars": "0.5200"
    }]}
    async with client_for({"/markets": payload}) as client:
        quotes = await PublicKalshiMarketData(client).quotes(["BTC-OPEN"])
    quote = quotes["BTC-OPEN"]
    assert quote.up_bid == Decimal("0.5100")
    assert quote.up_ask == Decimal("0.5200")
    assert quote.down_bid == Decimal("0.4800")
    assert quote.down_ask == Decimal("0.4900")


@pytest.mark.asyncio
async def test_quotes_support_legacy_integer_cents() -> None:
    payload = {"markets": [{"ticker": "BTC-OPEN", "yes_bid": 51, "yes_ask": 52}]}
    async with client_for({"/markets": payload}) as client:
        quote = (await PublicKalshiMarketData(client).quotes(["BTC-OPEN"]))["BTC-OPEN"]
    assert quote.up_bid == Decimal("0.51")


@pytest.mark.asyncio
async def test_market_result_maps_yes_and_no_to_user_sides() -> None:
    async with client_for({"/markets/BTC-OPEN": {"market": {
        "ticker": "BTC-OPEN", "status": "settled", "result": "yes"
    }}}) as client:
        result = await PublicKalshiMarketData(client).result("BTC-OPEN")
    assert result == "up"


@pytest.mark.asyncio
async def test_fee_schedule_reads_verified_quadratic_series_metadata_and_caches() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"series": {
            "ticker": "KXBTC15M", "fee_type": "quadratic", "fee_multiplier": 1,
            "last_updated_ts": "2026-07-01T18:05:04Z",
        }})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as client:
        data = PublicKalshiMarketData(client)
        first = await data.fee_schedule("KXBTC15M")
        second = await data.fee_schedule("KXBTC15M")
    assert first == second
    assert first.fee_type == "quadratic"
    assert first.multiplier == Decimal("1")
    assert first.taker_rate == Decimal("0.07")
    assert first.version == "2026-07-01T18:05:04Z"
    assert calls == 1


@pytest.mark.asyncio
async def test_fee_schedule_refreshes_metadata_for_each_new_run() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"series": {
            "fee_type": "quadratic", "fee_multiplier": calls,
            "last_updated_ts": f"version-{calls}",
        }})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as client:
        data = PublicKalshiMarketData(client)
        first = await data.fee_schedule("KXBTC15M")
        second = await data.fee_schedule("KXBTC15M", refresh=True)

    assert first.multiplier == Decimal("1")
    assert second.multiplier == Decimal("2")
    assert calls == 2


@pytest.mark.asyncio
async def test_fee_schedule_rejects_unknown_or_incomplete_metadata() -> None:
    async with client_for({"/series/KXBTC15M": {"series": {
        "fee_type": "flat", "fee_multiplier": 1,
    }}}) as client:
        with pytest.raises(ValueError, match="unsupported fee type"):
            await PublicKalshiMarketData(client).fee_schedule("KXBTC15M")


@pytest.mark.asyncio
async def test_orderbook_derives_executable_asks_and_sums_depth() -> None:
    payload = {"orderbook_fp": {
        "yes_dollars": [["0.79", "3.00"], ["0.78", "4.00"]],
        "no_dollars": [["0.80", "2.00"], ["0.79", "5.00"]],
    }}
    async with client_for({"/markets/BTC-OPEN/orderbook": payload}) as client:
        book = await PublicKalshiMarketData(client).orderbook("BTC-OPEN")

    assert book.up_asks[0].price == Decimal("0.20")
    assert book.down_asks[0].price == Decimal("0.21")
    assert book.available("up", Decimal("0.21")) == Decimal("7.00")
    assert book.available("down", Decimal("0.21")) == Decimal("3.00")
