from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from kaishi_bot.domain import OrderRequest
from kaishi_bot.exchange import KalshiDemoAdapter


class FakeMarkets:
    def __init__(self, markets: list[object], by_ticker: dict[str, object] | None = None):
        self.markets = markets
        self.by_ticker = by_ticker or {}
        self.list_kwargs: dict[str, object] = {}

    def list(self, **kwargs: object) -> list[object]:
        self.list_kwargs = kwargs
        return self.markets

    def get(self, ticker: str) -> object:
        return self.by_ticker[ticker]


class FakeOrders:
    def __init__(self, existing: list[object] | None = None):
        self.requests: list[object] = []
        self.existing = existing or []

    def create_v2(self, *, request: object) -> object:
        self.requests.append(request)
        return SimpleNamespace(
            order_id="exchange-order-1",
            client_order_id=request.client_order_id,
        )

    def list_all(self, **_: object):
        yield from self.existing


class FakePortfolio:
    def __init__(self, fills: list[object] | None = None):
        self._fills = fills or []

    def fills_all(self, **_: object):
        yield from self._fills


class FakeClient:
    def __init__(
        self,
        *,
        markets: FakeMarkets,
        orders: FakeOrders | None = None,
        portfolio: FakePortfolio | None = None,
    ) -> None:
        self.markets = markets
        self.orders = orders or FakeOrders()
        self.portfolio = portfolio or FakePortfolio()


class FakeWebSocket:
    def __init__(self, *, ticker_messages: list[object], fill_messages: list[object]):
        self.ticker_messages = ticker_messages
        self.fill_messages = fill_messages
        self.ticker_subscription: list[str] | None = None

    @asynccontextmanager
    async def connect(self):
        yield self

    async def subscribe_ticker(self, *, tickers: list[str]):
        self.ticker_subscription = tickers
        for message in self.ticker_messages:
            yield message

    async def subscribe_fill(self):
        for message in self.fill_messages:
            yield message


def adapter(
    client: FakeClient,
    websocket: FakeWebSocket | None = None,
) -> KalshiDemoAdapter:
    return KalshiDemoAdapter(
        client=client,
        websocket=websocket or FakeWebSocket(ticker_messages=[], fill_messages=[]),
    )


def test_demo_config_is_hardcoded() -> None:
    config = KalshiDemoAdapter.demo_config()

    assert config.base_url == "https://demo-api.kalshi.co/trade-api/v2"
    assert config.ws_base_url == "wss://demo-api.kalshi.co/trade-api/ws/v2"


@pytest.mark.asyncio
async def test_discovery_selects_current_market_and_ignores_future() -> None:
    now = datetime.now(UTC)
    future = SimpleNamespace(
        ticker="FUTURE",
        status="open",
        open_time=now + timedelta(minutes=5),
        close_time=now + timedelta(minutes=20),
    )
    current = SimpleNamespace(
        ticker="CURRENT",
        status="open",
        open_time=now - timedelta(minutes=5),
        close_time=now + timedelta(minutes=10),
    )
    markets = FakeMarkets([future, current])
    exchange = adapter(FakeClient(markets=markets))

    found = await exchange.discover_active_market("KXBTC15M", now=now)

    assert found.ticker == "CURRENT"
    assert markets.list_kwargs["series_ticker"] == "KXBTC15M"
    assert markets.list_kwargs["status"] == "open"


@pytest.mark.asyncio
async def test_discovery_fails_closed_when_no_market_is_current() -> None:
    now = datetime.now(UTC)
    markets = FakeMarkets(
        [
            SimpleNamespace(
                ticker="FUTURE",
                status="open",
                open_time=now + timedelta(minutes=5),
                close_time=now + timedelta(minutes=20),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="no active KXBTC15M market"):
        await adapter(FakeClient(markets=markets)).discover_active_market(
            "KXBTC15M",
            now=now,
        )


@pytest.mark.asyncio
async def test_current_quotes_use_yes_bid_and_ask() -> None:
    market = SimpleNamespace(
        yes_bid=Decimal("0.74"),
        yes_ask=Decimal("0.76"),
    )
    exchange = adapter(
        FakeClient(markets=FakeMarkets([], by_ticker={"MKT": market}))
    )

    quotes = await exchange.current_quotes("MKT")

    assert quotes.up_ask == Decimal("0.76")
    assert quotes.down_ask == Decimal("0.26")


@pytest.mark.asyncio
async def test_place_order_builds_v2_request() -> None:
    orders = FakeOrders()
    exchange = adapter(FakeClient(markets=FakeMarkets([]), orders=orders))

    result = await exchange.place_order(
        OrderRequest(
            ticker="MKT",
            client_order_id="client-1",
            book_side="ask",
            count=Decimal("0.35"),
            yes_price=Decimal("0.75"),
            reduce_only=True,
        )
    )

    sdk_request = orders.requests[0]
    assert sdk_request.time_in_force == "good_till_canceled"
    assert sdk_request.cancel_order_on_pause is True
    assert sdk_request.reduce_only is True
    assert result.order_id == "exchange-order-1"


@pytest.mark.asyncio
async def test_find_order_and_reconcile_fills() -> None:
    order = SimpleNamespace(order_id="order-1", client_order_id="client-1")
    fill = SimpleNamespace(
        fill_id="fill-1",
        order_id="order-1",
        ticker="MKT",
        count=Decimal("0.35"),
    )
    exchange = adapter(
        FakeClient(
            markets=FakeMarkets([]),
            orders=FakeOrders([order]),
            portfolio=FakePortfolio([fill]),
        )
    )

    found = await exchange.find_order_by_client_id("client-1")
    fills = await exchange.reconcile_fills("MKT")

    assert found is not None and found.order_id == "order-1"
    assert fills[0].fill_id == "fill-1"
    assert fills[0].quantity == Decimal("0.35")


@pytest.mark.asyncio
async def test_websocket_streams_parse_and_filter_market() -> None:
    ticker_messages = [
        SimpleNamespace(
            msg=SimpleNamespace(
                market_ticker="MKT",
                yes_bid=Decimal("0.74"),
                yes_ask=Decimal("0.76"),
            )
        )
    ]
    fill_messages = [
        SimpleNamespace(
            msg=SimpleNamespace(
                trade_id="trade-other",
                order_id="order-other",
                market_ticker="OTHER",
                count=Decimal("1"),
            )
        ),
        SimpleNamespace(
            msg=SimpleNamespace(
                trade_id="trade-1",
                order_id="order-1",
                market_ticker="MKT",
                count=Decimal("0.5"),
            )
        ),
    ]
    websocket = FakeWebSocket(
        ticker_messages=ticker_messages,
        fill_messages=fill_messages,
    )
    exchange = adapter(FakeClient(markets=FakeMarkets([])), websocket)

    quotes = [quote async for quote in exchange.quote_stream("MKT")]
    fills = [fill async for fill in exchange.fill_stream("MKT")]

    assert websocket.ticker_subscription == ["MKT"]
    assert quotes[0].down_ask == Decimal("0.26")
    assert [fill.fill_id for fill in fills] == ["trade-1"]
