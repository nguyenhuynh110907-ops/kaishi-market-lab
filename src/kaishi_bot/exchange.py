from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from kalshi import (
    CreateOrderV2Request,
    KalshiAuth,
    KalshiClient,
    KalshiConfig,
)
from kalshi.ws import KalshiWebSocket

from kaishi_bot.config import Credentials
from kaishi_bot.domain import (
    Fill,
    Market,
    OrderRequest,
    OrderResult,
    SideQuotes,
)
from kaishi_bot.pricing import side_quotes


class ExchangePort(Protocol):
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit one idempotent order."""

    async def find_order_by_client_id(
        self,
        client_order_id: str,
    ) -> OrderResult | None:
        """Reconcile an order after an ambiguous write response."""

    async def discover_active_market(self, series: str) -> Market:
        """Return the single currently tradable market for a series."""

    async def current_quotes(self, ticker: str) -> SideQuotes:
        """Return executable UP/DOWN asks from an authoritative snapshot."""

    def quote_stream(self, ticker: str) -> AsyncIterator[SideQuotes]:
        """Stream executable UP/DOWN asks."""

    def fill_stream(self, ticker: str) -> AsyncIterator[Fill]:
        """Stream account fills for one market."""

    async def reconcile_fills(self, ticker: str) -> list[Fill]:
        """Read authoritative fills after startup or reconnect."""


class KalshiDemoAdapter:
    """`kalshi-sdk` adapter with no production configuration path."""

    def __init__(self, *, client: Any, websocket: Any) -> None:
        self._client = client
        self._websocket_factory = lambda: websocket

    @staticmethod
    def demo_config() -> KalshiConfig:
        return KalshiConfig.demo()

    @classmethod
    def create(cls, credentials: Credentials) -> "KalshiDemoAdapter":
        config = cls.demo_config()
        client = KalshiClient(
            key_id=credentials.key_id,
            private_key_path=credentials.private_key_path,
            config=config,
        )
        auth = KalshiAuth.from_key_path(
            credentials.key_id,
            credentials.private_key_path,
        )
        instance = cls.__new__(cls)
        instance._client = client
        instance._websocket_factory = lambda: KalshiWebSocket(
            auth=auth,
            config=config,
        )
        return instance

    async def discover_active_market(
        self,
        series: str,
        *,
        now: datetime | None = None,
    ) -> Market:
        observed_at = now or datetime.now(UTC)

        def fetch() -> list[Any]:
            return list(
                self._client.markets.list(
                    status="open",
                    series_ticker=series,
                    limit=200,
                )
            )

        markets = await asyncio.to_thread(fetch)
        current = [
            market
            for market in markets
            if market.status == "open"
            and market.open_time <= observed_at < market.close_time
        ]
        if not current:
            raise RuntimeError(f"no active {series} market")
        if len(current) != 1:
            tickers = ", ".join(sorted(market.ticker for market in current))
            raise RuntimeError(f"ambiguous active {series} markets: {tickers}")
        market = current[0]
        return Market(ticker=market.ticker, close_time=market.close_time)

    async def current_quotes(self, ticker: str) -> SideQuotes:
        market = await asyncio.to_thread(self._client.markets.get, ticker)
        return side_quotes(
            yes_bid=Decimal(str(market.yes_bid)),
            yes_ask=Decimal(str(market.yes_ask)),
        )

    async def place_order(self, request: OrderRequest) -> OrderResult:
        sdk_request = CreateOrderV2Request(
            ticker=request.ticker,
            client_order_id=request.client_order_id,
            side=request.book_side,
            count=request.count,
            price=request.yes_price,
            time_in_force=request.time_in_force,
            self_trade_prevention_type="taker_at_cross",
            cancel_order_on_pause=True,
            reduce_only=request.reduce_only,
        )
        response = await asyncio.to_thread(
            self._client.orders.create_v2,
            request=sdk_request,
        )
        return OrderResult(
            order_id=response.order_id,
            client_order_id=response.client_order_id or request.client_order_id,
        )

    async def find_order_by_client_id(
        self,
        client_order_id: str,
    ) -> OrderResult | None:
        def find() -> OrderResult | None:
            for order in self._client.orders.list_all(max_pages=10):
                if order.client_order_id == client_order_id:
                    return OrderResult(
                        order_id=order.order_id,
                        client_order_id=client_order_id,
                    )
            return None

        return await asyncio.to_thread(find)

    async def reconcile_fills(self, ticker: str) -> list[Fill]:
        def fetch() -> list[Fill]:
            return [
                Fill(
                    fill_id=str(fill.fill_id),
                    order_id=str(fill.order_id),
                    ticker=str(fill.ticker),
                    quantity=Decimal(str(fill.count)),
                )
                for fill in self._client.portfolio.fills_all(
                    ticker=ticker,
                    max_pages=10,
                )
            ]

        return await asyncio.to_thread(fetch)

    async def quote_stream(self, ticker: str) -> AsyncIterator[SideQuotes]:
        websocket = self._websocket_factory()
        async with websocket.connect() as connected:
            async for message in connected.subscribe_ticker(tickers=[ticker]):
                payload = message.msg
                if payload.market_ticker != ticker:
                    continue
                yield side_quotes(
                    yes_bid=Decimal(str(payload.yes_bid)),
                    yes_ask=Decimal(str(payload.yes_ask)),
                )

    async def fill_stream(self, ticker: str) -> AsyncIterator[Fill]:
        websocket = self._websocket_factory()
        async with websocket.connect() as connected:
            async for message in connected.subscribe_fill():
                payload = message.msg
                if payload.market_ticker != ticker:
                    continue
                yield Fill(
                    fill_id=str(payload.trade_id),
                    order_id=str(payload.order_id),
                    ticker=payload.market_ticker,
                    quantity=Decimal(str(payload.count)),
                )

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)
