from __future__ import annotations

from collections.abc import AsyncIterator

from kaishi_bot.domain import (
    Fill,
    Market,
    OrderRequest,
    OrderResult,
    SideQuotes,
)


class FakeExchange:
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []
        self.orders_by_client_id: dict[str, OrderResult] = {}
        self.fail_next_write = False
        self.market: Market | None = None
        self.quotes: list[SideQuotes] = []
        self.fills: list[Fill] = []
        self.reconciled_fills: list[Fill] = []
        self.discover_calls = 0

    async def place_order(self, request: OrderRequest) -> OrderResult:
        self.requests.append(request)
        result = OrderResult(
            order_id=f"order-{len(self.requests)}",
            client_order_id=request.client_order_id,
        )
        self.orders_by_client_id[request.client_order_id] = result
        if self.fail_next_write:
            self.fail_next_write = False
            raise TimeoutError("ambiguous write")
        return result

    async def find_order_by_client_id(
        self,
        client_order_id: str,
    ) -> OrderResult | None:
        return self.orders_by_client_id.get(client_order_id)

    async def discover_active_market(self, series: str) -> Market:
        self.discover_calls += 1
        if self.market is None:
            raise RuntimeError(f"no active {series} market")
        return self.market

    async def current_quotes(self, ticker: str) -> SideQuotes:
        if not self.quotes:
            raise RuntimeError(f"no quotes for {ticker}")
        return self.quotes[-1]

    async def quote_stream(self, ticker: str) -> AsyncIterator[SideQuotes]:
        for quote in self.quotes:
            yield quote

    async def fill_stream(self, ticker: str) -> AsyncIterator[Fill]:
        for fill in self.fills:
            yield fill

    async def reconcile_fills(self, ticker: str) -> list[Fill]:
        return list(self.reconciled_fills)

    async def close(self) -> None:
        return None
