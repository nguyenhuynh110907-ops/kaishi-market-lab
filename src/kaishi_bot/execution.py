from __future__ import annotations

from uuid import uuid4

from kaishi_bot.config import BotConfig
from kaishi_bot.domain import EntryIntent, Fill, OrderRequest, OrderResult
from kaishi_bot.exchange import ExchangePort
from kaishi_bot.pricing import to_v2_entry, to_v2_exit
from kaishi_bot.store import StateStore


class ExecutionManager:
    """Translate intents/fills into durable, idempotent exchange orders."""

    def __init__(
        self,
        exchange: ExchangePort,
        store: StateStore,
        config: BotConfig,
    ) -> None:
        self.exchange = exchange
        self.store = store
        self.config = config

    async def submit_entry(self, intent: EntryIntent) -> bool:
        client_order_id = str(uuid4())
        if not self.store.reserve_entry(
            intent.ticker,
            intent.side,
            client_order_id,
        ):
            return False

        book_side, yes_price = to_v2_entry(intent.side, intent.side_price)
        request = OrderRequest(
            ticker=intent.ticker,
            client_order_id=client_order_id,
            book_side=book_side,
            count=intent.count,
            yes_price=yes_price,
            reduce_only=False,
        )
        result = await self._place_or_reconcile(request)
        self.store.record_entry_order(client_order_id, result.order_id)
        return True

    async def handle_fill(self, fill: Fill) -> bool:
        side = self.store.entry_side_for_order(fill.order_id)
        if side is None:
            return False
        if not self.store.record_fill(
            fill.fill_id,
            fill.ticker,
            side,
            fill.quantity,
        ):
            return False

        uncovered = self.store.uncovered_quantity(fill.ticker, side)
        if uncovered <= 0:
            return False

        client_order_id = str(uuid4())
        book_side, yes_price = to_v2_exit(
            side,
            self.config.take_profit_price,
        )
        if not self.store.reserve_take_profit(
            fill.ticker,
            side,
            client_order_id,
            uncovered,
        ):
            return False
        request = OrderRequest(
            ticker=fill.ticker,
            client_order_id=client_order_id,
            book_side=book_side,
            count=uncovered,
            yes_price=yes_price,
            reduce_only=True,
        )
        result = await self._place_or_reconcile(request)
        self.store.record_take_profit_order(client_order_id, result.order_id)
        return True

    async def _place_or_reconcile(self, request: OrderRequest) -> OrderResult:
        try:
            return await self.exchange.place_order(request)
        except Exception:
            result = await self.exchange.find_order_by_client_id(
                request.client_order_id
            )
            if result is None:
                raise
            return result
