from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from kaishi_bot.config import BotConfig
from kaishi_bot.execution import ExecutionManager
from kaishi_bot.exchange import ExchangePort
from kaishi_bot.store import StateStore
from kaishi_bot.strategy import EntryStrategy

logger = logging.getLogger(__name__)


class BotService:
    """Coordinate discovery, reconciliation, streams, strategy, and execution."""

    def __init__(
        self,
        exchange: ExchangePort,
        store: StateStore,
        config: BotConfig,
    ) -> None:
        self.exchange = exchange
        self.store = store
        self.config = config
        self.strategy = EntryStrategy(config)
        self.execution = ExecutionManager(exchange, store, config)

    async def check(self):
        market = await self.exchange.discover_active_market(self.config.series)
        quotes = await self.exchange.current_quotes(market.ticker)
        return market.ticker, quotes

    async def reconcile(self, ticker: str) -> None:
        fills = await self.exchange.reconcile_fills(ticker)
        for fill in fills:
            await self.execution.handle_fill(fill)

    async def run_market_once(self) -> None:
        market = await self.exchange.discover_active_market(self.config.series)
        logger.info("active market %s", market.ticker)
        await self.reconcile(market.ticker)

        async def consume_quotes() -> None:
            async for quotes in self.exchange.quote_stream(market.ticker):
                locked = self.store.locked_sides(market.ticker)
                intents = self.strategy.evaluate(
                    market,
                    quotes,
                    datetime.now(UTC),
                    locked=locked,
                )
                for intent in intents:
                    placed = await self.execution.submit_entry(intent)
                    if placed:
                        logger.info(
                            "entry submitted ticker=%s side=%s side_price=%s count=%s",
                            intent.ticker,
                            intent.side.value,
                            intent.side_price,
                            intent.count,
                        )
                        await self.reconcile(market.ticker)

        async def consume_fills() -> None:
            async for fill in self.exchange.fill_stream(market.ticker):
                if await self.execution.handle_fill(fill):
                    logger.info(
                        "take profit submitted ticker=%s fill_id=%s quantity=%s",
                        fill.ticker,
                        fill.fill_id,
                        fill.quantity,
                    )

        seconds_to_rollover = max(
            1.0,
            (market.close_time - datetime.now(UTC)).total_seconds() + 2.0,
        )
        try:
            async with asyncio.timeout(seconds_to_rollover):
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(consume_quotes())
                    tasks.create_task(consume_fills())
        except TimeoutError:
            logger.info("market window closed ticker=%s", market.ticker)

    async def run(self) -> None:
        delay = 1
        while True:
            try:
                await self.run_market_once()
                delay = 1
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("cycle paused after error: %s", error)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
