from __future__ import annotations

from datetime import datetime

from kaishi_bot.config import BotConfig
from kaishi_bot.domain import EntryIntent, Market, Side, SideQuotes


class EntryStrategy:
    """Pure threshold strategy with no exchange side effects."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def evaluate(
        self,
        market: Market,
        quotes: SideQuotes,
        now: datetime,
        *,
        locked: frozenset[Side],
    ) -> list[EntryIntent]:
        seconds_remaining = (market.close_time - now).total_seconds()
        if seconds_remaining <= self.config.min_seconds_before_close:
            return []

        candidates = (
            (Side.UP, self.config.trade_up, quotes.up_ask),
            (Side.DOWN, self.config.trade_down, quotes.down_ask),
        )
        return [
            EntryIntent(
                ticker=market.ticker,
                side=side,
                side_price=price,
                count=self.config.contracts,
            )
            for side, enabled, price in candidates
            if enabled and side not in locked and price <= self.config.entry_price
        ]
