from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kaishi_bot.config import BotConfig
from kaishi_bot.domain import Market, Side, SideQuotes
from kaishi_bot.strategy import EntryStrategy


def market_closing_in(now: datetime, seconds: int) -> Market:
    return Market(ticker="KXBTC15M-TEST", close_time=now + timedelta(seconds=seconds))


def test_both_enabled_sides_trigger_at_or_below_threshold() -> None:
    now = datetime.now(UTC)
    strategy = EntryStrategy(BotConfig())
    quotes = SideQuotes(up_ask=Decimal("0.25"), down_ask=Decimal("0.24"))

    intents = strategy.evaluate(
        market_closing_in(now, 300),
        quotes,
        now,
        locked=frozenset(),
    )

    assert [intent.side for intent in intents] == [Side.UP, Side.DOWN]
    assert all(intent.count == Decimal("1") for intent in intents)


def test_price_above_threshold_does_not_trigger() -> None:
    now = datetime.now(UTC)
    strategy = EntryStrategy(BotConfig())

    intents = strategy.evaluate(
        market_closing_in(now, 300),
        SideQuotes(up_ask=Decimal("0.251"), down_ask=Decimal("0.30")),
        now,
        locked=frozenset(),
    )

    assert intents == []


def test_disabled_and_locked_sides_do_not_trigger() -> None:
    now = datetime.now(UTC)
    strategy = EntryStrategy(BotConfig(trade_down=False))

    intents = strategy.evaluate(
        market_closing_in(now, 300),
        SideQuotes(up_ask=Decimal("0.20"), down_ask=Decimal("0.20")),
        now,
        locked=frozenset({Side.UP}),
    )

    assert intents == []


def test_close_guard_blocks_all_entries_at_boundary() -> None:
    now = datetime.now(UTC)
    strategy = EntryStrategy(BotConfig(min_seconds_before_close=60))

    intents = strategy.evaluate(
        market_closing_in(now, 60),
        SideQuotes(up_ask=Decimal("0.20"), down_ask=Decimal("0.20")),
        now,
        locked=frozenset(),
    )

    assert intents == []
