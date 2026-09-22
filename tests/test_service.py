from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from kaishi_bot.config import BotConfig
from kaishi_bot.domain import Fill, Market, Side, SideQuotes
from kaishi_bot.service import BotService
from kaishi_bot.store import StateStore
from tests.fakes import FakeExchange


def active_market() -> Market:
    return Market(
        ticker="MKT",
        close_time=datetime.now(UTC) + timedelta(minutes=5),
    )


@pytest.mark.asyncio
async def test_check_is_read_only(tmp_path: Path) -> None:
    exchange = FakeExchange()
    exchange.market = active_market()
    exchange.quotes = [SideQuotes(Decimal("0.60"), Decimal("0.25"))]
    with StateStore(tmp_path / "state.sqlite3") as store:
        ticker, quotes = await BotService(exchange, store, BotConfig()).check()

    assert ticker == "MKT"
    assert quotes.down_ask == Decimal("0.25")
    assert exchange.requests == []


@pytest.mark.asyncio
async def test_quote_trigger_reaches_execution_once(tmp_path: Path) -> None:
    exchange = FakeExchange()
    exchange.market = active_market()
    exchange.quotes = [
        SideQuotes(Decimal("0.60"), Decimal("0.30")),
        SideQuotes(Decimal("0.60"), Decimal("0.25")),
        SideQuotes(Decimal("0.60"), Decimal("0.24")),
    ]
    with StateStore(tmp_path / "state.sqlite3") as store:
        await BotService(exchange, store, BotConfig()).run_market_once()

    assert len(exchange.requests) == 1
    assert exchange.requests[0].reduce_only is False


@pytest.mark.asyncio
async def test_startup_reconciliation_covers_existing_entry_fill(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    exchange.market = active_market()
    with StateStore(tmp_path / "state.sqlite3") as store:
        store.reserve_entry("MKT", Side.UP, "entry-client")
        store.record_entry_order("entry-client", "entry-order")
        exchange.reconciled_fills = [
            Fill("fill-1", "entry-order", "MKT", Decimal("0.5"))
        ]

        await BotService(exchange, store, BotConfig()).run_market_once()

    assert len(exchange.requests) == 1
    assert exchange.requests[0].reduce_only is True
    assert exchange.requests[0].count == Decimal("0.5")


@pytest.mark.asyncio
async def test_new_entry_is_immediately_reconciled_for_fast_fill(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    exchange.market = active_market()
    exchange.quotes = [SideQuotes(Decimal("0.25"), Decimal("0.80"))]

    original_place_order = exchange.place_order

    async def place_and_publish_fill(request):
        result = await original_place_order(request)
        if not request.reduce_only:
            exchange.reconciled_fills = [
                Fill("fill-fast", result.order_id, request.ticker, Decimal("1"))
            ]
        return result

    exchange.place_order = place_and_publish_fill

    with StateStore(tmp_path / "state.sqlite3") as store:
        await BotService(exchange, store, BotConfig()).run_market_once()

    assert [request.reduce_only for request in exchange.requests] == [False, True]
