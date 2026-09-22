from decimal import Decimal
from pathlib import Path

import pytest

from kaishi_bot.config import BotConfig
from kaishi_bot.domain import EntryIntent, Fill, Side
from kaishi_bot.execution import ExecutionManager
from kaishi_bot.store import StateStore
from tests.fakes import FakeExchange


@pytest.mark.asyncio
async def test_duplicate_intent_places_one_entry(tmp_path: Path) -> None:
    exchange = FakeExchange()
    with StateStore(tmp_path / "state.sqlite3") as store:
        manager = ExecutionManager(exchange, store, BotConfig())
        intent = EntryIntent("MKT", Side.DOWN, Decimal("0.25"), Decimal("1"))

        assert await manager.submit_entry(intent) is True
        assert await manager.submit_entry(intent) is False

    assert len(exchange.requests) == 1
    assert exchange.requests[0].book_side == "ask"
    assert exchange.requests[0].yes_price == Decimal("0.75")
    assert exchange.requests[0].reduce_only is False


@pytest.mark.asyncio
async def test_ambiguous_entry_write_reconciles_client_id(tmp_path: Path) -> None:
    exchange = FakeExchange()
    exchange.fail_next_write = True
    with StateStore(tmp_path / "state.sqlite3") as store:
        manager = ExecutionManager(exchange, store, BotConfig())

        placed = await manager.submit_entry(
            EntryIntent("MKT", Side.UP, Decimal("0.25"), Decimal("1"))
        )

        assert placed is True
        assert store.entry_side_for_order("order-1") is Side.UP
    assert len(exchange.requests) == 1


@pytest.mark.asyncio
async def test_partial_fill_places_reduce_only_take_profit(tmp_path: Path) -> None:
    exchange = FakeExchange()
    with StateStore(tmp_path / "state.sqlite3") as store:
        store.reserve_entry("MKT", Side.DOWN, "entry-client")
        store.record_entry_order("entry-client", "entry-order")
        manager = ExecutionManager(exchange, store, BotConfig(take_profit_price="0.40"))

        assert await manager.handle_fill(
            Fill("fill-1", "entry-order", "MKT", Decimal("0.35"))
        ) is True

    request = exchange.requests[-1]
    assert request.reduce_only is True
    assert request.count == Decimal("0.35")
    assert request.book_side == "bid"
    assert request.yes_price == Decimal("0.60")


@pytest.mark.asyncio
async def test_duplicate_fill_does_not_duplicate_take_profit(tmp_path: Path) -> None:
    exchange = FakeExchange()
    fill = Fill("fill-1", "entry-order", "MKT", Decimal("0.35"))
    with StateStore(tmp_path / "state.sqlite3") as store:
        store.reserve_entry("MKT", Side.UP, "entry-client")
        store.record_entry_order("entry-client", "entry-order")
        manager = ExecutionManager(exchange, store, BotConfig())

        assert await manager.handle_fill(fill) is True
        assert await manager.handle_fill(fill) is False

    assert len(exchange.requests) == 1


@pytest.mark.asyncio
async def test_take_profit_fill_is_ignored(tmp_path: Path) -> None:
    exchange = FakeExchange()
    with StateStore(tmp_path / "state.sqlite3") as store:
        store.reserve_take_profit("MKT", Side.UP, "tp-client", Decimal("1"))
        store.record_take_profit_order("tp-client", "tp-order")
        manager = ExecutionManager(exchange, store, BotConfig())

        handled = await manager.handle_fill(
            Fill("fill-tp", "tp-order", "MKT", Decimal("1"))
        )

    assert handled is False
    assert exchange.requests == []


@pytest.mark.asyncio
async def test_ambiguous_take_profit_write_reconciles_client_id(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    exchange.fail_next_write = True
    with StateStore(tmp_path / "state.sqlite3") as store:
        store.reserve_entry("MKT", Side.UP, "entry-client")
        store.record_entry_order("entry-client", "entry-order")
        manager = ExecutionManager(exchange, store, BotConfig())

        handled = await manager.handle_fill(
            Fill("fill-1", "entry-order", "MKT", Decimal("0.5"))
        )

        assert handled is True
        assert store.summary()["take_profits"] == 1
    assert len(exchange.requests) == 1
