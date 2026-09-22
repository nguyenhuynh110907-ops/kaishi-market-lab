import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from kaishi_bot.research_capture import (
    ResearchCaptureSupervisor, parse_rti_message, parse_ticker_quote,
)
from kaishi_bot.research_config import ResearchCaptureConfig
from kaishi_bot.research_store import ResearchStore


def value_message(source_ms: int, seq: int = 1, value: str = "63310.36000000"):
    average = SimpleNamespace(
        value=Decimal(value), window_size=60,
        window_start_ts_ms=source_ms - 60_000,
        window_end_ts_exclusive=source_ms,
    )
    return SimpleNamespace(
        type="cfbenchmarks_value", seq=seq,
        msg=SimpleNamespace(
            index_id="BRTI", received_at=source_ms + 50,
            data=json.dumps({"id": "BRTI", "time": source_ms, "value": value}),
            avg_60s_data=average, last_60s_windowed_average_15min=None,
        ),
    )


def test_parse_rti_message_preserves_timestamps_decimal_latency_and_gap() -> None:
    source_ms = 1_786_476_600_000
    received = datetime.fromtimestamp((source_ms + 80) / 1000, tz=UTC)
    event = parse_rti_message(
        value_message(source_ms), asset="BTC", session_id="session",
        collector_received_at=received,
        previous_source_timestamp_ms=source_ms - 3_000,
        stale_after_seconds=3,
    )
    assert event.price == Decimal("63310.36000000")
    assert event.upstream_latency_ms == 50
    assert event.transport_latency_ms == 30
    assert event.collector_latency_ms == 80
    assert event.gap_detected is True
    assert event.missing_sample_count == 2
    assert event.is_stale is False


def test_status_exposes_current_rti_and_official_target(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    source_ms = 1_786_476_600_000
    received = datetime.fromtimestamp((source_ms + 80) / 1000, tz=UTC)
    parsed = parse_rti_message(
        value_message(source_ms), asset="BTC", session_id="session",
        collector_received_at=received, previous_source_timestamp_ms=None,
        stale_after_seconds=3,
    )
    from kaishi_bot.research_market_data import parse_market_payload
    market = parse_market_payload("BTC", "KXBTC15M", {
        "ticker": "KXBTC15M-TEST", "series_ticker": "KXBTC15M",
        "open_time": received.isoformat(),
        "close_time": datetime.fromtimestamp((source_ms + 900_080) / 1000, tz=UTC).isoformat(),
        "floor_strike": Decimal("63300.12000000"),
    }, received)
    store.save_market(market)
    supervisor = ResearchCaptureSupervisor(
        ResearchCaptureConfig(enabled=False, assets=("BTC",)),
        store, None, None, MemorySink(),
    )
    try:
        supervisor._latest_rti["BRTI"] = parsed
        supervisor._last_received_by_index["BRTI"] = datetime.now(UTC)
        item = supervisor.status()["indexes"][0]
        assert item["current_price"] == "63310.36000000"
        assert item["market"]["target_price"] == "63300.12000000"
        assert item["market"]["target_source_field"] == "floor_strike"
    finally:
        store.close()


class MemorySink:
    def __init__(self) -> None:
        self.events = []

    @staticmethod
    def available() -> bool:
        return True

    def write(self, events) -> int:
        self.events.extend(events)
        return len(events)


class MemoryCaptureSink:
    def __init__(self) -> None:
        self.quotes = []
        self.books = []
        self.checkpoints = []

    @staticmethod
    def available() -> bool:
        return True

    def write(self, *, quotes=(), books=(), checkpoints=()) -> int:
        self.quotes.extend(quotes)
        self.books.extend(books)
        self.checkpoints.extend(checkpoints)
        return len(quotes) + len(books) + len(checkpoints)


class FakeWebSocket:
    def __init__(self, messages) -> None:
        self.messages = messages

    def connect(self):
        websocket = self

        class Context:
            async def __aenter__(self):
                return websocket

            async def __aexit__(self, *args):
                return None

        return Context()

    async def subscribe_cfbenchmarks_value(self, **_):
        async def stream():
            for message in self.messages:
                yield message
        return stream()


class FakeBookWebSocket(FakeWebSocket):
    async def subscribe_ticker(self, *, tickers, **_):
        async def stream():
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            yield SimpleNamespace(
                type="ticker", seq=None,
                msg=SimpleNamespace(
                    market_ticker=tickers[0], yes_bid=Decimal("0.60"),
                    yes_ask=Decimal("0.65"), no_bid=Decimal("0.35"),
                    no_ask=Decimal("0.40"), ts_ms=now_ms, ts=now_ms // 1000,
                ),
            )
            await asyncio.sleep(60)
        return stream()

    async def subscribe_orderbook_delta(self, *, tickers, **_):
        async def stream():
            yield SimpleNamespace(
                type="orderbook_snapshot", seq=1,
                msg=SimpleNamespace(
                    market_ticker=tickers[0],
                    yes={Decimal("0.60"): Decimal("10")},
                    no={Decimal("0.35"): Decimal("8")},
                ),
            )
            await asyncio.sleep(60)
        return stream()


def test_quote_and_orderbook_toggles_control_capture_outputs(tmp_path) -> None:
    async def scenario(persist_quotes: bool, persist_books: bool):
        store = ResearchStore(tmp_path / f"{persist_quotes}-{persist_books}.sqlite3")
        now = datetime.now(UTC)
        from kaishi_bot.research_market_data import parse_market_payload
        store.save_market(parse_market_payload("BTC", "KXBTC15M", {
            "ticker": "BTC-1", "series_ticker": "KXBTC15M",
            "open_time": now.isoformat(),
            "close_time": datetime.fromtimestamp(now.timestamp() + 900, tz=UTC).isoformat(),
            "floor_strike": "63000",
        }, now))
        capture = MemoryCaptureSink()
        supervisor = ResearchCaptureSupervisor(
            ResearchCaptureConfig(
                enabled=True, root=tmp_path / "research", assets=("BTC",),
                persist_market_metadata=False, persist_rti=False,
                persist_contract_quotes=persist_quotes,
                persist_orderbook=persist_books, writer_batch_rows=1,
                flush_interval_seconds=0.05,
            ),
            store, None, lambda: FakeBookWebSocket([]), MemorySink(), capture,
        )
        try:
            await supervisor.start()
            for _ in range(50):
                if capture.quotes or capture.books:
                    break
                await asyncio.sleep(0.01)
            assert bool(capture.quotes) is persist_quotes
            assert bool(capture.books) is persist_books
        finally:
            await supervisor.close()
            store.close()

    asyncio.run(scenario(True, False))
    asyncio.run(scenario(False, True))
    asyncio.run(scenario(False, False))


def test_parse_ticker_quote_preserves_executable_prices() -> None:
    source_ms = 1_786_476_600_000
    message = SimpleNamespace(
        type="ticker", seq=None,
        msg=SimpleNamespace(
            market_ticker="BTC-1", yes_bid=Decimal("0.60"),
            yes_ask=Decimal("0.62"), no_bid=None, no_ask=None,
            ts_ms=source_ms, ts=source_ms // 1000,
        ),
    )
    quote = parse_ticker_quote(
        message, asset="BTC", series="KXBTC15M", session_id="session",
        collector_received_at=datetime.fromtimestamp(
            (source_ms + 80) / 1000, tz=UTC
        ),
        sequence=1, stale_after_seconds=3,
    )
    assert quote.up_bid == Decimal("0.60")
    assert quote.up_ask == Decimal("0.62")
    assert quote.down_bid == Decimal("0.38")
    assert quote.down_ask == Decimal("0.40")
    assert quote.event_kind == "ticker"
    assert quote.book_valid is True


@pytest.mark.asyncio
async def test_restart_checkpoint_dedupes_source_timestamp(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    source_ms = 1_786_476_600_000
    store.register_file_and_checkpoints(
        file_id="existing", dataset="rti_events", path="existing.parquet",
        sha256="hash", row_count=1,
        min_event_time=datetime.fromtimestamp(source_ms / 1000, tz=UTC),
        max_event_time=datetime.fromtimestamp(source_ms / 1000, tz=UTC),
        min_seq=1, max_seq=1, session_id="old",
        checkpoints=[("BRTI", source_ms, 1, "event")],
    )
    sink = MemorySink()
    config = ResearchCaptureConfig(
        enabled=True, root=tmp_path / "research", assets=("BTC",),
        persist_market_metadata=False, persist_contract_quotes=False,
        writer_batch_rows=1, flush_interval_seconds=0.05,
    )
    messages = [value_message(source_ms, 2), value_message(source_ms + 1000, 3)]
    supervisor = ResearchCaptureSupervisor(
        config, store, None, lambda: FakeWebSocket(messages), sink,
    )
    try:
        await supervisor.start()
        for _ in range(50):
            if sink.events:
                break
            await asyncio.sleep(0.01)
        assert [event.source_timestamp_ms for event in sink.events] == [source_ms + 1000]
    finally:
        await supervisor.close()
        store.close()


@pytest.mark.asyncio
async def test_disabled_capture_starts_no_tasks_and_needs_no_credentials(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    supervisor = ResearchCaptureSupervisor(
        ResearchCaptureConfig(enabled=False), store, None, None, MemorySink()
    )
    try:
        await supervisor.start()
        assert supervisor.status()["enabled"] is False
        assert supervisor.status()["rti_connected"] is False
    finally:
        await supervisor.close()
        store.close()


@pytest.mark.asyncio
async def test_capture_writer_flushes_on_wall_clock_during_continuous_stream(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    capture = MemoryCaptureSink()
    supervisor = ResearchCaptureSupervisor(
        ResearchCaptureConfig(
            enabled=True, root=tmp_path / "research", assets=("BTC",),
            persist_market_metadata=False, persist_rti=False,
            persist_contract_quotes=True, writer_batch_rows=10_000,
            flush_interval_seconds=0.03,
        ),
        store, None, None, MemorySink(), capture,
    )
    writer = asyncio.create_task(supervisor._capture_writer_loop())
    try:
        # Frequent events must not postpone the flush deadline indefinitely.
        for index in range(8):
            await supervisor.capture_queue.put((None, f"quote-{index}", None))
            await asyncio.sleep(0.01)
        for _ in range(20):
            if capture.quotes:
                break
            await asyncio.sleep(0.01)
        assert capture.quotes
    finally:
        await supervisor.capture_queue.put(None)
        await writer
        store.close()
