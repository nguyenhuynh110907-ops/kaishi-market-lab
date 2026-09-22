from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from kaishi_bot.config import Credentials
from kaishi_bot.research_config import ResearchCaptureConfig
from kaishi_bot.research_book import ResearchOrderBookSet
from kaishi_bot.research_event_parquet import ParquetCaptureSink
from kaishi_bot.research_market_data import ResearchMarketClient
from kaishi_bot.research_models import BookCheckpoint, ContractQuoteEvent, OrderBookEvent, RtiEvent
from kaishi_bot.research_parquet import ParquetRtiSink
from kaishi_bot.research_store import ResearchStore


logger = logging.getLogger(__name__)


def _millis_time(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def _canonical_config_hash(config: ResearchCaptureConfig) -> str:
    raw = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _build_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True,
            text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_rti_message(
    message: Any, *, asset: str, session_id: str,
    collector_received_at: datetime, previous_source_timestamp_ms: int | None,
    stale_after_seconds: float,
) -> RtiEvent:
    payload = message.msg
    raw_data = str(payload.data)
    raw = json.loads(raw_data)
    source_ms = int(raw["time"])
    kalshi_ms = int(payload.received_at)
    source_time = _millis_time(source_ms)
    kalshi_received = _millis_time(kalshi_ms)
    collector_received = collector_received_at.astimezone(UTC)
    collector_ms = int(collector_received.timestamp() * 1000)
    delta = (
        source_ms - previous_source_timestamp_ms
        if previous_source_timestamp_ms is not None else 1000
    )
    missing = max(0, round(delta / 1000) - 1) if delta > 1500 else 0
    gap = missing > 0
    avg = getattr(payload, "avg_60s_data", None)
    final = getattr(payload, "last_60s_windowed_average_15min", None)
    event_hash = hashlib.sha256(
        f"{payload.index_id}\0{source_ms}\0{raw_data}".encode()
    ).hexdigest()
    return RtiEvent(
        asset=asset,
        index_id=str(payload.index_id),
        source_timestamp_ms=source_ms,
        source_time_utc=source_time,
        kalshi_received_at=kalshi_received,
        collector_received_at=collector_received,
        price=Decimal(str(raw["value"])),
        seq=getattr(message, "seq", None),
        session_id=session_id,
        upstream_latency_ms=kalshi_ms - source_ms,
        transport_latency_ms=collector_ms - kalshi_ms,
        collector_latency_ms=collector_ms - source_ms,
        avg_60s=Decimal(str(avg.value)) if avg is not None else None,
        final_15m_avg=Decimal(str(final.value)) if final is not None else None,
        avg_60s_window_size=int(avg.window_size) if avg is not None else None,
        avg_60s_window_start_ms=int(avg.window_start_ts_ms) if avg is not None else None,
        avg_60s_window_end_exclusive_ms=(
            int(avg.window_end_ts_exclusive) if avg is not None else None
        ),
        final_15m_window_size=int(final.window_size) if final is not None else None,
        final_15m_window_start_ms=(
            int(final.window_start_ts_ms) if final is not None else None
        ),
        final_15m_window_end_exclusive_ms=(
            int(final.window_end_ts_exclusive) if final is not None else None
        ),
        is_stale=(collector_ms - source_ms) > stale_after_seconds * 1000,
        gap_detected=gap,
        missing_sample_count=missing,
        raw_data_json=raw_data,
        event_sha256=event_hash,
    )


def parse_ticker_quote(
    message: Any, *, asset: str, series: str, session_id: str,
    collector_received_at: datetime, sequence: int,
    stale_after_seconds: float,
) -> ContractQuoteEvent:
    """Convert the latest-wins ticker channel into an executable quote."""
    payload = message.msg
    ticker = str(payload.market_ticker)
    up_bid = Decimal(str(payload.yes_bid))
    up_ask = Decimal(str(payload.yes_ask))
    raw_no_bid = getattr(payload, "no_bid", None)
    raw_no_ask = getattr(payload, "no_ask", None)
    down_bid = (
        Decimal(str(raw_no_bid)) if raw_no_bid is not None
        else Decimal("1") - up_ask
    )
    down_ask = (
        Decimal(str(raw_no_ask)) if raw_no_ask is not None
        else Decimal("1") - up_bid
    )
    timestamp_ms = int(getattr(payload, "ts_ms", 0) or 0)
    timestamp_s = int(getattr(payload, "ts", 0) or 0)
    source_timestamp = (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        if timestamp_ms > 0
        else datetime.fromtimestamp(timestamp_s, tz=UTC)
        if timestamp_s > 0 else None
    )
    collector_received_at = collector_received_at.astimezone(UTC)
    raw_payload = (
        message.model_dump(mode="json")
        if hasattr(message, "model_dump") else {
            "type": "ticker", "seq": getattr(message, "seq", None),
            "msg": vars(payload) if hasattr(payload, "__dict__") else str(payload),
        }
    )
    raw_json = json.dumps(
        raw_payload, sort_keys=True, separators=(",", ":"), default=str,
    )
    quote_payload = {
        "ticker": ticker, "source_timestamp": source_timestamp,
        "up_bid": str(up_bid), "up_ask": str(up_ask),
        "down_bid": str(down_bid), "down_ask": str(down_ask),
    }
    event_hash = hashlib.sha256(
        json.dumps(quote_payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    stable_id = hashlib.sha256(
        f"contract_quote_events\0{ticker}\0{session_id}\0{sequence}\0{event_hash}".encode()
    ).hexdigest()
    book_valid = (
        Decimal("0") <= up_bid <= up_ask <= Decimal("1")
        and Decimal("0") <= down_bid <= down_ask <= Decimal("1")
    )
    age = (
        (collector_received_at - source_timestamp).total_seconds()
        if source_timestamp is not None else 0.0
    )
    return ContractQuoteEvent(
        ticker=ticker, asset=asset, series_ticker=series,
        source_timestamp=source_timestamp,
        collector_received_at=collector_received_at,
        available_at=collector_received_at,
        up_bid=up_bid, up_ask=up_ask, down_bid=down_bid, down_ask=down_ask,
        up_spread=up_ask - up_bid, down_spread=down_ask - down_bid,
        book_sequence=sequence, collector_session_id=session_id,
        event_kind="ticker", is_stale=age > stale_after_seconds,
        gap_detected=False, book_valid=book_valid,
        price_convention="yes_price_v2", event_sha256=event_hash,
        stable_row_id=stable_id, raw_payload_json=raw_json,
    )


def production_websocket_factory(credentials: Credentials) -> Callable[[], Any]:
    from kalshi import KalshiAuth, KalshiConfig
    from kalshi.ws import KalshiWebSocket

    class ResearchKalshiWebSocket(KalshiWebSocket):
        async def _process_frame(self, raw: str) -> None:
            # Kalshi omits an empty side in some production snapshots. The
            # SDK intentionally rejects that shape, but research capture must
            # represent a genuinely empty side rather than enter a resync
            # storm. Keep this compatibility behavior local to this separate
            # research WebSocket; Live's client class is unchanged.
            data = self._json_loads(raw)
            if data.get("type") == "orderbook_snapshot":
                payload = data.setdefault("msg", {})
                payload.setdefault("market_id", "")
                if not any(
                    key in payload for key in ("yes", "yes_dollars", "yes_dollars_fp")
                ):
                    payload["yes"] = []
                if not any(
                    key in payload for key in ("no", "no_dollars", "no_dollars_fp")
                ):
                    payload["no"] = []
                raw = json.dumps(data, separators=(",", ":"))
            await super()._process_frame(raw)

    config = KalshiConfig.production()
    auth = KalshiAuth.from_key_path(credentials.key_id, credentials.private_key_path)
    return lambda: ResearchKalshiWebSocket(auth=auth, config=config)


class ResearchCaptureSupervisor:
    """Independent, fail-closed supervisor for Phase 0–1 research capture."""

    def __init__(
        self, config: ResearchCaptureConfig, store: ResearchStore,
        market_client: ResearchMarketClient | None,
        websocket_factory: Callable[[], Any] | None,
        sink: ParquetRtiSink | Any | None = None,
        capture_sink: ParquetCaptureSink | Any | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.market_client = market_client
        self.websocket_factory = websocket_factory
        self._config_hash = _canonical_config_hash(config)
        self.sink = sink or ParquetRtiSink(
            config.root, store, config.compression,
            config_sha256=self._config_hash,
        )
        self.capture_sink = capture_sink or ParquetCaptureSink(
            config.root, store, config.compression,
            config_sha256=_canonical_config_hash(config),
        )
        self.queue: asyncio.Queue[RtiEvent | None] = asyncio.Queue(config.queue_maxsize)
        self.capture_queue: asyncio.Queue[
            tuple[OrderBookEvent | None, ContractQuoteEvent | None, BookCheckpoint | None]
            | None
        ] = asyncio.Queue(config.queue_maxsize)
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop = asyncio.Event()
        self._writer_task: asyncio.Task[Any] | None = None
        self._capture_writer_task: asyncio.Task[Any] | None = None
        self._last_source_ms: dict[str, int] = {}
        self._registered_indexes: set[str] = set()
        self._latest_rti: dict[str, RtiEvent] = {}
        self._recent_rti: dict[str, deque[RtiEvent]] = {
            self.config.index_ids[asset]: deque(maxlen=180)
            for asset in self.config.assets
        }
        self._active_markets: dict[str, str] = {}
        self._connected = False
        self._quote_connected = False
        self._book_connected = False
        self._last_quote_received_at: datetime | None = None
        self._last_book_received_at: datetime | None = None
        self._last_book_sequence: int | None = None
        self._book_events_received = 0
        self._quotes_enqueued = 0
        self._book_gaps = 0
        self._connected_since: datetime | None = None
        self._last_received_at: datetime | None = None
        self._last_received_by_index: dict[str, datetime] = {}
        self._received_counts: dict[str, int] = {
            self.config.index_ids[asset]: 0 for asset in self.config.assets
        }
        self._gap_counts: dict[str, int] = {
            self.config.index_ids[asset]: 0 for asset in self.config.assets
        }
        self._last_missing_index_event: dict[str, datetime] = {}
        self._healthy_streak: dict[str, int] = {
            self.config.index_ids[asset]: 0 for asset in self.config.assets
        }
        self._last_fee_refresh: dict[str, datetime] = {}
        self._quote_quality_marked: set[str] = set()
        self._book_quality_marked: set[str] = set()
        self._build_commit = _build_commit()

    async def start(self) -> None:
        if not self.config.enabled or self._tasks or self._writer_task is not None:
            return
        self.config.root.mkdir(parents=True, exist_ok=True)
        for asset, market in self.store.latest_markets_by_asset(self.config.assets).items():
            self._active_markets[asset] = str(market["ticker"])
        if self.config.persist_rti:
            sink_unavailable = (
                not self.sink.available() if hasattr(self.sink, "available") else False
            )
            if sink_unavailable:
                self.store.quality_event(
                    stream="rti_events", reason_code="dependency_missing", severity="error",
                    details={"required_extra": "research"},
                )
            elif self.websocket_factory is None:
                self.store.quality_event(
                    stream="rti_events", reason_code="credentials_unavailable", severity="error"
                )
            else:
                self._load_checkpoints()
                self._writer_task = asyncio.create_task(self._writer_loop(), name="research-rti-writer")
                self._tasks.append(asyncio.create_task(self._rti_loop(), name="research-rti-reader"))
                self._tasks.append(asyncio.create_task(self._health_loop(), name="research-rti-health"))
        if self.config.persist_contract_quotes or self.config.persist_orderbook:
            sink_unavailable = (
                not self.capture_sink.available()
                if hasattr(self.capture_sink, "available") else False
            )
            if sink_unavailable:
                self.store.quality_event(
                    stream="contract_quote_events", reason_code="dependency_missing",
                    severity="error", details={"required_extra": "research"},
                )
            elif self.websocket_factory is None:
                self.store.quality_event(
                    stream="contract_quote_events", reason_code="credentials_unavailable",
                    severity="error",
                )
            else:
                self._capture_writer_task = asyncio.create_task(
                    self._capture_writer_loop(), name="research-book-writer"
                )
                if self.config.persist_contract_quotes:
                    self._tasks.append(asyncio.create_task(
                        self._quote_loop(), name="research-quote-reader"
                    ))
                if self.config.persist_orderbook:
                    self._tasks.append(asyncio.create_task(
                        self._book_loop(), name="research-book-reader"
                    ))
        if self.config.persist_market_metadata and self.market_client is not None:
            self._tasks.append(
                asyncio.create_task(self._metadata_loop(), name="research-market-metadata")
            )

    async def close(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._writer_task is not None:
            await self.queue.put(None)
            await self._writer_task
            self._writer_task = None
        if self._capture_writer_task is not None:
            await self.capture_queue.put(None)
            await self._capture_writer_task
            self._capture_writer_task = None
        if self.market_client is not None:
            await self.market_client.close()
        self._connected = False
        self._quote_connected = False
        self._book_connected = False

    async def _quote_loop(self) -> None:
        """Capture top-of-book without reconstructing the full delta stream."""
        assert self.websocket_factory is not None
        backoff = 1.0
        persisted_hashes: dict[str, str | None] = {}
        while not self._stop.is_set():
            tickers_by_asset = {
                asset: ticker for asset, ticker in self._active_markets.items()
                if asset in self.config.assets
            }
            if not tickers_by_asset:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.5)
                except TimeoutError:
                    continue
                return
            tickers = sorted(tickers_by_asset.values())
            asset_by_ticker = {ticker: asset for asset, ticker in tickers_by_asset.items()}
            for ticker in tickers:
                if ticker not in persisted_hashes:
                    checkpoint = self.store.checkpoint("contract_quote_events", ticker)
                    persisted_hashes[ticker] = (
                        str(checkpoint["last_event_hash"])
                        if checkpoint is not None else None
                    )
            session_id = str(uuid.uuid4())
            sequence = 0
            self.store.start_session(
                session_id, "contract_quote_events", self._config_hash,
                self._build_commit,
            )
            try:
                websocket = self.websocket_factory()
                async with websocket.connect() as session:
                    stream = await session.subscribe_ticker(
                        tickers=tickers, maxsize=self.config.queue_maxsize,
                    )
                    self._quote_connected = True
                    backoff = 1.0
                    iterator = stream.__aiter__()
                    next_message: asyncio.Task[Any] | None = None
                    while not self._stop.is_set():
                        current = sorted(
                            ticker for asset, ticker in self._active_markets.items()
                            if asset in self.config.assets
                        )
                        if current != tickers:
                            break
                        if next_message is None:
                            next_message = asyncio.create_task(anext(iterator))
                        done, _ = await asyncio.wait({next_message}, timeout=1.0)
                        if not done:
                            continue
                        try:
                            message = next_message.result()
                        except StopAsyncIteration:
                            break
                        finally:
                            next_message = None
                        ticker = str(message.msg.market_ticker)
                        asset = asset_by_ticker.get(ticker)
                        if asset is None:
                            continue
                        sequence += 1
                        received_at = datetime.now(UTC)
                        self._last_quote_received_at = received_at
                        quote = parse_ticker_quote(
                            message, asset=asset, series=self.config.series[asset],
                            session_id=session_id,
                            collector_received_at=received_at, sequence=sequence,
                            stale_after_seconds=self.config.stale_after_seconds,
                        )
                        if persisted_hashes.get(ticker) == quote.event_sha256:
                            continue
                        persisted_hashes[ticker] = quote.event_sha256
                        try:
                            self.capture_queue.put_nowait((None, quote, None))
                            self._quotes_enqueued += 1
                        except asyncio.QueueFull:
                            self.store.quality_event(
                                stream="contract_quote_events",
                                reason_code="queue_overflow", severity="error",
                                ticker=ticker, queue_depth=self.capture_queue.qsize(),
                                session_id=session_id,
                            )
                            break
                        if quote.book_valid and ticker not in self._quote_quality_marked:
                            self.store.mark_quality_dimension(
                                ticker, "contract_quotes_complete", True,
                                reason="quote_gap",
                            )
                            self._quote_quality_marked.add(ticker)
                    if next_message is not None:
                        next_message.cancel()
                        await asyncio.gather(next_message, return_exceptions=True)
                    self.store.finish_session(
                        session_id, "closed", last_seq=sequence,
                    )
            except asyncio.CancelledError:
                self.store.finish_session(
                    session_id, "stopped", last_seq=sequence,
                )
                raise
            except Exception as error:
                logger.warning("research ticker stream failed: %s", error)
                self.store.finish_session(
                    session_id, "failed", last_seq=sequence,
                    error_code=type(error).__name__,
                )
                self.store.quality_event(
                    stream="contract_quote_events",
                    reason_code="stream_disconnected", severity="error",
                    session_id=session_id,
                    details={"error": type(error).__name__},
                )
            finally:
                self._quote_connected = False
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except TimeoutError:
                backoff = min(30.0, backoff * 2)

    async def _book_loop(self) -> None:
        assert self.websocket_factory is not None
        backoff = 1.0
        persisted_book_hashes: dict[str, str | None] = {}
        persisted_quote_hashes: dict[str, str | None] = {}
        while not self._stop.is_set():
            tickers_by_asset = {
                asset: ticker for asset, ticker in self._active_markets.items()
                if asset in self.config.assets
            }
            if not tickers_by_asset:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.5)
                except TimeoutError:
                    continue
                return
            tickers = sorted(tickers_by_asset.values())
            asset_by_ticker = {ticker: asset for asset, ticker in tickers_by_asset.items()}
            for ticker in tickers:
                if ticker not in persisted_book_hashes:
                    book_checkpoint = self.store.checkpoint("orderbook_events", ticker)
                    quote_checkpoint = self.store.checkpoint(
                        "contract_quote_events", ticker
                    )
                    persisted_book_hashes[ticker] = (
                        str(book_checkpoint["last_event_hash"])
                        if book_checkpoint is not None else None
                    )
                    persisted_quote_hashes[ticker] = (
                        str(quote_checkpoint["last_event_hash"])
                        if quote_checkpoint is not None else None
                    )
            session_id = str(uuid.uuid4())
            last_sequence: int | None = None
            self.store.start_session(
                session_id, "orderbook_events", self._config_hash, self._build_commit
            )
            try:
                websocket = self.websocket_factory()
                async with websocket.connect() as session:
                    stream = await session.subscribe_orderbook_delta(
                        tickers=tickers, maxsize=self.config.queue_maxsize
                    )
                    state = ResearchOrderBookSet(
                        assets_by_ticker=asset_by_ticker,
                        series_by_asset=self.config.series,
                        session_id=session_id, depth=self.config.orderbook_depth,
                    )
                    iterator = stream.__aiter__()
                    self._book_connected = True
                    backoff = 1.0
                    last_checkpoints: dict[str, datetime] = {}
                    next_message: asyncio.Task[Any] | None = None
                    while not self._stop.is_set():
                        current = sorted(
                            ticker for asset, ticker in self._active_markets.items()
                            if asset in self.config.assets
                        )
                        if current != tickers:
                            if next_message is not None:
                                next_message.cancel()
                                await asyncio.gather(next_message, return_exceptions=True)
                            break
                        if next_message is None:
                            next_message = asyncio.create_task(anext(iterator))
                        done, _ = await asyncio.wait({next_message}, timeout=1.0)
                        if not done:
                            now = datetime.now(UTC)
                            if self.config.persist_orderbook:
                                for ticker in tickers:
                                    previous = last_checkpoints.get(ticker)
                                    if (
                                        previous is None
                                        or (now - previous).total_seconds() >= 30
                                    ):
                                        checkpoint = state.checkpoint(ticker, now)
                                        if checkpoint.is_complete:
                                            await self.capture_queue.put(
                                                (None, None, checkpoint)
                                            )
                                            last_checkpoints[ticker] = now
                            continue
                        try:
                            message = next_message.result()
                        except StopAsyncIteration:
                            break
                        finally:
                            next_message = None
                        received_at = datetime.now(UTC)
                        result = state.apply(message, received_at)
                        last_sequence = result.event.sequence
                        self._last_book_received_at = received_at
                        self._last_book_sequence = last_sequence
                        self._book_events_received += 1
                        book_event = result.event if self.config.persist_orderbook else None
                        # Quote-only capture uses the independent ticker feed.
                        # This stateful stream is retained only for full books.
                        quote = None
                        checkpoint = result.checkpoint if self.config.persist_orderbook else None
                        if book_event is not None:
                            if (
                                persisted_book_hashes.get(book_event.ticker)
                                == book_event.event_sha256
                            ):
                                book_event = None
                            else:
                                persisted_book_hashes[book_event.ticker] = (
                                    book_event.event_sha256
                                )
                        if quote is not None:
                            if (
                                persisted_quote_hashes.get(quote.ticker)
                                == quote.event_sha256
                            ):
                                quote = None
                            else:
                                persisted_quote_hashes[quote.ticker] = quote.event_sha256
                        if book_event is not None or quote is not None or checkpoint is not None:
                            try:
                                self.capture_queue.put_nowait((book_event, quote, checkpoint))
                                if quote is not None:
                                    self._quotes_enqueued += 1
                            except asyncio.QueueFull:
                                self.store.quality_event(
                                    stream="orderbook_events", reason_code="queue_overflow",
                                    severity="error", ticker=result.event.ticker,
                                    queue_depth=self.capture_queue.qsize(), session_id=session_id,
                                )
                                break
                        if (
                            quote is not None and quote.book_valid
                            and quote.ticker not in self._quote_quality_marked
                        ):
                            self.store.mark_quality_dimension(
                                quote.ticker, "contract_quotes_complete", True,
                                reason="quote_gap",
                            )
                            self._quote_quality_marked.add(quote.ticker)
                        if (
                            checkpoint is not None and checkpoint.is_complete
                            and checkpoint.ticker not in self._book_quality_marked
                        ):
                            self.store.mark_quality_dimension(
                                checkpoint.ticker, "orderbook_complete", True,
                                reason="book_sequence_gap",
                            )
                            self._book_quality_marked.add(checkpoint.ticker)
                        if result.sequence_gap:
                            self._book_gaps += 1
                            self._quote_quality_marked.discard(result.event.ticker)
                            self._book_quality_marked.discard(result.event.ticker)
                            if self.config.persist_contract_quotes:
                                self.store.mark_quality_dimension(
                                    result.event.ticker,
                                    "contract_quotes_complete", False,
                                    reason="quote_gap",
                                )
                            if self.config.persist_orderbook:
                                self.store.mark_quality_dimension(
                                    result.event.ticker, "orderbook_complete", False,
                                    reason="book_sequence_gap",
                                )
                            self.store.quality_event(
                                stream="orderbook_events", reason_code="book_sequence_gap",
                                severity="error", ticker=result.event.ticker,
                                expected_seq=result.expected_sequence,
                                observed_seq=last_sequence,
                                session_id=session_id,
                            )
                            # Reconnect instead of trusting any post-gap delta.
                            # A new subscription must begin with fresh snapshots.
                            break
                    if next_message is not None:
                        next_message.cancel()
                        await asyncio.gather(next_message, return_exceptions=True)
                self.store.finish_session(session_id, "closed", last_seq=last_sequence)
            except asyncio.CancelledError:
                self.store.finish_session(session_id, "stopped", last_seq=last_sequence)
                raise
            except Exception as error:
                logger.warning("research order-book stream failed: %s", error)
                self.store.finish_session(
                    session_id, "failed", last_seq=last_sequence,
                    error_code=type(error).__name__,
                )
                self.store.quality_event(
                    stream="orderbook_events", reason_code="stream_disconnected",
                    severity="error", session_id=session_id,
                    details={"error": type(error).__name__},
                )
            finally:
                self._book_connected = False
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except TimeoutError:
                backoff = min(30.0, backoff * 2)

    async def _capture_writer_loop(self) -> None:
        books: list[OrderBookEvent] = []
        quotes: list[ContractQuoteEvent] = []
        checkpoints: list[BookCheckpoint] = []
        loop = asyncio.get_running_loop()
        flush_deadline = loop.time() + self.config.flush_interval_seconds
        while True:
            try:
                item = await asyncio.wait_for(
                    self.capture_queue.get(),
                    timeout=max(0.001, flush_deadline - loop.time()),
                )
            except TimeoutError:
                item = "flush"
            if isinstance(item, tuple):
                book, quote, checkpoint = item
                if book is not None:
                    books.append(book)
                if quote is not None:
                    quotes.append(quote)
                if checkpoint is not None:
                    checkpoints.append(checkpoint)
            size = len(books) + len(quotes) + len(checkpoints)
            if size and (
                size >= self.config.writer_batch_rows or item == "flush" or item is None
            ):
                try:
                    await asyncio.to_thread(
                        self.capture_sink.write, quotes=quotes, books=books,
                        checkpoints=checkpoints,
                    )
                except Exception as error:
                    logger.exception("research book writer failed")
                    self.store.quality_event(
                        stream="orderbook_events", reason_code="manifest_error",
                        severity="error", details={"error": type(error).__name__},
                    )
                books, quotes, checkpoints = [], [], []
                flush_deadline = loop.time() + self.config.flush_interval_seconds
            elif item == "flush":
                flush_deadline = loop.time() + self.config.flush_interval_seconds
            if item is None:
                return

    def _load_checkpoints(self) -> None:
        for asset in self.config.assets:
            index_id = self.config.index_ids[asset]
            row = self.store.checkpoint("rti_events", index_id)
            if row is not None and row["last_source_timestamp_ms"] is not None:
                self._last_source_ms[index_id] = int(row["last_source_timestamp_ms"])

    async def _metadata_loop(self) -> None:
        assert self.market_client is not None
        while not self._stop.is_set():
            for asset in self.config.assets:
                series = self.config.series[asset]
                try:
                    market = await self.market_client.active(asset, series)
                    if market is None:
                        self.store.quality_event(
                            stream="market_metadata", reason_code="active_market_missing",
                            severity="warning", details={"asset": asset, "series": series},
                        )
                        continue
                    previous = self._active_markets.get(asset)
                    self.store.save_market(market)
                    self.store.refresh_market_core_quality(market)
                    self._active_markets[asset] = market.ticker
                    fee_refreshed = self._last_fee_refresh.get(series)
                    if (
                        fee_refreshed is None
                        or (datetime.now(UTC) - fee_refreshed).total_seconds() >= 300
                    ):
                        for version in await self.market_client.fee_versions(series):
                            self.store.save_fee_version(version)
                            if (
                                version.effective_from <= market.open_time
                                and (version.effective_to is None
                                     or version.effective_to > market.open_time)
                            ):
                                self.store.mark_quality_dimension(
                                    market.ticker, "fee_metadata_complete", True,
                                    reason="missing_fee_metadata",
                                )
                        self._last_fee_refresh[series] = datetime.now(UTC)
                    if previous and previous != market.ticker:
                        settled = await self.market_client.detail(asset, series, previous)
                        self.store.save_market(settled)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("research metadata refresh failed for %s: %s", asset, error)
                    self.store.quality_event(
                        stream="market_metadata", reason_code="refresh_failed",
                        severity="error", details={"asset": asset, "error": type(error).__name__},
                    )
            for asset, ticker in self.store.legacy_markets_missing_metadata(limit=5):
                series = self.config.series.get(asset)
                if series is None:
                    self.store.record_backfill_attempt(ticker, "unsupported_asset")
                    continue
                try:
                    historical = await self.market_client.detail(asset, series, ticker)
                    self.store.save_market(historical)
                    self.store.record_backfill_attempt(ticker, "complete")
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self.store.record_backfill_attempt(
                        ticker, "failed", type(error).__name__
                    )
                    self.store.quality_event(
                        stream="market_metadata", reason_code="backfill_failed",
                        severity="warning", ticker=ticker,
                        details={"asset": asset, "error": type(error).__name__},
                    )
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.metadata_refresh_seconds
                )
            except TimeoutError:
                pass

    async def _rti_loop(self) -> None:
        assert self.websocket_factory is not None
        backoff = 1.0
        while not self._stop.is_set():
            session_id = str(uuid.uuid4())
            last_seq: int | None = None
            self.store.start_session(
                session_id, "rti_events", self._config_hash, self._build_commit
            )
            try:
                websocket = self.websocket_factory()
                async with websocket.connect() as session:
                    stream = await session.subscribe_cfbenchmarks_value(
                        index_ids=[self.config.index_ids[a] for a in self.config.assets],
                        maxsize=self.config.queue_maxsize,
                    )
                    self._connected = True
                    self._connected_since = datetime.now(UTC)
                    backoff = 1.0
                    async for message in stream:
                        if self._stop.is_set():
                            break
                        seq = getattr(message, "seq", None)
                        previous_seq = last_seq
                        sequence_gap = (
                            previous_seq is not None and seq is not None
                            and seq != previous_seq + 1
                        )
                        last_seq = seq if seq is not None else last_seq
                        if str(getattr(message, "type", "")) != "cfbenchmarks_value":
                            continue
                        received_at = datetime.now(UTC)
                        index_id = str(message.msg.index_id)
                        asset = next(
                            (a for a in self.config.assets if self.config.index_ids[a] == index_id),
                            None,
                        )
                        if asset is None:
                            self.store.quality_event(
                                stream="rti_events", reason_code="unknown_index",
                                severity="error", index_id=index_id, session_id=session_id,
                            )
                            continue
                        if index_id not in self._registered_indexes:
                            self.store.register_index(asset, index_id, received_at)
                            self._registered_indexes.add(index_id)
                        event = parse_rti_message(
                            message, asset=asset, session_id=session_id,
                            collector_received_at=received_at,
                            previous_source_timestamp_ms=self._last_source_ms.get(index_id),
                            stale_after_seconds=self.config.stale_after_seconds,
                        )
                        previous = self._last_source_ms.get(index_id)
                        if previous is not None and event.source_timestamp_ms <= previous:
                            continue
                        self._last_source_ms[index_id] = event.source_timestamp_ms
                        self._latest_rti[index_id] = event
                        self._recent_rti[index_id].append(event)
                        self._last_received_at = received_at
                        self._last_received_by_index[index_id] = received_at
                        self._received_counts[index_id] += 1
                        if event.is_stale or event.gap_detected or sequence_gap:
                            self._healthy_streak[index_id] = 0
                        else:
                            self._healthy_streak[index_id] += 1
                            ticker = self._active_markets.get(asset)
                            if ticker and self._healthy_streak[index_id] == 2:
                                self.store.recover_market_quality(
                                    ticker,
                                    {"index_not_updating", "stale_source"},
                                    received_at,
                                )
                        if sequence_gap:
                            event = event.model_copy(update={"gap_detected": True})
                            self.store.quality_event(
                                stream="rti_events", reason_code="sequence_gap",
                                severity="error", index_id=index_id,
                                ticker=self._active_markets.get(asset),
                                expected_seq=(previous_seq + 1) if previous_seq is not None else None,
                                observed_seq=seq, session_id=session_id,
                            )
                        if event.gap_detected:
                            self._gap_counts[index_id] += 1
                            self.store.quality_event(
                                stream="rti_events", reason_code="rti_gap",
                                severity="error", index_id=index_id,
                                ticker=self._active_markets.get(asset),
                                event_time=event.source_time_utc,
                                missing_sample_count=event.missing_sample_count,
                                session_id=session_id,
                            )
                        if event.is_stale:
                            self.store.quality_event(
                                stream="rti_events", reason_code="stale_source",
                                severity="error", index_id=index_id,
                                ticker=self._active_markets.get(asset),
                                event_time=event.source_time_utc,
                                stale_seconds=str(Decimal(event.collector_latency_ms) / 1000),
                                session_id=session_id,
                            )
                        try:
                            self.queue.put_nowait(event)
                        except asyncio.QueueFull:
                            self.store.quality_event(
                                stream="rti_events", reason_code="queue_overflow",
                                severity="error", index_id=index_id,
                                ticker=self._active_markets.get(asset),
                                queue_depth=self.queue.qsize(), session_id=session_id,
                            )
                self.store.finish_session(session_id, "closed", last_seq=last_seq)
            except asyncio.CancelledError:
                self.store.finish_session(session_id, "stopped", last_seq=last_seq)
                raise
            except Exception as error:
                logger.warning("research RTI stream failed: %s", error)
                self.store.finish_session(
                    session_id, "failed", last_seq=last_seq,
                    error_code=type(error).__name__,
                )
                self.store.quality_event(
                    stream="rti_events", reason_code="stream_disconnected",
                    severity="error", session_id=session_id,
                    details={"error": type(error).__name__},
                )
            finally:
                self._connected = False
                self._connected_since = None
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except TimeoutError:
                backoff = min(30.0, backoff * 2)

    async def _health_loop(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(UTC)
            if self._connected and self._connected_since is not None:
                for asset in self.config.assets:
                    index_id = self.config.index_ids[asset]
                    last = self._last_received_by_index.get(index_id, self._connected_since)
                    age = (now - last).total_seconds()
                    never_received = index_id not in self._last_received_by_index
                    grace = max(10.0, self.config.stale_after_seconds * 3)
                    last_alert = self._last_missing_index_event.get(index_id)
                    if (
                        age > (grace if never_received else self.config.stale_after_seconds)
                        and (last_alert is None or (now - last_alert).total_seconds() >= 30)
                    ):
                        self._healthy_streak[index_id] = 0
                        self.store.quality_event(
                            stream="rti_events", reason_code="index_not_updating",
                            severity="error", index_id=index_id,
                            ticker=self._active_markets.get(asset),
                            stale_seconds=str(Decimal(str(age))),
                            details={"asset": asset},
                        )
                        self._last_missing_index_event[index_id] = now
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except TimeoutError:
                pass

    async def _writer_loop(self) -> None:
        batch: list[RtiEvent] = []
        loop = asyncio.get_running_loop()
        flush_deadline = loop.time() + self.config.flush_interval_seconds
        while True:
            try:
                item = await asyncio.wait_for(
                    self.queue.get(), timeout=max(0.001, flush_deadline - loop.time())
                )
            except TimeoutError:
                item = None if self._stop.is_set() else "flush"
            if isinstance(item, RtiEvent):
                batch.append(item)
            if batch and (
                len(batch) >= self.config.writer_batch_rows
                or item is None or item == "flush"
            ):
                try:
                    await asyncio.to_thread(self.sink.write, batch)
                    batch = []
                except Exception as error:
                    logger.exception("research RTI writer failed")
                    self.store.quality_event(
                        stream="rti_events", reason_code="manifest_error",
                        severity="error", details={"error": type(error).__name__},
                    )
                    batch = []
                flush_deadline = loop.time() + self.config.flush_interval_seconds
            elif item == "flush":
                flush_deadline = loop.time() + self.config.flush_interval_seconds
            if item is None:
                return

    def status(self) -> dict[str, object]:
        now = datetime.now(UTC)
        markets = self.store.latest_markets_by_asset(self.config.assets)
        connected_seconds = (
            max(0, int((now - self._connected_since).total_seconds()))
            if self._connected_since else 0
        )
        indexes = []
        for asset in self.config.assets:
            index_id = self.config.index_ids[asset]
            received_at = self._last_received_by_index.get(index_id)
            latest = self._latest_rti.get(index_id)
            market = markets.get(asset)
            age_ms = (
                max(0, int((now - received_at).total_seconds() * 1000))
                if received_at else None
            )
            indexes.append({
                "asset": asset,
                "index_id": index_id,
                "last_source_timestamp_ms": self._last_source_ms.get(index_id),
                "last_received_at": received_at.isoformat() if received_at else None,
                "age_ms": age_ms,
                "is_stale": age_ms is None or age_ms > self.config.stale_after_seconds * 1000,
                "samples_expected_this_connection": connected_seconds,
                "samples_received_this_process": self._received_counts[index_id],
                "gaps_this_process": self._gap_counts[index_id],
                "current_price": str(latest.price) if latest else None,
                "source_time": latest.source_time_utc.isoformat() if latest else None,
                "collector_latency_ms": latest.collector_latency_ms if latest else None,
                "market": market,
            })
        return {
            "enabled": self.config.enabled,
            "rti_connected": self._connected,
            "quote_connected": self._quote_connected,
            "orderbook_connected": self._book_connected,
            "last_received_at": (
                self._last_received_at.isoformat() if self._last_received_at else None
            ),
            "queue_depth": self.queue.qsize(),
            "book_queue_depth": self.capture_queue.qsize(),
            "orderbook": {
                "quote_last_received_at": (
                    self._last_quote_received_at.isoformat()
                    if self._last_quote_received_at else None
                ),
                "last_received_at": (
                    self._last_book_received_at.isoformat()
                    if self._last_book_received_at else None
                ),
                "last_sequence": self._last_book_sequence,
                "events_received_this_process": self._book_events_received,
                "quotes_enqueued_this_process": self._quotes_enqueued,
                "gaps_this_process": self._book_gaps,
            },
            "queue_maxsize": self.config.queue_maxsize,
            "root": str(self.config.root),
            "indexes": indexes,
            **self.store.status(),
        }

    def recent_rti(self, asset: str) -> tuple[RtiEvent, ...]:
        """Return an immutable recent RTI view for Paper-shadow inference."""
        index_id = self.config.index_ids.get(asset)
        if index_id is None:
            return ()
        return tuple(self._recent_rti.get(index_id, ()))
