from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping

from kaishi_bot.research_models import (
    BookCheckpoint,
    BookLevel,
    ContractQuoteEvent,
    OrderBookEvent,
)


ONE = Decimal("1")


def _json_safe(payload: object) -> object:
    if isinstance(payload, Mapping):
        return {
            str(key): _json_safe(value)
            for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
        }
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, (Decimal, datetime)):
        return str(payload)
    if hasattr(payload, "model_dump"):
        return _json_safe(payload.model_dump(mode="json"))
    return payload


def _canonical(payload: object) -> str:
    return json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), default=str,
        ensure_ascii=False,
    )


def _source_time(payload: Any) -> datetime | None:
    milliseconds = getattr(payload, "ts_ms", None)
    if milliseconds:
        return datetime.fromtimestamp(int(milliseconds) / 1000, tz=UTC)
    value = getattr(payload, "ts", None)
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC)
    return None


def _levels(value: Any) -> dict[Decimal, Decimal]:
    if value is None:
        return {}
    pairs = value.items() if isinstance(value, Mapping) else value
    result: dict[Decimal, Decimal] = {}
    for price, quantity in pairs:
        parsed_quantity = Decimal(str(quantity))
        if parsed_quantity > 0:
            result[Decimal(str(price))] = parsed_quantity
    return result


def _snapshot_side(payload: Any, side: str) -> dict[Decimal, Decimal]:
    for name in (side, f"{side}_dollars_fp"):
        value = getattr(payload, name, None)
        if value is not None:
            return _levels(value)
    return {}


@dataclass(slots=True)
class _Book:
    yes: dict[Decimal, Decimal] = field(default_factory=dict)
    no: dict[Decimal, Decimal] = field(default_factory=dict)
    valid: bool = False
    last_sequence: int | None = None
    last_top: tuple[Decimal | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class BookApplyResult:
    event: OrderBookEvent
    quote: ContractQuoteEvent | None
    checkpoint: BookCheckpoint | None
    sequence_gap: bool
    expected_sequence: int | None
    tickers_needing_snapshot: tuple[str, ...]


class ResearchOrderBookSet:
    """Deterministic subscription-level snapshot/delta state machine.

    Sequence numbers belong to the subscription. A gap therefore invalidates
    every maintained ticker until each receives a fresh snapshot.
    """

    def __init__(
        self, *, assets_by_ticker: Mapping[str, str], series_by_asset: Mapping[str, str],
        session_id: str, depth: int = 10,
    ) -> None:
        if depth < 10:
            raise ValueError("research order-book depth must be at least 10")
        self.assets_by_ticker = dict(assets_by_ticker)
        self.series_by_asset = dict(series_by_asset)
        self.session_id = session_id
        self.depth = depth
        self.books: dict[str, _Book] = {
            ticker: _Book() for ticker in self.assets_by_ticker
        }
        self.last_sequence: int | None = None

    def apply(self, message: Any, received_at: datetime) -> BookApplyResult:
        received_at = received_at.astimezone(UTC)
        payload = message.msg
        ticker = str(payload.market_ticker)
        if ticker not in self.books:
            raise ValueError(f"unsubscribed research ticker: {ticker}")
        message_type = str(getattr(message, "type", ""))
        if message_type not in {"orderbook_snapshot", "orderbook_delta"}:
            raise ValueError(f"unsupported order-book message: {message_type}")
        sequence = int(getattr(message, "seq", 0))
        if sequence <= 0:
            raise ValueError("order-book message requires a positive sequence")

        expected_sequence = self.last_sequence + 1 if self.last_sequence is not None else None
        sequence_gap = expected_sequence is not None and sequence != expected_sequence
        if sequence_gap:
            for item in self.books.values():
                item.valid = False
        self.last_sequence = sequence

        book = self.books[ticker]
        raw_payload = {
            "type": message_type,
            "seq": sequence,
            "msg": vars(payload) if hasattr(payload, "__dict__") else payload,
        }
        raw_json = _canonical(raw_payload)
        event_hash = hashlib.sha256(raw_json.encode()).hexdigest()
        stable_id = hashlib.sha256(
            f"orderbook_events\0{ticker}\0{self.session_id}\0{sequence}\0{event_hash}".encode()
        ).hexdigest()
        source_timestamp = _source_time(payload)

        if message_type == "orderbook_snapshot":
            book.yes = _snapshot_side(payload, "yes")
            book.no = _snapshot_side(payload, "no")
            book.valid = True
            book.last_sequence = sequence
            event = OrderBookEvent(
                ticker=ticker, asset=self.assets_by_ticker[ticker],
                event_type="snapshot", source_timestamp=source_timestamp,
                collector_received_at=received_at, available_at=received_at,
                session_id=self.session_id, sequence=sequence,
                yes_levels=self._level_models(book.yes),
                no_levels=self._level_models(book.no),
                gap_detected=sequence_gap, book_valid_after_event=True,
                raw_payload_json=raw_json, event_sha256=event_hash,
                stable_row_id=stable_id,
            )
        else:
            side = str(getattr(payload, "side", ""))
            if side not in {"yes", "no"}:
                raise ValueError("order-book delta side must be yes or no")
            price = Decimal(str(getattr(payload, "price", getattr(payload, "price_dollars", ""))))
            delta = Decimal(str(getattr(payload, "delta", getattr(payload, "delta_fp", ""))))
            if book.valid and not sequence_gap:
                levels = book.yes if side == "yes" else book.no
                quantity = levels.get(price, Decimal("0")) + delta
                if quantity > 0:
                    levels[price] = quantity
                else:
                    levels.pop(price, None)
                book.last_sequence = sequence
            event = OrderBookEvent(
                ticker=ticker, asset=self.assets_by_ticker[ticker], event_type="delta",
                source_timestamp=source_timestamp, collector_received_at=received_at,
                available_at=received_at, session_id=self.session_id, sequence=sequence,
                side=side, price=price, signed_quantity_delta=delta,
                gap_detected=sequence_gap, book_valid_after_event=book.valid,
                raw_payload_json=raw_json, event_sha256=event_hash,
                stable_row_id=stable_id,
            )

        quote = self._quote(ticker, event, resnapshot=message_type == "orderbook_snapshot")
        checkpoint = self.checkpoint(ticker, received_at) if message_type == "orderbook_snapshot" else None
        needs_snapshot = tuple(sorted(key for key, value in self.books.items() if not value.valid))
        return BookApplyResult(
            event, quote, checkpoint, sequence_gap, expected_sequence, needs_snapshot
        )

    def _tops(self, book: _Book) -> tuple[Decimal | None, ...]:
        yes_bid = max(book.yes, default=None)
        no_bid = max(book.no, default=None)
        return (
            yes_bid, ONE - no_bid if no_bid is not None else None,
            no_bid, ONE - yes_bid if yes_bid is not None else None,
        )

    def _quote(
        self, ticker: str, event: OrderBookEvent, *, resnapshot: bool,
    ) -> ContractQuoteEvent | None:
        book = self.books[ticker]
        if not book.valid:
            return None
        tops = self._tops(book)
        if not resnapshot and tops == book.last_top:
            return None
        book.last_top = tops
        up_bid, up_ask, down_bid, down_ask = tops
        quote_payload = {
            "ticker": ticker, "sequence": event.sequence,
            "up_bid": up_bid, "up_ask": up_ask,
            "down_bid": down_bid, "down_ask": down_ask,
        }
        quote_hash = hashlib.sha256(_canonical(quote_payload).encode()).hexdigest()
        stable_id = hashlib.sha256(
            f"contract_quote_events\0{ticker}\0{self.session_id}\0{event.sequence}\0{quote_hash}".encode()
        ).hexdigest()
        asset = self.assets_by_ticker[ticker]
        return ContractQuoteEvent(
            ticker=ticker, asset=asset, series_ticker=self.series_by_asset[asset],
            source_timestamp=event.source_timestamp,
            collector_received_at=event.collector_received_at,
            available_at=event.available_at, up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            up_spread=(up_ask - up_bid) if up_ask is not None and up_bid is not None else None,
            down_spread=(down_ask - down_bid) if down_ask is not None and down_bid is not None else None,
            book_sequence=event.sequence, collector_session_id=self.session_id,
            event_kind="resnapshot" if resnapshot else "top_change",
            gap_detected=event.gap_detected, book_valid=True,
            event_sha256=quote_hash, stable_row_id=stable_id,
            raw_payload_json=event.raw_payload_json,
        )

    def _level_models(self, levels: Mapping[Decimal, Decimal]) -> tuple[BookLevel, ...]:
        return tuple(
            BookLevel(price=price, quantity=levels[price])
            for price in sorted(levels, reverse=True)[: self.depth]
        )

    def checkpoint(self, ticker: str, available_at: datetime) -> BookCheckpoint:
        book = self.books[ticker]
        yes_levels = self._level_models(book.yes)
        no_levels = self._level_models(book.no)
        up_bid, up_ask, down_bid, down_ask = self._tops(book)
        total = sum((item.quantity for item in yes_levels + no_levels), Decimal("0"))
        yes_total = sum((item.quantity for item in yes_levels), Decimal("0"))
        no_total = sum((item.quantity for item in no_levels), Decimal("0"))
        imbalance = (
            (yes_total - no_total) / (yes_total + no_total)
            if yes_total + no_total > 0 else None
        )
        def aggregate_depth(distance: Decimal) -> Decimal:
            value = Decimal("0")
            if up_bid is not None:
                value += sum((q for p, q in book.yes.items() if p >= up_bid - distance), Decimal("0"))
            if down_bid is not None:
                value += sum((q for p, q in book.no.items() if p >= down_bid - distance), Decimal("0"))
            return value
        state = _canonical({"yes": yes_levels, "no": no_levels, "sequence": book.last_sequence})
        reasons = () if book.valid else ("book_sequence_gap",)
        return BookCheckpoint(
            ticker=ticker, asset=self.assets_by_ticker[ticker],
            available_at=available_at.astimezone(UTC), session_id=self.session_id,
            last_sequence=book.last_sequence or 0, yes_levels=yes_levels, no_levels=no_levels,
            up_best_bid=up_bid, up_best_ask=up_ask,
            down_best_bid=down_bid, down_best_ask=down_ask,
            depth_1c=aggregate_depth(Decimal("0.01")),
            depth_3c=aggregate_depth(Decimal("0.03")),
            depth_5c=aggregate_depth(Decimal("0.05")),
            total_visible_depth=total, book_imbalance=imbalance,
            is_complete=book.valid, incomplete_reasons=reasons,
            state_sha256=hashlib.sha256(state.encode()).hexdigest(),
        )
