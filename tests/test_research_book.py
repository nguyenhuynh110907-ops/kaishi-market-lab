from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from kaishi_bot.research_book import ResearchOrderBookSet


NOW = datetime(2026, 8, 11, 12, tzinfo=UTC)


def message(kind, sequence, **payload):
    return SimpleNamespace(
        type=kind, seq=sequence,
        msg=SimpleNamespace(market_ticker="BTC-1", **payload),
    )


def books():
    return ResearchOrderBookSet(
        assets_by_ticker={"BTC-1": "BTC"},
        series_by_asset={"BTC": "KXBTC15M"}, session_id="s1", depth=10,
    )


def test_snapshot_and_delta_create_executable_up_down_quotes() -> None:
    state = books()
    first = state.apply(message(
        "orderbook_snapshot", 1,
        yes={Decimal("0.60"): Decimal("10")},
        no={Decimal("0.35"): Decimal("8")},
    ), NOW)
    assert first.quote is not None
    assert (first.quote.up_bid, first.quote.up_ask) == (
        Decimal("0.60"), Decimal("0.65"),
    )
    assert (first.quote.down_bid, first.quote.down_ask) == (
        Decimal("0.35"), Decimal("0.40"),
    )

    changed = state.apply(message(
        "orderbook_delta", 2, side="no", price=Decimal("0.38"),
        delta=Decimal("3"), ts_ms=1_786_428_000_000,
    ), NOW)
    assert changed.quote is not None
    assert changed.quote.up_ask == Decimal("0.62")
    assert changed.event.source_timestamp is not None


def test_non_top_delta_does_not_emit_redundant_quote() -> None:
    state = books()
    state.apply(message(
        "orderbook_snapshot", 1, yes={Decimal("0.60"): Decimal("10")},
        no={Decimal("0.35"): Decimal("8")},
    ), NOW)
    result = state.apply(message(
        "orderbook_delta", 2, side="yes", price=Decimal("0.20"),
        delta=Decimal("2"),
    ), NOW)
    assert result.quote is None


def test_sequence_gap_invalidates_book_until_new_snapshot() -> None:
    state = books()
    state.apply(message(
        "orderbook_snapshot", 1, yes={Decimal("0.60"): Decimal("10")},
        no={Decimal("0.35"): Decimal("8")},
    ), NOW)
    gap = state.apply(message(
        "orderbook_delta", 3, side="yes", price=Decimal("0.61"),
        delta=Decimal("2"),
    ), NOW)
    assert gap.sequence_gap is True
    assert gap.event.book_valid_after_event is False
    assert gap.quote is None
    assert gap.tickers_needing_snapshot == ("BTC-1",)

    still_invalid = state.apply(message(
        "orderbook_delta", 4, side="yes", price=Decimal("0.62"),
        delta=Decimal("2"),
    ), NOW)
    assert still_invalid.quote is None
    recovered = state.apply(message(
        "orderbook_snapshot", 5, yes={Decimal("0.62"): Decimal("4")},
        no={},
    ), NOW)
    assert recovered.quote is not None
    assert recovered.quote.book_valid is True
    assert recovered.quote.up_ask is None
    assert recovered.checkpoint is not None
    assert recovered.checkpoint.no_levels == ()


def test_event_identity_is_deterministic_within_session() -> None:
    left = books().apply(message(
        "orderbook_snapshot", 1, yes={Decimal("0.60"): Decimal("10")}, no={},
    ), NOW)
    right = books().apply(message(
        "orderbook_snapshot", 1, yes={Decimal("0.60"): Decimal("10")}, no={},
    ), NOW)
    assert left.event.stable_row_id == right.event.stable_row_id
    assert left.event.event_sha256 == right.event.event_sha256
