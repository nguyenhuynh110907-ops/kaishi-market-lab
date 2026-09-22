from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kaishi_bot.fees import FeeSchedule
from kaishi_bot.research_features import (
    BuiltObservation,
    FeatureObservation,
    RejectedObservation,
    ResearchObservationAdapter,
    PointInTimeFeatureBuilder,
    build_settlement_features,
)
from kaishi_bot.research_market_data import parse_market_payload
from kaishi_bot.research_models import ContractQuoteEvent, FeeMetadataVersion, RtiEvent


def market(close: datetime):
    return parse_market_payload("BTC", "KXBTC15M", {
        "ticker": "BTC-1", "series_ticker": "KXBTC15M",
        "open_time": (close - timedelta(minutes=15)).isoformat(),
        "close_time": close.isoformat(), "floor_strike": "100",
    }, close - timedelta(minutes=15))


def rti(source: datetime, available: datetime, price: str, *, average=None):
    source_ms = int(source.timestamp() * 1000)
    return RtiEvent(
        asset="BTC", index_id="BRTI", source_timestamp_ms=source_ms,
        source_time_utc=source, kalshi_received_at=available,
        collector_received_at=available, price=Decimal(price), session_id="s",
        upstream_latency_ms=0, transport_latency_ms=0, collector_latency_ms=0,
        avg_60s=Decimal(average) if average else None,
        avg_60s_window_size=60 if average else None,
        avg_60s_window_start_ms=source_ms - 59_000 if average else None,
        avg_60s_window_end_exclusive_ms=source_ms + 1_000 if average else None,
        raw_data_json="{}", event_sha256=f"{source_ms}-{price}",
    )


def test_settlement_window_is_open_left_closed_right_and_decimal() -> None:
    close = datetime(2026, 8, 11, 12, tzinfo=UTC)
    item = market(close)
    start = close - timedelta(seconds=60)
    events = [
        rti(start, start, "50"),
        rti(start + timedelta(seconds=1), start + timedelta(seconds=1), "99"),
        rti(close, close, "101"),
        rti(close + timedelta(seconds=1), close + timedelta(seconds=1), "200"),
    ]
    result = build_settlement_features(item, events, close, current_brti=Decimal("101"))
    assert result.locked_sample_count == 2
    assert result.locked_sample_sum == Decimal("200")
    assert result.required_remaining_average == Decimal("100")
    assert result.missing_sample_count == 58


def test_future_available_sample_and_duplicates_are_not_used() -> None:
    close = datetime(2026, 8, 11, 12, tzinfo=UTC)
    item = market(close)
    source = close - timedelta(seconds=30)
    visible = rti(source, source, "99")
    duplicate = rti(source, source + timedelta(seconds=1), "105")
    future = rti(source + timedelta(seconds=1), close + timedelta(seconds=5), "500")
    result = build_settlement_features(item, [future, duplicate, visible], close)
    assert result.locked_sample_count == 1
    assert result.locked_sample_sum == Decimal("99")


def test_server_average_mismatch_emits_quality_event() -> None:
    close = datetime(2026, 8, 11, 12, tzinfo=UTC)
    start = close - timedelta(seconds=60)
    events = [
        rti(
            start + timedelta(seconds=second),
            start + timedelta(seconds=second), "100",
            average="101" if second == 60 else None,
        )
        for second in range(1, 61)
    ]
    quality = []
    result = build_settlement_features(
        market(close), events, close,
        quality_event=lambda reason, detail: quality.append((reason, detail)),
    )
    assert result.locked_sample_count == 60
    assert quality[0][0] == "server_average_mismatch"


def feature(**changes):
    now = datetime(2026, 8, 11, 12, tzinfo=UTC)
    values = dict(
        dataset_version="d1", feature_schema_version="1", ticker="BTC-1",
        asset="BTC", observation_time=now, available_at=now,
        market_manifest_id="m", rti_manifest_id="r", quote_manifest_id="q",
        orderbook_manifest_id=None, fee_version="f", build_commit="abc",
        seconds_remaining=30, up_bid=Decimal("0.70"), up_ask=Decimal("0.71"),
        down_bid=Decimal("0.29"), down_ask=Decimal("0.30"),
        target_price=Decimal("100"), brti_price=Decimal("101"),
        brti_sigma_per_sqrt_second=Decimal("0.5"), locked_sample_count=1,
        locked_sample_sum=Decimal("99"), fee_metadata_available=True,
        feature_complete=True,
    )
    values.update(changes)
    return FeatureObservation(**values)


def test_adapter_builds_exact_market_observation_without_api() -> None:
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "f")
    result = ResearchObservationAdapter().build(feature(), fee_schedule=schedule)
    assert isinstance(result, BuiltObservation)
    assert result.observation.ticker == "BTC-1"
    assert result.observation.entry_fee_up == Decimal("0.02")


def test_adapter_rejects_missing_incomplete_and_future_rows() -> None:
    result = ResearchObservationAdapter().build(
        feature(
            up_ask=None, feature_complete=False,
            available_at=datetime(2026, 8, 11, 12, 0, 1, tzinfo=UTC),
        ),
        fee_schedule=None,
    )
    assert isinstance(result, RejectedObservation)
    assert {"missing_up_ask", "feature_incomplete", "missing_fee_metadata", "future_available_at"} <= set(result.reasons)


def quote(available: datetime, price: str) -> ContractQuoteEvent:
    value = Decimal(price)
    return ContractQuoteEvent(
        ticker="BTC-1", asset="BTC", series_ticker="KXBTC15M",
        collector_received_at=available, available_at=available,
        up_bid=value - Decimal("0.01"), up_ask=value,
        down_bid=Decimal("1") - value - Decimal("0.01"),
        down_ask=Decimal("1") - value, up_spread=Decimal("0.01"),
        down_spread=Decimal("0.01"), book_sequence=1,
        collector_session_id="s", book_valid=True,
        event_sha256=str(available.timestamp()), stable_row_id=str(available.timestamp()),
    )


def test_feature_builder_excludes_future_quote_and_rti_from_current_and_history() -> None:
    close = datetime(2026, 8, 11, 12, 15, tzinfo=UTC)
    item = market(close)
    observed = close - timedelta(seconds=30)
    rti_rows = [
        rti(observed - timedelta(seconds=2), observed - timedelta(seconds=2), "99"),
        rti(observed - timedelta(seconds=1), observed - timedelta(seconds=1), "100"),
        rti(observed, observed, "101"),
        rti(observed + timedelta(seconds=1), observed + timedelta(seconds=1), "999"),
    ]
    quotes = [
        quote(observed - timedelta(seconds=1), "0.70"),
        quote(observed, "0.71"),
        quote(observed + timedelta(seconds=1), "0.99"),
    ]
    fee = FeeMetadataVersion(
        fee_version="f", series_ticker="KXBTC15M", fee_type="quadratic",
        fee_multiplier=Decimal("1"), taker_rate=Decimal("0.07"),
        effective_from=item.open_time, observed_at=item.open_time,
        source_kind="test", source_payload_hash="h", raw_payload_json="{}",
    )
    built = PointInTimeFeatureBuilder().build(
        market=item, observation_time=observed, rti_events=rti_rows,
        quote_events=quotes, book_checkpoints=[], fee_version=fee,
        dataset_version="d", market_manifest_id="m", rti_manifest_id="r",
        quote_manifest_id="q", orderbook_manifest_id=None, build_commit="c",
    )
    assert built.up_ask == Decimal("0.71")
    assert built.brti_price == Decimal("101")
    assert Decimal("999") not in {
        value for _, value in built.contract_history if value is not None
    }
