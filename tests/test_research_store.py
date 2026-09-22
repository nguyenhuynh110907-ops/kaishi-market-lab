import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kaishi_bot.research_market_data import parse_market_payload
from kaishi_bot.research_store import ResearchStore
from kaishi_bot.research_models import FeeMetadataVersion


def make_market(now: datetime, target: str = "63310.36000000"):
    return parse_market_payload("BTC", "KXBTC15M", {
        "ticker": "KXBTC15M-TEST", "series_ticker": "KXBTC15M",
        "title": "BTC 15 minute", "open_time": now.isoformat(),
        "close_time": (now + timedelta(minutes=15)).isoformat(),
        "floor_strike": Decimal(target),
    }, now)


def test_research_store_persists_exact_target_and_versions_payloads(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    try:
        now = datetime(2026, 8, 11, 19, 0, tzinfo=UTC)
        market = make_market(now)
        assert store.save_market(market) is True
        assert store.save_market(market.model_copy(update={"refreshed_at": now + timedelta(seconds=5)})) is False
        changed_payload = {**market.raw_payload, "title": "changed"}
        changed = parse_market_payload(
            "BTC", "KXBTC15M", changed_payload, now + timedelta(seconds=10)
        )
        assert store.save_market(changed) is True

        row = store.connection.execute(
            "SELECT * FROM research_markets WHERE ticker=?", (market.ticker,)
        ).fetchone()
        versions = store.connection.execute(
            "SELECT COUNT(*) FROM research_market_versions WHERE ticker=?",
            (market.ticker,),
        ).fetchone()[0]
        assert row["target_price"] == "63310.36000000"
        assert row["target_source_field"] == "floor_strike"
        assert row["target_window_start"] == (now - timedelta(seconds=60)).isoformat()
        assert versions == 2
        latest = store.latest_markets_by_asset(("BTC", "ETH"))
        assert latest["BTC"]["target_price"] == "63310.36000000"
        assert "ETH" not in latest
    finally:
        store.close()


def test_error_quality_event_marks_market_incomplete(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    try:
        now = datetime(2026, 8, 11, 19, 0, tzinfo=UTC)
        market = make_market(now)
        store.save_market(market)
        store.quality_event(
            stream="rti_events", reason_code="rti_gap", severity="error",
            ticker=market.ticker, index_id="BRTI", missing_sample_count=2,
        )
        row = store.connection.execute(
            "SELECT * FROM research_market_quality WHERE ticker=?", (market.ticker,)
        ).fetchone()
        projection = store.connection.execute(
            "SELECT quality_status FROM research_markets WHERE ticker=?", (market.ticker,)
        ).fetchone()[0]
        assert row["status"] == "incomplete"
        assert row["rti_gaps"] == 1
        assert projection == "incomplete"

        # Routine metadata refreshes cannot erase an observed RTI gap.
        store.save_market(
            market.model_copy(update={"refreshed_at": now + timedelta(seconds=15)})
        )
        refreshed = store.connection.execute(
            "SELECT * FROM research_market_quality WHERE ticker=?", (market.ticker,)
        ).fetchone()
        assert refreshed["status"] == "incomplete"
        assert refreshed["reason_code"] == "rti_gap"
        assert refreshed["rti_gaps"] == 1
    finally:
        store.close()


def test_legacy_metadata_backfill_is_bounded_and_stops_after_three_failures(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    try:
        store.connection.execute(
            """CREATE TABLE quote_events(
                id INTEGER PRIMARY KEY,asset TEXT,ticker TEXT,observed_at TEXT
            )"""
        )
        store.connection.execute(
            "INSERT INTO quote_events VALUES (1,'BTC','OLD-TICKER','2026-08-11T10:00:00+00:00')"
        )
        store.connection.commit()
        assert store.legacy_markets_missing_metadata() == [("BTC", "OLD-TICKER")]
        for _ in range(3):
            store.record_backfill_attempt("OLD-TICKER", "failed", "HTTPStatusError")
        assert store.legacy_markets_missing_metadata() == []
    finally:
        store.close()


def test_transient_stale_quality_can_recover_but_a_real_gap_cannot(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    try:
        now = datetime(2026, 8, 11, 19, 0, tzinfo=UTC)
        market = make_market(now)
        store.save_market(market)
        store.quality_event(
            stream="rti_events", reason_code="index_not_updating", severity="error",
            ticker=market.ticker, index_id="BRTI",
        )
        assert store.recover_market_quality(
            market.ticker, {"index_not_updating", "stale_source"}, now
        ) is True
        assert store.connection.execute(
            "SELECT quality_status FROM research_markets WHERE ticker=?", (market.ticker,)
        ).fetchone()[0] == "complete"

        store.quality_event(
            stream="rti_events", reason_code="rti_gap", severity="error",
            ticker=market.ticker, index_id="BRTI",
        )
        assert store.recover_market_quality(
            market.ticker, {"index_not_updating", "stale_source"}, now
        ) is False
    finally:
        store.close()
def test_phase_a_migration_versions_fees_and_quality_dimensions(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    now = datetime(2026, 8, 11, tzinfo=UTC)
    version = FeeMetadataVersion(
        fee_version="fee-v1", series_ticker="KXBTC15M", fee_type="quadratic",
        fee_multiplier=Decimal("1"), taker_rate=Decimal("0.07"),
        effective_from=now, observed_at=now, source_kind="fee_changes",
        source_payload_hash="abc", raw_payload_json="{}",
    )
    try:
        assert store.save_fee_version(version) is True
        assert store.save_fee_version(version) is False
        assert store.fee_version_at("KXBTC15M", now).fee_version == "fee-v1"
        assert store.fee_version_at("KXBTC15M", now - timedelta(seconds=1)) is None
        store.set_quality_dimensions(
            "BTC-1", metadata_complete=True, target_complete=True,
            incomplete_reasons=["missing_fee_metadata"],
        )
        row = store.connection.execute(
            "SELECT * FROM research_quality_dimensions WHERE ticker='BTC-1'"
        ).fetchone()
        assert row["metadata_complete"] == 1
        assert row["fee_metadata_complete"] == 0
        assert json.loads(row["incomplete_reasons_json"]) == ["missing_fee_metadata"]
    finally:
        store.close()
