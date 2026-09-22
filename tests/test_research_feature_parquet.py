from datetime import UTC, datetime
from decimal import Decimal

from kaishi_bot.research_feature_parquet import ParquetFeatureSink
from kaishi_bot.research_features import FeatureObservation
from kaishi_bot.research_store import ResearchStore


def feature():
    now = datetime(2026, 8, 11, 12, tzinfo=UTC)
    return FeatureObservation(
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


def test_feature_export_is_versioned_partitioned_and_links_parent_manifests(tmp_path) -> None:
    store = ResearchStore(tmp_path / "control.sqlite3")
    try:
        sink = ParquetFeatureSink(tmp_path / "research", store)
        assert sink.write([feature()], split="train") == 1
        row = store.connection.execute(
            "SELECT * FROM research_files WHERE dataset='feature_observations'"
        ).fetchone()
        assert row["row_count"] == 1
        assert "dataset_version=d1/split=train" in row["path"]
        assert set(__import__("json").loads(row["parent_manifest_ids_json"])) == {"m", "q", "r"}
        assert (tmp_path / "research" / row["path"]).is_file()
    finally:
        store.close()
