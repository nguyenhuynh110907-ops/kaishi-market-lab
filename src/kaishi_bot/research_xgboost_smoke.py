from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kaishi_bot.agentic_strategy import (
    MarketObservation,
    TrainingRow,
    XGBoostConfig,
    XGBoostTrainer,
    chronological_ticker_split,
    observation_features,
    ticker_balanced_weights,
)


@dataclass(frozen=True, slots=True)
class SmokeDatasetSummary:
    asset: str
    eligible_markets: int
    represented_markets: int
    observations: int
    rejected: dict[str, int]
    first_close: str
    last_close: str


@dataclass(frozen=True, slots=True)
class SmokeMetrics:
    market_count: int
    observation_count: int
    yes_rate: float
    brier: float
    log_loss: float
    accuracy: float


def _load_markets(database: Path, asset: str) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT ticker,asset,open_time,close_time,target_price,official_result
               FROM research_markets
               WHERE asset=? AND quality_status='complete'
                 AND official_result IN ('yes','no') AND target_price IS NOT NULL""",
            (asset,),
        ).fetchall()
    finally:
        connection.close()
    return {
        str(row["ticker"]): {
            "asset": str(row["asset"]),
            "open": datetime.fromisoformat(str(row["open_time"])),
            "close": datetime.fromisoformat(str(row["close_time"])),
            "target": Decimal(str(row["target_price"])),
            "label": 1 if row["official_result"] == "yes" else 0,
        }
        for row in rows
    }


def _read_parquet(
    root: Path, dataset: str, columns: Sequence[str], *, asset: str | None = None,
):
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError("smoke dataset build requires: pip install -e '.[research]'") from error
    path = root / dataset
    if not path.is_dir():
        raise ValueError(f"missing research dataset: {path}")
    source = ds.dataset(path, format="parquet", partitioning="hive")
    predicate = ds.field("asset") == asset if asset is not None else None
    return source.to_table(columns=list(columns), filter=predicate)


def build_smoke_rows(
    database: Path,
    research_root: Path,
    *,
    asset: str = "BTC",
    maximum_seconds_remaining: int = 390,
    minimum_seconds_remaining: int = 8,
    sample_interval_seconds: int = 1,
) -> tuple[list[TrainingRow], SmokeDatasetSummary]:
    """Build a conservative point-in-time dataset for a technical smoke test."""

    if maximum_seconds_remaining <= minimum_seconds_remaining:
        raise ValueError("invalid observation window")
    if sample_interval_seconds < 1:
        raise ValueError("sample interval must be positive")
    asset = asset.upper()
    markets = _load_markets(database, asset)
    if not markets:
        raise ValueError(f"no complete labeled markets for {asset}")

    quotes = _read_parquet(research_root, "contract_quote_events", (
        "ticker", "asset", "available_at", "up_bid", "up_ask", "down_bid",
        "down_ask", "book_valid", "gap_detected", "is_stale",
    ), asset=asset)
    rti = _read_parquet(research_root, "rti_events", (
        "asset", "source_timestamp_ms", "collector_received_at", "price",
        "is_stale", "gap_detected",
    ), asset=asset)

    # Keep the latest valid quote observed in each sampling bucket. The row's
    # actual available_at remains its decision time, so this does not move data
    # backward in time.
    sampled: dict[tuple[str, int], dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    # Push ticker, decision-window, and quality filtering into Arrow before
    # materializing Python dictionaries. This keeps multi-asset training from
    # converting hundreds of thousands of irrelevant quote rows per asset.
    import pyarrow as pa
    import pyarrow.compute as pc

    market_closes = pa.table({
        "ticker": list(markets),
        "_market_close": [markets[ticker]["close"] for ticker in markets],
    })
    quotes = quotes.join(market_closes, keys="ticker", join_type="inner")
    remaining_duration = pc.subtract(
        quotes["_market_close"], quotes["available_at"]
    )
    quotes = quotes.filter(pc.and_kleene(
        pc.greater(
            remaining_duration,
            pa.scalar(timedelta(seconds=minimum_seconds_remaining)),
        ),
        pc.less_equal(
            remaining_duration,
            pa.scalar(timedelta(seconds=maximum_seconds_remaining)),
        ),
    ))
    complete_mask = pc.and_kleene(
        pc.and_kleene(
            pc.equal(quotes["book_valid"], True),
            pc.equal(quotes["gap_detected"], False),
        ),
        pc.and_kleene(
            pc.equal(quotes["is_stale"], False),
            pc.and_kleene(
                pc.and_kleene(
                    pc.is_valid(quotes["up_bid"]), pc.is_valid(quotes["up_ask"]),
                ),
                pc.and_kleene(
                    pc.is_valid(quotes["down_bid"]), pc.is_valid(quotes["down_ask"]),
                ),
            ),
        ),
    )
    crossed_mask = pc.and_kleene(
        complete_mask,
        pc.or_kleene(
            pc.greater(quotes["up_bid"], quotes["up_ask"]),
            pc.greater(quotes["down_bid"], quotes["down_ask"]),
        ),
    )
    rejected["invalid_quote"] += int(
        pc.sum(pc.cast(pc.invert(complete_mask), pa.int64())).as_py() or 0
    )
    rejected["crossed_quote"] += int(
        pc.sum(pc.cast(crossed_mask, pa.int64())).as_py() or 0
    )
    quotes = quotes.filter(pc.and_kleene(complete_mask, pc.invert(crossed_mask)))
    bucket_divisor = 1_000_000 * sample_interval_seconds
    buckets = pc.cast(
        pc.floor(pc.divide(
            pc.cast(quotes["available_at"], pa.int64()),
            pa.scalar(bucket_divisor, pa.int64()),
        )),
        pa.int64(),
    )
    quotes = quotes.append_column("_sample_bucket", buckets)
    latest = quotes.group_by(["ticker", "_sample_bucket"]).aggregate([
        ("available_at", "max"),
    ])
    quotes = quotes.join(
        latest, keys=["ticker", "_sample_bucket"], join_type="inner"
    )
    quotes = quotes.filter(pc.equal(
        quotes["available_at"], quotes["available_at_max"]
    ))
    for row in quotes.to_pylist():
        ticker = str(row["ticker"])
        market = markets.get(ticker)
        if market is None or str(row["asset"]) != asset:
            continue
        observed = row["available_at"]
        bucket = int(row["_sample_bucket"])
        key = (ticker, bucket)
        previous = sampled.get(key)
        if previous is None or previous["available_at"] < observed:
            sampled[key] = row

    rti_rows = [
        row for row in rti.to_pylist()
        if str(row["asset"]) == asset
    ]
    rti_rows.sort(key=lambda row: row["collector_received_at"])
    received = [row["collector_received_at"] for row in rti_rows]
    rows: list[TrainingRow] = []

    for quote in sorted(sampled.values(), key=lambda row: (row["ticker"], row["available_at"])):
        ticker = str(quote["ticker"])
        market = markets[ticker]
        observed = quote["available_at"]
        visible_end = bisect.bisect_right(received, observed)
        if visible_end == 0:
            rejected["missing_rti"] += 1
            continue
        latest = rti_rows[visible_end - 1]
        if (
            latest["is_stale"] or latest["gap_detected"]
            or observed - latest["collector_received_at"] > timedelta(seconds=3)
        ):
            rejected["invalid_rti"] += 1
            continue

        history_start = bisect.bisect_left(
            received, observed - timedelta(seconds=65), 0, visible_end
        )
        history = [
            item for item in rti_rows[history_start:visible_end]
            if not item["is_stale"] and not item["gap_detected"]
        ]
        history_prices = np.asarray([float(item["price"]) for item in history])
        if len(history_prices) < 10:
            rejected["insufficient_volatility"] += 1
            continue
        changes = np.diff(history_prices)
        sigma = float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0
        if not math.isfinite(sigma):
            rejected["invalid_volatility"] += 1
            continue

        locked_by_timestamp: dict[int, Decimal] = {}
        window_start_ms = int((market["close"] - timedelta(seconds=60)).timestamp() * 1000)
        window_end_ms = int(market["close"].timestamp() * 1000)
        for item in rti_rows[max(0, visible_end - 180):visible_end]:
            source_ms = int(item["source_timestamp_ms"])
            if window_start_ms < source_ms <= window_end_ms:
                locked_by_timestamp.setdefault(source_ms, Decimal(str(item["price"])))
        if len(locked_by_timestamp) > 60:
            rejected["too_many_settlement_samples"] += 1
            continue

        remaining = max(0, int((market["close"] - observed).total_seconds()))
        observation = MarketObservation(
            ticker=ticker,
            observed_at=observed,
            seconds_remaining=remaining,
            up_bid=Decimal(str(quote["up_bid"])),
            up_ask=Decimal(str(quote["up_ask"])),
            down_bid=Decimal(str(quote["down_bid"])),
            down_ask=Decimal(str(quote["down_ask"])),
            target_price=market["target"],
            brti_price=Decimal(str(latest["price"])),
            brti_sigma_per_sqrt_second=Decimal(str(sigma)),
            locked_sample_count=len(locked_by_timestamp),
            locked_sample_sum=sum(locked_by_timestamp.values(), Decimal("0")),
        )
        rows.append(TrainingRow(
            ticker=ticker,
            market_close_time=market["close"],
            observed_at=observed,
            label_yes=market["label"],
            features=observation_features(observation),
        ))

    represented = {row.ticker for row in rows}
    if not represented:
        raise ValueError("point-in-time filters produced no smoke observations")
    closes = [markets[ticker]["close"] for ticker in represented]
    return rows, SmokeDatasetSummary(
        asset=asset,
        eligible_markets=len(markets),
        represented_markets=len(represented),
        observations=len(rows),
        rejected=dict(sorted(rejected.items())),
        first_close=min(closes).isoformat(),
        last_close=max(closes).isoformat(),
    )


def _metrics(rows: Sequence[TrainingRow], estimator) -> SmokeMetrics:
    predictions = np.asarray([
        float(estimator.predict_features(row.features)[0]) for row in rows
    ])
    labels = np.asarray([row.label_yes for row in rows], dtype=np.float64)
    weights = ticker_balanced_weights(rows)
    weights /= weights.sum()
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    return SmokeMetrics(
        market_count=len({row.ticker for row in rows}),
        observation_count=len(rows),
        yes_rate=float(np.sum(weights * labels)),
        brier=float(np.sum(weights * (predictions - labels) ** 2)),
        log_loss=float(-np.sum(weights * (
            labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
        ))),
        accuracy=float(np.sum(weights * ((predictions >= 0.5) == labels))),
    )


def run_smoke(
    database: Path,
    research_root: Path,
    artifact_directory: Path,
    *,
    asset: str = "BTC",
) -> dict[str, Any]:
    rows, summary = build_smoke_rows(database, research_root, asset=asset)
    split = chronological_ticker_split(rows, embargo_tickers=1)
    trainer = XGBoostTrainer(XGBoostConfig(
        max_depth=3,
        min_child_weight=12,
        learning_rate=0.05,
        rounds=250,
        early_stopping_rounds=25,
        calibration_bins=8,
    ))
    estimator = trainer.train(rows, split, artifact_directory)
    test_rows = [row for row in rows if row.ticker in set(split.test_tickers)]
    report = {
        "purpose": "technical_smoke_test_only",
        "warning": "Do not use this report for hyperparameter selection or live trading.",
        "dataset": asdict(summary),
        "split": asdict(split),
        "test": asdict(_metrics(test_rows, estimator)),
        "artifact_directory": str(artifact_directory),
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "smoke_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-data XGBoost smoke test")
    parser.add_argument("--database", type=Path, default=Path("dashboard.sqlite3"))
    parser.add_argument("--research-root", type=Path, default=Path("data/research"))
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--asset", default="BTC")
    args = parser.parse_args()
    print(json.dumps(run_smoke(
        args.database, args.research_root, args.artifact_directory, asset=args.asset,
    ), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
