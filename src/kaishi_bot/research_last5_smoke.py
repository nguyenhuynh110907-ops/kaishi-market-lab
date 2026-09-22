from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kaishi_bot.agentic_strategy import (
    DEFAULT_FEATURE_NAMES,
    MarketObservation,
    TrainingRow,
    XGBoostConfig,
    XGBoostTrainer,
    chronological_ticker_split,
    observation_features,
    ticker_balanced_weights,
)
from kaishi_bot.research_xgboost_smoke import _load_markets, _read_parquet


DECISION_OFFSETS = (300, 240, 180, 120, 60)
CONTEXT_WINDOWS = (30, 60, 180, 300, 600)
CONTEXT_FEATURE_NAMES = (
    "decision_offset_seconds",
    "quote_age_seconds",
    "market_implied_yes",
    "spot_return_30s",
    "spot_return_60s",
    "spot_return_180s",
    "spot_return_300s",
    "spot_return_600s",
    "realized_volatility_60s",
    "realized_volatility_300s",
    "realized_volatility_600s",
    "distance_z_remaining",
    "up_mid_change_60s",
    "up_mid_change_300s",
    "has_quote_history_60s",
    "has_quote_history_300s",
)
LAST5_FEATURE_NAMES = DEFAULT_FEATURE_NAMES + CONTEXT_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class LastFiveDatasetSummary:
    asset: str
    eligible_markets: int
    represented_markets: int
    observations: int
    observations_per_market: float
    rejected: dict[str, int]
    first_close: str
    last_close: str


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    market_count: int
    observation_count: int
    yes_rate: float
    brier: float
    log_loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TradingMetrics:
    minimum_edge: float
    trades: int
    wins: int
    win_rate: float | None
    total_pnl_per_contract: float
    average_pnl_per_trade: float | None


def _valid_quote(row: Mapping[str, Any]) -> bool:
    values = (row["up_bid"], row["up_ask"], row["down_bid"], row["down_ask"])
    return (
        bool(row["book_valid"])
        and not bool(row["gap_detected"])
        and not bool(row["is_stale"])
        and all(value is not None for value in values)
        and Decimal(str(row["up_bid"])) <= Decimal(str(row["up_ask"]))
        and Decimal(str(row["down_bid"])) <= Decimal(str(row["down_ask"]))
    )


def _asof_index(times: Sequence[datetime], timestamp: datetime) -> int:
    return bisect.bisect_right(times, timestamp) - 1


def _raw_fee_per_contract(price: float, rate: float) -> float:
    return rate * price * (1.0 - price)


def _context_features(
    observed: datetime,
    decision_offset: int,
    quote: Mapping[str, Any],
    quote_rows: Sequence[Mapping[str, Any]],
    quote_times: Sequence[datetime],
    rti_rows: Sequence[Mapping[str, Any]],
    rti_times: Sequence[datetime],
    latest_rti_index: int,
    target: Decimal,
) -> tuple[dict[str, float], float]:
    latest_price = float(rti_rows[latest_rti_index]["price"])
    prices_by_window: dict[int, np.ndarray] = {}
    returns: dict[int, float] = {}
    for window in CONTEXT_WINDOWS:
        start = bisect.bisect_left(
            rti_times, observed - timedelta(seconds=window), 0, latest_rti_index + 1
        )
        values = np.asarray(
            [float(row["price"]) for row in rti_rows[start : latest_rti_index + 1]],
            dtype=np.float64,
        )
        prices_by_window[window] = values
        returns[window] = (
            (latest_price - values[0]) / values[0] if len(values) >= 2 and values[0] else 0.0
        )

    def realized_volatility(window: int) -> float:
        values = prices_by_window[window]
        if len(values) < 10:
            return 0.0
        log_returns = np.diff(np.log(values))
        return float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0

    up_mid = (float(quote["up_bid"]) + float(quote["up_ask"])) / 2.0
    down_mid = (float(quote["down_bid"]) + float(quote["down_ask"])) / 2.0
    denominator = up_mid + down_mid
    implied_yes = up_mid / denominator if denominator > 0 else 0.5

    quote_changes: dict[int, float] = {}
    quote_history: dict[int, float] = {}
    for window in (60, 300):
        index = _asof_index(quote_times, observed - timedelta(seconds=window))
        if index < 0:
            quote_changes[window] = 0.0
            quote_history[window] = 0.0
            continue
        previous = quote_rows[index]
        previous_mid = (float(previous["up_bid"]) + float(previous["up_ask"])) / 2.0
        quote_changes[window] = up_mid - previous_mid
        quote_history[window] = 1.0

    sigma_price = realized_volatility(60) * latest_price
    distance = latest_price - float(target)
    remaining_scale = max(sigma_price * math.sqrt(max(1, decision_offset)), float(target) * 1e-8)
    features = {
        "decision_offset_seconds": float(decision_offset),
        "quote_age_seconds": max(0.0, (observed - quote["available_at"]).total_seconds()),
        "market_implied_yes": implied_yes,
        **{f"spot_return_{window}s": returns[window] for window in CONTEXT_WINDOWS},
        "realized_volatility_60s": realized_volatility(60),
        "realized_volatility_300s": realized_volatility(300),
        "realized_volatility_600s": realized_volatility(600),
        "distance_z_remaining": distance / remaining_scale,
        "up_mid_change_60s": quote_changes[60],
        "up_mid_change_300s": quote_changes[300],
        "has_quote_history_60s": quote_history[60],
        "has_quote_history_300s": quote_history[300],
    }
    sigma_absolute = float(np.std(np.diff(prices_by_window[60]), ddof=1))
    return features, sigma_absolute


def build_last5_rows(
    database: Path,
    research_root: Path,
    *,
    asset: str,
    decision_offsets: Sequence[int] = DECISION_OFFSETS,
    context_seconds: int = 600,
    maximum_quote_age_seconds: int = 30,
    fee_rate: float = 0.07,
) -> tuple[list[TrainingRow], LastFiveDatasetSummary]:
    """Create five point-in-time decisions after a ten-minute warm-up.

    The first decision is T-300. Every feature is computed as-of its decision
    timestamp; the final label is used only after the whole ticker is assigned
    to one chronological split.
    """

    asset = asset.upper()
    offsets = tuple(sorted({int(value) for value in decision_offsets}, reverse=True))
    if not offsets or min(offsets) <= 0 or max(offsets) > 300:
        raise ValueError("last-five decision offsets must be in [1, 300]")
    if context_seconds < 600:
        raise ValueError("last-five smoke test requires at least ten minutes of context")
    markets = _load_markets(database, asset)
    if not markets:
        raise ValueError(f"no complete labeled markets for {asset}")

    quote_table = _read_parquet(research_root, "contract_quote_events", (
        "ticker", "asset", "available_at", "up_bid", "up_ask", "down_bid",
        "down_ask", "book_valid", "gap_detected", "is_stale",
    ), asset=asset)
    rti_table = _read_parquet(research_root, "rti_events", (
        "asset", "source_timestamp_ms", "collector_received_at", "price",
        "is_stale", "gap_detected",
    ), asset=asset)

    quotes_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    # Quote delta files are dense and numerous. Restrict them to labeled
    # sessions, the usable context window, and one latest row per ten-second
    # bucket in Arrow before converting anything to Python objects.
    import pyarrow as pa
    import pyarrow.compute as pc

    market_closes = pa.table({
        "ticker": list(markets),
        "_market_close": [markets[ticker]["close"] for ticker in markets],
    })
    quote_table = quote_table.join(market_closes, keys="ticker", join_type="inner")
    remaining_duration = pc.subtract(
        quote_table["_market_close"], quote_table["available_at"]
    )
    quote_table = quote_table.filter(pc.and_kleene(
        pc.greater(remaining_duration, pa.scalar(timedelta(0))),
        pc.less_equal(
            remaining_duration,
            pa.scalar(timedelta(seconds=context_seconds + max(offsets))),
        ),
    ))
    complete_mask = pc.and_kleene(
        pc.and_kleene(
            pc.equal(quote_table["book_valid"], True),
            pc.equal(quote_table["gap_detected"], False),
        ),
        pc.and_kleene(
            pc.equal(quote_table["is_stale"], False),
            pc.and_kleene(
                pc.and_kleene(
                    pc.is_valid(quote_table["up_bid"]),
                    pc.is_valid(quote_table["up_ask"]),
                ),
                pc.and_kleene(
                    pc.is_valid(quote_table["down_bid"]),
                    pc.is_valid(quote_table["down_ask"]),
                ),
            ),
        ),
    )
    crossed_mask = pc.and_kleene(
        complete_mask,
        pc.or_kleene(
            pc.greater(quote_table["up_bid"], quote_table["up_ask"]),
            pc.greater(quote_table["down_bid"], quote_table["down_ask"]),
        ),
    )
    rejected["invalid_quote"] += int(
        pc.sum(pc.cast(pc.invert(complete_mask), pa.int64())).as_py() or 0
    )
    rejected["crossed_quote"] += int(
        pc.sum(pc.cast(crossed_mask, pa.int64())).as_py() or 0
    )
    quote_table = quote_table.filter(
        pc.and_kleene(complete_mask, pc.invert(crossed_mask))
    )
    buckets = pc.cast(pc.floor(pc.divide(
        pc.cast(quote_table["available_at"], pa.int64()),
        pa.scalar(10_000_000, pa.int64()),
    )), pa.int64())
    quote_table = quote_table.append_column("_context_bucket", buckets)
    latest = quote_table.group_by(["ticker", "_context_bucket"]).aggregate([
        ("available_at", "max"),
    ])
    quote_table = quote_table.join(
        latest, keys=["ticker", "_context_bucket"], join_type="inner"
    )
    quote_table = quote_table.filter(pc.equal(
        quote_table["available_at"], quote_table["available_at_max"]
    ))
    for row in quote_table.to_pylist():
        ticker = str(row["ticker"])
        market = markets.get(ticker)
        if market is None:
            continue
        quotes_by_ticker[ticker].append(row)
    for values in quotes_by_ticker.values():
        values.sort(key=lambda row: row["available_at"])

    rti_rows = [
        row for row in rti_table.to_pylist()
        if str(row["asset"]) == asset
    ]
    rti_rows.sort(key=lambda row: row["collector_received_at"])
    rti_times = [row["collector_received_at"] for row in rti_rows]
    rows: list[TrainingRow] = []

    for ticker, market in sorted(markets.items(), key=lambda item: item[1]["close"]):
        quote_rows = quotes_by_ticker.get(ticker, [])
        if not quote_rows:
            rejected["missing_quotes"] += len(offsets)
            continue
        quote_times = [row["available_at"] for row in quote_rows]
        for offset in offsets:
            observed = market["close"] - timedelta(seconds=offset)
            quote_index = _asof_index(quote_times, observed)
            if quote_index < 0:
                rejected["missing_decision_quote"] += 1
                continue
            quote = quote_rows[quote_index]
            if (observed - quote["available_at"]).total_seconds() > maximum_quote_age_seconds:
                rejected["stale_decision_quote"] += 1
                continue
            rti_index = _asof_index(rti_times, observed)
            if rti_index < 0:
                rejected["missing_rti"] += 1
                continue
            latest_rti = rti_rows[rti_index]
            if (
                latest_rti["is_stale"] or latest_rti["gap_detected"]
                or observed - latest_rti["collector_received_at"] > timedelta(seconds=3)
            ):
                rejected["invalid_rti"] += 1
                continue
            context_start = bisect.bisect_left(
                rti_times, observed - timedelta(seconds=context_seconds), 0, rti_index + 1
            )
            if (
                rti_index - context_start < int(context_seconds * 0.8)
                or rti_times[context_start] > observed - timedelta(seconds=context_seconds - 5)
            ):
                rejected["incomplete_ten_minute_context"] += 1
                continue
            try:
                context, sigma = _context_features(
                    observed, offset, quote, quote_rows, quote_times,
                    rti_rows, rti_times, rti_index, market["target"],
                )
            except (ValueError, FloatingPointError):
                rejected["invalid_context"] += 1
                continue
            if not math.isfinite(sigma) or sigma <= 0:
                rejected["invalid_volatility"] += 1
                continue

            locked: dict[int, Decimal] = {}
            settlement_start_ms = int((market["close"] - timedelta(seconds=60)).timestamp() * 1000)
            for event in rti_rows[max(0, rti_index - 180) : rti_index + 1]:
                source_ms = int(event["source_timestamp_ms"])
                if settlement_start_ms < source_ms <= int(market["close"].timestamp() * 1000):
                    locked.setdefault(source_ms, Decimal(str(event["price"])))
            up_ask = float(quote["up_ask"])
            down_ask = float(quote["down_ask"])
            observation = MarketObservation(
                ticker=ticker,
                observed_at=observed,
                seconds_remaining=offset,
                up_bid=Decimal(str(quote["up_bid"])),
                up_ask=Decimal(str(quote["up_ask"])),
                down_bid=Decimal(str(quote["down_bid"])),
                down_ask=Decimal(str(quote["down_ask"])),
                target_price=market["target"],
                brti_price=Decimal(str(latest_rti["price"])),
                brti_sigma_per_sqrt_second=Decimal(str(sigma)),
                locked_sample_count=len(locked),
                locked_sample_sum=sum(locked.values(), Decimal("0")),
                entry_fee_up=Decimal(str(_raw_fee_per_contract(up_ask, fee_rate))),
                entry_fee_down=Decimal(str(_raw_fee_per_contract(down_ask, fee_rate))),
            )
            features = observation_features(observation)
            features.update(context)
            if not all(math.isfinite(float(value)) for value in features.values()):
                rejected["non_finite_features"] += 1
                continue
            rows.append(TrainingRow(
                ticker=ticker,
                market_close_time=market["close"],
                observed_at=observed,
                label_yes=market["label"],
                features=features,
            ))

    represented = {row.ticker for row in rows}
    if not represented:
        raise ValueError("last-five filters produced no observations")
    closes = [markets[ticker]["close"] for ticker in represented]
    return rows, LastFiveDatasetSummary(
        asset=asset,
        eligible_markets=len(markets),
        represented_markets=len(represented),
        observations=len(rows),
        observations_per_market=len(rows) / len(represented),
        rejected=dict(sorted(rejected.items())),
        first_close=min(closes).isoformat(),
        last_close=max(closes).isoformat(),
    )


def _probability_metrics(
    rows: Sequence[TrainingRow], predictions: np.ndarray
) -> ProbabilityMetrics:
    labels = np.asarray([row.label_yes for row in rows], dtype=np.float64)
    weights = ticker_balanced_weights(rows)
    weights /= weights.sum()
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    return ProbabilityMetrics(
        market_count=len({row.ticker for row in rows}),
        observation_count=len(rows),
        yes_rate=float(np.sum(weights * labels)),
        brier=float(np.sum(weights * (predictions - labels) ** 2)),
        log_loss=float(-np.sum(weights * (
            labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
        ))),
        accuracy=float(np.sum(weights * ((predictions >= 0.5) == labels))),
    )


def _market_predictions(rows: Sequence[TrainingRow]) -> np.ndarray:
    return np.asarray([float(row.features["market_implied_yes"]) for row in rows])


def _trade_metrics(
    rows: Sequence[TrainingRow], predictions: np.ndarray, minimum_edge: float
) -> TradingMetrics:
    by_ticker: dict[str, list[tuple[TrainingRow, float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_ticker[row.ticker].append((row, float(prediction)))
    pnl: list[float] = []
    wins = 0
    for decisions in by_ticker.values():
        # The first qualifying signal is reproducible online. Selecting the
        # maximum edge in hindsight would leak the rest of the five-minute window.
        for row, prediction in sorted(decisions, key=lambda item: item[0].observed_at):
            up_ask = float(row.features["up_ask"])
            down_ask = float(row.features["down_ask"])
            up_fee = float(row.features["entry_fee_up"])
            down_fee = float(row.features["entry_fee_down"])
            candidates = (
                (prediction - up_ask - up_fee, row.label_yes - up_ask - up_fee),
                ((1.0 - prediction) - down_ask - down_fee,
                 (1 - row.label_yes) - down_ask - down_fee),
            )
            edge, realized = max(candidates, key=lambda item: item[0])
            if edge >= minimum_edge:
                pnl.append(realized)
                wins += int(realized > 0)
                break
    return TradingMetrics(
        minimum_edge=minimum_edge,
        trades=len(pnl),
        wins=wins,
        win_rate=(wins / len(pnl)) if pnl else None,
        total_pnl_per_contract=float(sum(pnl)),
        average_pnl_per_trade=(float(np.mean(pnl)) if pnl else None),
    )


def run_last5_smoke(
    database: Path,
    research_root: Path,
    artifact_directory: Path,
    *,
    asset: str,
    device: str = "cuda",
) -> dict[str, Any]:
    rows, summary = build_last5_rows(database, research_root, asset=asset)
    split = chronological_ticker_split(rows, embargo_tickers=1)
    trainer = XGBoostTrainer(XGBoostConfig(
        max_depth=2,
        # Ticker-balanced weights give each market total weight 1. With only
        # ~30 training markets a conventional value such as 4 prevents every
        # split (the root Hessian itself is only about N/4).
        min_child_weight=0.25,
        learning_rate=0.04,
        rounds=250,
        early_stopping_rounds=25,
        calibration_bins=4,
        device=device,
    ), feature_names=LAST5_FEATURE_NAMES)
    estimator = trainer.train(rows, split, artifact_directory)
    test_tickers = set(split.test_tickers)
    test_rows = [row for row in rows if row.ticker in test_tickers]
    model_predictions = np.asarray([
        float(estimator.predict_features(row.features)[0]) for row in test_rows
    ])
    market_predictions = _market_predictions(test_rows)
    yes_rate = _probability_metrics(
        test_rows,
        np.full(len(test_rows), np.mean([row.label_yes for row in test_rows])),
    ).yes_rate
    majority_predictions = np.full(
        len(test_rows), 1.0 if yes_rate >= 0.5 else 0.0
    )
    report = {
        "purpose": "last_five_minutes_technical_smoke_test_only",
        "warning": "Small held-out market count; do not use for live trading.",
        "strategy": {
            "warmup_seconds": 600,
            "decision_offsets_seconds": list(DECISION_OFFSETS),
            "entry_policy": "first qualifying signal; hold to settlement",
            "fee_rate": 0.07,
            "training_device": device,
        },
        "dataset": asdict(summary),
        "split": asdict(split),
        "test": {
            "model": asdict(_probability_metrics(test_rows, model_predictions)),
            "kalshi_market": asdict(_probability_metrics(test_rows, market_predictions)),
            "majority_accuracy": asdict(
                _probability_metrics(test_rows, majority_predictions)
            )["accuracy"],
            "trading": [
                asdict(_trade_metrics(test_rows, model_predictions, threshold))
                for threshold in (0.0, 0.02, 0.04)
            ],
        },
        "artifact_directory": str(artifact_directory),
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "last5_smoke_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 10-minute-context/last-5 smoke model")
    parser.add_argument("--database", type=Path, default=Path("dashboard.sqlite3"))
    parser.add_argument("--research-root", type=Path, default=Path("data/research"))
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run_last5_smoke(
        args.database, args.research_root, args.artifact_directory,
        asset=args.asset, device=args.device,
    ), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
