from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx
import numpy as np

from kaishi_bot.agentic_strategy import (
    TrainingRow,
    XGBoostConfig,
    XGBoostTrainer,
    chronological_ticker_split,
    ticker_balanced_weights,
)
from kaishi_bot.market_data import PRODUCTION_BASE_URL


HISTORICAL_FEATURE_NAMES = (
    "seconds_remaining",
    "market_yes_mid",
    "yes_bid_close",
    "yes_ask_close",
    "spread_close",
    "trade_close",
    "trade_mean",
    "log_volume",
    "log_open_interest",
    "bid_range",
    "ask_range",
    "trade_range",
    "mid_momentum_1m",
    "mid_momentum_3m",
    "mid_volatility_3m",
    "mid_volatility_5m",
    "relative_volume_3m",
)


@dataclass(frozen=True, slots=True)
class HistoricalMarket:
    ticker: str
    asset: str
    series_ticker: str
    open_time: datetime
    close_time: datetime
    target_price: Decimal
    label_yes: int


@dataclass(frozen=True, slots=True)
class HistoricalCandle:
    ticker: str
    end_time: datetime
    yes_bid_open: float
    yes_bid_high: float
    yes_bid_low: float
    yes_bid_close: float
    yes_ask_open: float
    yes_ask_high: float
    yes_ask_low: float
    yes_ask_close: float
    trade_open: float | None
    trade_high: float | None
    trade_low: float | None
    trade_close: float | None
    trade_mean: float | None
    volume: float
    open_interest: float


@dataclass(frozen=True, slots=True)
class HistoricalDatasetManifest:
    dataset_version: str
    source: str
    asset: str
    series_ticker: str
    start_time: str
    end_time: str
    market_count: int
    candle_count: int
    created_at: str
    candle_period_minutes: int = 1
    point_in_time_rule: str = "candle_available_at_end_period_ts"


def _time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _decimal(payload: dict[str, Any], *names: str) -> Decimal | None:
    for name in names:
        value = payload.get(name)
        if value not in {None, ""}:
            return Decimal(str(value))
    return None


def _number(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(f"{name}_dollars", payload.get(name))
    return float(value) if value not in {None, ""} else None


class HistoricalKalshiClient:
    """Public, read-only client for settled markets and one-minute candles."""

    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    @classmethod
    def production(cls) -> "HistoricalKalshiClient":
        return cls(httpx.Client(base_url=PRODUCTION_BASE_URL, timeout=30.0))

    def close(self) -> None:
        self.client.close()

    def _get(self, path: str, *, params: dict[str, object]) -> dict[str, Any]:
        last_error: Exception | None = None
        response_detail = ""
        for attempt in range(4):
            try:
                response = self.client.get(path, params=params)
                response_detail = response.text[:500]
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return json.loads(response.content, parse_float=Decimal)
            except (httpx.HTTPError, json.JSONDecodeError) as error:
                last_error = error
                if attempt == 3:
                    break
                time.sleep(0.25 * (2 ** attempt))
        assert last_error is not None
        raise RuntimeError(
            f"Kalshi historical request failed: {path}: {response_detail}"
        ) from last_error

    def settled_markets(
        self,
        *,
        asset: str,
        series_ticker: str,
        start: datetime,
        end: datetime,
        maximum_markets: int | None = None,
    ) -> list[HistoricalMarket]:
        cursor = ""
        result: list[HistoricalMarket] = []
        while True:
            document = self._get("/markets", params={
                "status": "settled",
                "series_ticker": series_ticker,
                "min_close_ts": int(start.timestamp()),
                "max_close_ts": int(end.timestamp()),
                "limit": 1000,
                "cursor": cursor,
            })
            for payload in document.get("markets", []):
                close_time = _time(payload["close_time"])
                open_time = _time(payload["open_time"])
                label = str(payload.get("result") or "").lower()
                target = _decimal(payload, "floor_strike_dollars", "floor_strike")
                if (
                    not start <= close_time <= end
                    or label not in {"yes", "no"}
                    or target is None
                ):
                    continue
                result.append(HistoricalMarket(
                    ticker=str(payload["ticker"]),
                    asset=asset.upper(),
                    series_ticker=series_ticker,
                    open_time=open_time,
                    close_time=close_time,
                    target_price=target,
                    label_yes=int(label == "yes"),
                ))
            cursor = str(document.get("cursor") or "")
            if not cursor or (maximum_markets and len(result) >= maximum_markets):
                break
        ordered = sorted(
            {market.ticker: market for market in result}.values(),
            key=lambda market: (market.close_time, market.ticker),
        )
        return ordered[-maximum_markets:] if maximum_markets else ordered

    def candles(self, markets: Sequence[HistoricalMarket]) -> list[HistoricalCandle]:
        result: list[HistoricalCandle] = []
        # Although the API documents 100 tickers, long 15-minute ticker names
        # can exceed the gateway's practical query-string limit. Twenty-five
        # keeps requests bounded while remaining efficient.
        batch_size = 25
        for offset in range(0, len(markets), batch_size):
            batch = markets[offset:offset + batch_size]
            document = self._get("/markets/candlesticks", params={
                "market_tickers": ",".join(item.ticker for item in batch),
                "start_ts": int(min(item.open_time for item in batch).timestamp()),
                "end_ts": int(max(item.close_time for item in batch).timestamp()),
                "period_interval": 1,
            })
            lookup = {item.ticker: item for item in batch}
            for bundle in document.get("markets", []):
                ticker = str(bundle.get("market_ticker") or "")
                market = lookup.get(ticker)
                if market is None:
                    continue
                for payload in bundle.get("candlesticks") or []:
                    end_time = datetime.fromtimestamp(int(payload["end_period_ts"]), tz=UTC)
                    if not market.open_time < end_time <= market.close_time:
                        continue
                    bid = payload["yes_bid"]
                    ask = payload["yes_ask"]
                    price = payload.get("price") or {}
                    required = (
                        _number(bid, "open"), _number(bid, "high"),
                        _number(bid, "low"), _number(bid, "close"),
                        _number(ask, "open"), _number(ask, "high"),
                        _number(ask, "low"), _number(ask, "close"),
                    )
                    if any(value is None for value in required):
                        continue
                    result.append(HistoricalCandle(
                        ticker=ticker,
                        end_time=end_time,
                        yes_bid_open=float(required[0]), yes_bid_high=float(required[1]),
                        yes_bid_low=float(required[2]), yes_bid_close=float(required[3]),
                        yes_ask_open=float(required[4]), yes_ask_high=float(required[5]),
                        yes_ask_low=float(required[6]), yes_ask_close=float(required[7]),
                        trade_open=_number(price, "open"),
                        trade_high=_number(price, "high"),
                        trade_low=_number(price, "low"),
                        trade_close=_number(price, "close"),
                        trade_mean=_number(price, "mean"),
                        volume=float(payload.get("volume_fp", payload.get("volume", 0))),
                        open_interest=float(payload.get(
                            "open_interest_fp", payload.get("open_interest", 0)
                        )),
                    ))
        return sorted(result, key=lambda item: (item.ticker, item.end_time))


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("historical pretraining requires: pip install -e '.[research]'") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, temporary, compression="zstd")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_historical_dataset(
    directory: Path,
    markets: Sequence[HistoricalMarket],
    candles: Sequence[HistoricalCandle],
) -> HistoricalDatasetManifest:
    if not markets or not candles:
        raise ValueError("historical dataset cannot be empty")
    directory.mkdir(parents=True, exist_ok=True)
    market_rows = [{
        **asdict(item),
        "target_price": str(item.target_price),
    } for item in markets]
    candle_rows = [asdict(item) for item in candles]
    identity = json.dumps(
        {"markets": market_rows, "candles": candle_rows},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    version = hashlib.sha256(identity.encode()).hexdigest()
    _atomic_parquet(directory / "markets.parquet", market_rows)
    _atomic_parquet(directory / "candles.parquet", candle_rows)
    manifest = HistoricalDatasetManifest(
        dataset_version=version,
        source="kalshi_public_one_minute_candles",
        asset=markets[0].asset,
        series_ticker=markets[0].series_ticker,
        start_time=min(item.close_time for item in markets).isoformat(),
        end_time=max(item.close_time for item in markets).isoformat(),
        market_count=len(markets),
        candle_count=len(candles),
        created_at=datetime.now(UTC).isoformat(),
    )
    (directory / "manifest.json").write_text(
        json.dumps(asdict(manifest), sort_keys=True, indent=2), encoding="utf-8"
    )
    return manifest


def load_historical_dataset(
    directory: Path,
) -> tuple[list[HistoricalMarket], list[HistoricalCandle], HistoricalDatasetManifest]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("historical pretraining requires: pip install -e '.[research]'") from error
    manifest = HistoricalDatasetManifest(**json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    ))
    markets = [HistoricalMarket(
        ticker=str(row["ticker"]), asset=str(row["asset"]),
        series_ticker=str(row["series_ticker"]), open_time=row["open_time"],
        close_time=row["close_time"], target_price=Decimal(str(row["target_price"])),
        label_yes=int(row["label_yes"]),
    ) for row in pq.read_table(directory / "markets.parquet").to_pylist()]
    candles = [HistoricalCandle(**row) for row in (
        pq.read_table(directory / "candles.parquet").to_pylist()
    )]
    return markets, candles, manifest


def build_historical_training_rows(
    markets: Sequence[HistoricalMarket],
    candles: Sequence[HistoricalCandle],
    *,
    maximum_seconds_remaining: int = 390,
    minimum_seconds_remaining: int = 8,
) -> list[TrainingRow]:
    lookup = {item.ticker: item for item in markets}
    grouped: dict[str, list[HistoricalCandle]] = {}
    for candle in candles:
        if candle.ticker in lookup:
            grouped.setdefault(candle.ticker, []).append(candle)
    rows: list[TrainingRow] = []
    for ticker, sequence in grouped.items():
        market = lookup[ticker]
        sequence.sort(key=lambda item: item.end_time)
        mids: list[float] = []
        volumes: list[float] = []
        for candle in sequence:
            seconds_remaining = int((market.close_time - candle.end_time).total_seconds())
            mid = (candle.yes_bid_close + candle.yes_ask_close) / 2.0
            mids.append(mid)
            volumes.append(candle.volume)
            if not minimum_seconds_remaining < seconds_remaining <= maximum_seconds_remaining:
                continue
            trade_close = candle.trade_close if candle.trade_close is not None else mid
            trade_mean = candle.trade_mean if candle.trade_mean is not None else trade_close
            trade_range = (
                candle.trade_high - candle.trade_low
                if candle.trade_high is not None and candle.trade_low is not None else 0.0
            )
            last_three = mids[-3:]
            last_five = mids[-5:]
            volume_mean = sum(volumes[-3:]) / len(volumes[-3:])
            features = {
                "seconds_remaining": float(seconds_remaining),
                "market_yes_mid": mid,
                "yes_bid_close": candle.yes_bid_close,
                "yes_ask_close": candle.yes_ask_close,
                "spread_close": candle.yes_ask_close - candle.yes_bid_close,
                "trade_close": trade_close,
                "trade_mean": trade_mean,
                "log_volume": math.log1p(max(0.0, candle.volume)),
                "log_open_interest": math.log1p(max(0.0, candle.open_interest)),
                "bid_range": candle.yes_bid_high - candle.yes_bid_low,
                "ask_range": candle.yes_ask_high - candle.yes_ask_low,
                "trade_range": trade_range,
                "mid_momentum_1m": mid - mids[-2] if len(mids) >= 2 else 0.0,
                "mid_momentum_3m": mid - mids[-4] if len(mids) >= 4 else 0.0,
                "mid_volatility_3m": float(np.std(last_three)) if len(last_three) > 1 else 0.0,
                "mid_volatility_5m": float(np.std(last_five)) if len(last_five) > 1 else 0.0,
                "relative_volume_3m": candle.volume / volume_mean if volume_mean > 0 else 0.0,
            }
            rows.append(TrainingRow(
                ticker=ticker, market_close_time=market.close_time,
                observed_at=candle.end_time, label_yes=market.label_yes,
                features=features,
            ))
    return rows


def _probability_metrics(rows: Sequence[TrainingRow], probabilities: Sequence[float]) -> dict[str, float | int]:
    labels = np.asarray([row.label_yes for row in rows], dtype=np.float64)
    predicted = np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
    weights = ticker_balanced_weights(rows)
    weights /= weights.sum()
    return {
        "market_count": len({row.ticker for row in rows}),
        "observation_count": len(rows),
        "yes_rate": float(np.sum(weights * labels)),
        "brier": float(np.sum(weights * (predicted - labels) ** 2)),
        "log_loss": float(-np.sum(weights * (
            labels * np.log(predicted) + (1 - labels) * np.log(1 - predicted)
        ))),
        "accuracy": float(np.sum(weights * ((predicted >= 0.5) == labels))),
    }


def train_historical_model(
    dataset_directory: Path,
    artifact_directory: Path,
) -> dict[str, Any]:
    markets, candles, manifest = load_historical_dataset(dataset_directory)
    rows = build_historical_training_rows(markets, candles)
    if len({row.ticker for row in rows}) < 20:
        raise ValueError("historical pretraining requires at least 20 represented markets")
    split = chronological_ticker_split(rows, embargo_tickers=2)
    estimator = XGBoostTrainer(
        XGBoostConfig(
            max_depth=4, min_child_weight=8, learning_rate=0.04,
            rounds=500, early_stopping_rounds=40, calibration_bins=12,
        ),
        feature_names=HISTORICAL_FEATURE_NAMES,
    ).train(rows, split, artifact_directory)
    test_set = set(split.test_tickers)
    test_rows = [row for row in rows if row.ticker in test_set]
    model_probabilities = [
        float(estimator.predict_features(row.features)[0]) for row in test_rows
    ]
    market_probabilities = [row.features["market_yes_mid"] for row in test_rows]
    report = {
        "purpose": "historical_pretraining_research_only",
        "warning": "Not approved for Paper or Live; final test is reporting only.",
        "dataset": asdict(manifest),
        "feature_names": HISTORICAL_FEATURE_NAMES,
        "split": asdict(split),
        "xgboost_test": _probability_metrics(test_rows, model_probabilities),
        "market_mid_test": _probability_metrics(test_rows, market_probabilities),
        "artifact_directory": str(artifact_directory),
        "paper_ready": False,
        "live_ready": False,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "historical_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    (artifact_directory / "manifest.json").write_text(json.dumps({
        "model_id": artifact_directory.name,
        "model_type": "historical_xgboost",
        "model_version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "asset": manifest.asset,
        "dataset_version": manifest.dataset_version,
        "feature_schema_version": "historical-candle-v1",
        "purpose": "research",
        "status": "completed",
        "paper_ready": False,
        "live_ready": False,
        "files": {
            "model": "model.ubj", "metadata": "metadata.json",
            "report": "historical_report.json",
        },
    }, sort_keys=True, indent=2), encoding="utf-8")
    return report


def backfill_and_train(
    *,
    asset: str,
    series_ticker: str,
    start: datetime,
    end: datetime,
    dataset_directory: Path,
    artifact_directory: Path,
    maximum_markets: int | None = None,
    client: HistoricalKalshiClient | None = None,
) -> dict[str, Any]:
    owned = client is None
    source = client or HistoricalKalshiClient.production()
    try:
        markets = source.settled_markets(
            asset=asset, series_ticker=series_ticker, start=start, end=end,
            maximum_markets=maximum_markets,
        )
        candles = source.candles(markets)
    finally:
        if owned:
            source.close()
    save_historical_dataset(dataset_directory, markets, candles)
    return train_historical_model(dataset_directory, artifact_directory)


def _utc_argument(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamps must include a timezone")
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill and pretrain historical XGBoost")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--series", default="KXBTC15M")
    parser.add_argument("--start", type=_utc_argument, required=True)
    parser.add_argument("--end", type=_utc_argument, required=True)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--maximum-markets", type=int)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must follow --start")
    report = backfill_and_train(
        asset=args.asset.upper(), series_ticker=args.series,
        start=args.start, end=args.end,
        dataset_directory=args.dataset_directory,
        artifact_directory=args.artifact_directory,
        maximum_markets=args.maximum_markets,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
