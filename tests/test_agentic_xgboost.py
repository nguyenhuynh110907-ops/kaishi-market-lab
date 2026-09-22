from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from kaishi_bot.agentic_strategy import (
    DEFAULT_FEATURE_NAMES,
    HistogramCalibrator,
    MarketObservation,
    TrainingRow,
    XGBoostConfig,
    XGBoostProbabilityEstimator,
    XGBoostTrainer,
    chronological_ticker_split,
    observation_features,
    ticker_balanced_weights,
)


def row(ticker: str, day: int, label: int, value: float = 1.0) -> TrainingRow:
    close = datetime(2026, 8, day, 12, tzinfo=UTC)
    return TrainingRow(
        ticker=ticker,
        market_close_time=close,
        observed_at=close - timedelta(minutes=2),
        label_yes=label,
        features={"value": value},
    )


def test_chronological_split_keeps_whole_tickers_and_embargoes_boundaries() -> None:
    rows = [row(f"T{day}", day, day % 2) for day in range(1, 11)]
    rows += [row("T1", 1, 1, 2.0), row("T2", 2, 0, 2.0)]

    split = chronological_ticker_split(
        rows, train_ratio=0.5, validation_ratio=0.3, embargo_tickers=1
    )

    assert split.train_tickers == ("T1", "T2", "T3", "T4", "T5")
    assert split.validation_tickers == ("T7", "T8")
    assert split.test_tickers == ("T10",)
    assert "T6" not in split.train_tickers + split.validation_tickers + split.test_tickers
    assert "T9" not in split.train_tickers + split.validation_tickers + split.test_tickers


def test_ticker_balanced_weights_give_each_market_equal_total_weight() -> None:
    rows = [row("A", 1, 1) for _ in range(3)] + [row("B", 2, 0)]

    weights = ticker_balanced_weights(rows)

    assert np.isclose(weights[:3].sum(), 1.0)
    assert np.isclose(weights[3:].sum(), 1.0)


def test_histogram_calibration_uses_validation_outcomes_and_smoothing() -> None:
    calibrator = HistogramCalibrator.fit(
        [0.05, 0.10, 0.85, 0.95], [0, 0, 1, 1], bin_count=2
    )

    low, _ = calibrator.predict(0.1)
    high, _ = calibrator.predict(0.9)

    assert 0 < low < 0.5
    assert 0.5 < high < 1


def test_histogram_calibration_can_balance_repeated_market_rows() -> None:
    calibrator = HistogramCalibrator.fit(
        [0.7, 0.7, 0.7, 0.7],
        [1, 1, 1, 0],
        bin_count=2,
        sample_weights=[1 / 3, 1 / 3, 1 / 3, 1],
    )

    calibrated, _ = calibrator.predict(0.7)

    assert calibrated == pytest.approx(0.5)


def test_observation_features_include_settlement_state() -> None:
    observed = datetime.now(UTC)
    item = MarketObservation(
        ticker="BTC", observed_at=observed, seconds_remaining=30,
        up_bid=Decimal("0.70"), up_ask=Decimal("0.72"),
        down_bid=Decimal("0.28"), down_ask=Decimal("0.30"),
        target_price=Decimal("65000"), brti_price=Decimal("65020"),
        brti_sigma_per_sqrt_second=Decimal("2"), locked_sample_count=30,
        locked_sample_sum=Decimal("1949700"),
    )

    features = observation_features(item)

    assert features["is_final_minute"] == 1.0
    assert features["locked_sample_ratio"] == 0.5
    assert features["required_average_distance_ratio"] == pytest.approx(10 / 65000)


def test_training_row_rejects_post_close_observation() -> None:
    close = datetime.now(UTC)
    with pytest.raises(ValueError):
        TrainingRow("BTC", close, close + timedelta(seconds=1), 1, {"x": 1.0})


def test_xgboost_trains_saves_and_loads_at_best_iteration(tmp_path) -> None:
    pytest.importorskip("xgboost")
    rows = []
    for day in range(1, 13):
        label = day % 2
        for tick in range(3):
            features = {
                name: float(label) + tick * 0.01
                for name in DEFAULT_FEATURE_NAMES
            }
            rows.append(TrainingRow(
                ticker=f"T{day}",
                market_close_time=datetime(2026, 8, day, 12, tzinfo=UTC),
                observed_at=datetime(2026, 8, day, 11, 58, tick, tzinfo=UTC),
                label_yes=label,
                features=features,
            ))
    split = chronological_ticker_split(
        rows,
        train_ratio=0.5,
        validation_ratio=0.25,
        embargo_tickers=1,
    )
    estimator = XGBoostTrainer(XGBoostConfig(
        rounds=12,
        early_stopping_rounds=3,
        calibration_bins=4,
    )).train(rows, split, tmp_path)
    loaded = XGBoostProbabilityEstimator.load(tmp_path)

    assert (tmp_path / "model.ubj").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert loaded.inference_rounds == estimator.inference_rounds
    assert 1 <= loaded.inference_rounds <= 12
