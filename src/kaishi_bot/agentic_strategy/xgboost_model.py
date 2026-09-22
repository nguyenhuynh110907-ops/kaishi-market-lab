from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kaishi_bot.agentic_strategy.models import MarketObservation, ProbabilityEstimate


DEFAULT_FEATURE_NAMES = (
    "seconds_remaining",
    "is_final_minute",
    "up_bid",
    "up_ask",
    "down_bid",
    "down_ask",
    "up_spread",
    "down_spread",
    "brti_target_distance_ratio",
    "distance_normalized_by_volatility",
    "locked_sample_ratio",
    "locked_average_distance_ratio",
    "required_average_distance_ratio",
    "entry_fee_up",
    "entry_fee_down",
)


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """One point-in-time row. All rows for a ticker share its final label."""

    ticker: str
    market_close_time: datetime
    observed_at: datetime
    label_yes: int
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("ticker is required")
        if self.label_yes not in {0, 1}:
            raise ValueError("label_yes must be zero or one")
        if self.market_close_time.utcoffset() is None or self.observed_at.utcoffset() is None:
            raise ValueError("training timestamps must be timezone-aware")
        if self.observed_at > self.market_close_time:
            raise ValueError("pre-settlement training row cannot follow market close")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train_tickers: tuple[str, ...]
    validation_tickers: tuple[str, ...]
    test_tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = [set(self.train_tickers), set(self.validation_tickers), set(self.test_tickers)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("ticker cannot occur in more than one split")


@dataclass(frozen=True, slots=True)
class XGBoostConfig:
    max_depth: int = 5
    min_child_weight: float = 8.0
    learning_rate: float = 0.03
    rounds: int = 800
    early_stopping_rounds: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    gamma: float = 0.0
    seed: int = 17
    calibration_bins: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_depth < 1 or self.rounds < 1 or self.early_stopping_rounds < 1:
            raise ValueError("invalid XGBoost iteration settings")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning rate must be in (0, 1]")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("sampling ratios must be in (0, 1]")
        if self.calibration_bins < 2:
            raise ValueError("at least two calibration bins are required")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("XGBoost device must be cpu, cuda, or cuda:<ordinal>")


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    probability: float
    count: int


class HistogramCalibrator:
    """Validation-only probability calibration without an sklearn dependency."""

    def __init__(self, bins: Sequence[CalibrationBin]) -> None:
        if not bins:
            raise ValueError("calibration bins cannot be empty")
        self.bins = tuple(bins)

    @classmethod
    def fit(
        cls,
        predictions: Sequence[float],
        labels: Sequence[int],
        bin_count: int = 20,
        sample_weights: Sequence[float] | None = None,
    ) -> "HistogramCalibrator":
        if len(predictions) != len(labels) or not predictions:
            raise ValueError("calibration predictions and labels must be non-empty and aligned")
        weights = list(sample_weights) if sample_weights is not None else [1.0] * len(labels)
        if len(weights) != len(labels) or any(weight <= 0 for weight in weights):
            raise ValueError("calibration weights must be positive and aligned")
        buckets: list[list[tuple[int, float]]] = [[] for _ in range(bin_count)]
        for prediction, label, weight in zip(predictions, labels, weights):
            if not 0 <= prediction <= 1 or label not in {0, 1} or not np.isfinite(weight):
                raise ValueError("invalid calibration value")
            index = min(bin_count - 1, int(prediction * bin_count))
            buckets[index].append((label, weight))
        total_weight = sum(weights)
        global_rate = (
            sum(label * weight for label, weight in zip(labels, weights)) + 1
        ) / (total_weight + 2)
        bins = []
        for index, bucket in enumerate(buckets):
            bucket_weight = sum(weight for _, weight in bucket)
            positives = sum(label * weight for label, weight in bucket)
            # A two-effective-sample prior prevents small market groups from
            # producing 0/1. Ticker-balanced weights stop dense tickers from
            # dominating calibration.
            calibrated = (
                (positives + 2 * global_rate) / (bucket_weight + 2)
                if bucket else global_rate
            )
            bins.append(CalibrationBin(
                lower=index / bin_count,
                upper=(index + 1) / bin_count,
                probability=calibrated,
                count=len(bucket),
            ))
        return cls(bins)

    def predict(self, raw_probability: float) -> tuple[float, float]:
        value = min(1.0, max(0.0, raw_probability))
        index = min(len(self.bins) - 1, int(value * len(self.bins)))
        bucket = self.bins[index]
        uncertainty = (
            (bucket.probability * (1.0 - bucket.probability) / bucket.count) ** 0.5
            if bucket.count else 0.5
        )
        return bucket.probability, uncertainty

    def to_json(self) -> list[dict[str, float | int]]:
        return [asdict(item) for item in self.bins]

    @classmethod
    def from_json(cls, payload: Sequence[Mapping[str, float | int]]) -> "HistogramCalibrator":
        return cls([CalibrationBin(**dict(item)) for item in payload])


def chronological_ticker_split(
    rows: Sequence[TrainingRow],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    embargo_tickers: int = 1,
) -> DatasetSplit:
    """Split whole tickers by close time; embargo tickers are excluded entirely."""

    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("split ratios must be in (0, 1)")
    if train_ratio + validation_ratio >= 1 or embargo_tickers < 0:
        raise ValueError("invalid split or embargo")
    closes: dict[str, datetime] = {}
    labels: dict[str, int] = {}
    for row in rows:
        previous_label = labels.setdefault(row.ticker, row.label_yes)
        if previous_label != row.label_yes:
            raise ValueError(f"inconsistent label for ticker {row.ticker}")
        previous_close = closes.setdefault(row.ticker, row.market_close_time)
        if previous_close != row.market_close_time:
            raise ValueError(f"inconsistent close time for ticker {row.ticker}")
    ordered = sorted(closes, key=lambda ticker: (closes[ticker], ticker))
    if len(ordered) < 3 + 2 * embargo_tickers:
        raise ValueError("not enough tickers for chronological train/validation/test split")
    train_end = max(1, int(len(ordered) * train_ratio))
    validation_end = max(train_end + 1, int(len(ordered) * (train_ratio + validation_ratio)))
    validation_start = train_end + embargo_tickers
    test_start = validation_end + embargo_tickers
    train = ordered[:train_end]
    validation = ordered[validation_start:validation_end]
    test = ordered[test_start:]
    if not train or not validation or not test:
        raise ValueError("split ratios and embargo produced an empty split")
    return DatasetSplit(tuple(train), tuple(validation), tuple(test))


def ticker_balanced_weights(rows: Sequence[TrainingRow]) -> np.ndarray:
    """Give every ticker equal total weight despite different observation counts."""

    counts = Counter(row.ticker for row in rows)
    if not counts:
        return np.array([], dtype=np.float64)
    return np.asarray([1.0 / counts[row.ticker] for row in rows], dtype=np.float64)


def observation_features(observation: MarketObservation) -> dict[str, float]:
    target = float(observation.target_price)
    sigma = max(float(observation.brti_sigma_per_sqrt_second), target * 1e-8)
    distance = float(observation.brti_price - observation.target_price)
    locked_average = (
        observation.locked_sample_sum / observation.locked_sample_count
        if observation.locked_sample_count else observation.brti_price
    )
    required = observation.required_remaining_average()
    return {
        "seconds_remaining": float(observation.seconds_remaining),
        "is_final_minute": float(observation.is_final_minute),
        "up_bid": float(observation.up_bid),
        "up_ask": float(observation.up_ask),
        "down_bid": float(observation.down_bid),
        "down_ask": float(observation.down_ask),
        "up_spread": float(observation.up_ask - observation.up_bid),
        "down_spread": float(observation.down_ask - observation.down_bid),
        "brti_target_distance_ratio": distance / target,
        "distance_normalized_by_volatility": distance / sigma,
        "locked_sample_ratio": observation.locked_sample_count / 60.0,
        "locked_average_distance_ratio": float(locked_average - observation.target_price) / target,
        "required_average_distance_ratio": (
            float(observation.brti_price - required) / target if required is not None else 0.0
        ),
        "entry_fee_up": float(observation.entry_fee_up),
        "entry_fee_down": float(observation.entry_fee_down),
    }


def _matrix(rows: Sequence[TrainingRow], feature_names: Sequence[str]) -> np.ndarray:
    missing = sorted({name for row in rows for name in feature_names if name not in row.features})
    if missing:
        raise ValueError(f"missing training features: {', '.join(missing)}")
    values = np.asarray(
        [[float(row.features[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("training features must be finite")
    return values


class XGBoostTrainer:
    def __init__(
        self,
        config: XGBoostConfig | None = None,
        feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    ) -> None:
        self.config = config or XGBoostConfig()
        self.feature_names = tuple(feature_names)
        if not self.feature_names or len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature names must be non-empty and unique")

    @staticmethod
    def _xgboost():
        try:
            import xgboost as xgb
        except ImportError as error:
            raise RuntimeError(
                "XGBoost training requires: pip install -e '.[ml]'"
            ) from error
        return xgb

    def train(
        self,
        rows: Sequence[TrainingRow],
        split: DatasetSplit,
        artifact_directory: Path,
    ) -> "XGBoostProbabilityEstimator":
        xgb = self._xgboost()
        lookup = {
            "train": set(split.train_tickers),
            "validation": set(split.validation_tickers),
            "test": set(split.test_tickers),
        }
        selected = {
            name: [row for row in rows if row.ticker in tickers]
            for name, tickers in lookup.items()
        }
        if any(not group for group in selected.values()):
            raise ValueError("all dataset splits must contain observations")
        train_rows, validation_rows = selected["train"], selected["validation"]
        train_matrix = _matrix(train_rows, self.feature_names)
        validation_matrix = _matrix(validation_rows, self.feature_names)
        dtrain = xgb.DMatrix(
            train_matrix,
            label=np.asarray([row.label_yes for row in train_rows]),
            weight=ticker_balanced_weights(train_rows),
            feature_names=list(self.feature_names),
        )
        dvalidation = xgb.DMatrix(
            validation_matrix,
            label=np.asarray([row.label_yes for row in validation_rows]),
            weight=ticker_balanced_weights(validation_rows),
            feature_names=list(self.feature_names),
        )
        params = {
            "objective": "binary:logistic",
            # XGBoost does not expose Brier score as a built-in early-stopping
            # metric. We stop on validation log loss and report Brier score in
            # the experiment layer from the held-out predictions.
            "eval_metric": "logloss",
            "max_depth": self.config.max_depth,
            "min_child_weight": self.config.min_child_weight,
            "eta": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "alpha": self.config.reg_alpha,
            "lambda": self.config.reg_lambda,
            "gamma": self.config.gamma,
            "seed": self.config.seed,
            "tree_method": "hist",
            "device": self.config.device,
        }
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.rounds,
            evals=[(dtrain, "train"), (dvalidation, "validation")],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False,
        )
        inference_rounds = booster.best_iteration + 1
        raw_validation = booster.predict(
            dvalidation,
            iteration_range=(0, inference_rounds),
        )
        calibrator = HistogramCalibrator.fit(
            raw_validation.tolist(),
            [row.label_yes for row in validation_rows],
            self.config.calibration_bins,
            ticker_balanced_weights(validation_rows).tolist(),
        )
        artifact_directory.mkdir(parents=True, exist_ok=True)
        model_path = artifact_directory / "model.ubj"
        metadata_path = artifact_directory / "metadata.json"
        booster.save_model(str(model_path))
        metadata_path.write_text(json.dumps({
            "artifact_version": 1,
            "feature_names": self.feature_names,
            "config": asdict(self.config),
            "split": asdict(split),
            "calibration": calibrator.to_json(),
            "inference_rounds": inference_rounds,
            "test_was_used_for_selection": False,
        }, sort_keys=True, indent=2), encoding="utf-8")
        return XGBoostProbabilityEstimator(
            booster,
            self.feature_names,
            calibrator,
            inference_rounds,
        )


class XGBoostProbabilityEstimator:
    """Adapter compatible with ``AgenticPolicy``'s probability protocol."""

    def __init__(
        self,
        booster,
        feature_names: Sequence[str],
        calibrator: HistogramCalibrator,
        inference_rounds: int,
    ) -> None:
        if inference_rounds < 1:
            raise ValueError("inference_rounds must be positive")
        self.booster = booster
        self.feature_names = tuple(feature_names)
        self.calibrator = calibrator
        self.inference_rounds = inference_rounds

    @classmethod
    def load(cls, artifact_directory: Path) -> "XGBoostProbabilityEstimator":
        xgb = XGBoostTrainer._xgboost()
        metadata = json.loads((artifact_directory / "metadata.json").read_text(encoding="utf-8"))
        booster = xgb.Booster()
        booster.load_model(str(artifact_directory / "model.ubj"))
        return cls(
            booster,
            metadata["feature_names"],
            HistogramCalibrator.from_json(metadata["calibration"]),
            metadata["inference_rounds"],
        )

    def estimate(self, observation: MarketObservation) -> ProbabilityEstimate:
        features = observation_features(observation)
        yes, uncertainty = self.predict_features(features)
        return ProbabilityEstimate(
            yes=yes,
            no=Decimal("1") - yes,
            uncertainty=uncertainty,
            required_remaining_average=observation.required_remaining_average(),
        )

    def predict_features(
        self, features: Mapping[str, float]
    ) -> tuple[Decimal, Decimal]:
        """Predict directly from the frozen feature schema for offline evaluation."""

        xgb = XGBoostTrainer._xgboost()
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"feature mapping cannot provide: {', '.join(missing)}")
        matrix = np.asarray([[features[name] for name in self.feature_names]], dtype=np.float32)
        raw = float(self.booster.predict(
            xgb.DMatrix(matrix, feature_names=list(self.feature_names)),
            iteration_range=(0, self.inference_rounds),
        )[0])
        yes_float, uncertainty_float = self.calibrator.predict(raw)
        yes = Decimal(str(round(yes_float, 8)))
        return yes, Decimal(str(round(uncertainty_float, 8)))
