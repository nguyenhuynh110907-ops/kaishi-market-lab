from kaishi_bot.agentic_strategy import (
    HistogramCalibrator,
    XGBoostProbabilityEstimator,
)


class FakeBooster:
    pass


def test_direct_feature_prediction_rejects_missing_frozen_features() -> None:
    estimator = XGBoostProbabilityEstimator(
        FakeBooster(), ("one", "two"),
        HistogramCalibrator.fit([0.5], [1], bin_count=2), 1,
    )

    try:
        estimator.predict_features({"one": 1.0})
    except ValueError as error:
        assert "two" in str(error)
    else:
        raise AssertionError("missing frozen feature was accepted")
