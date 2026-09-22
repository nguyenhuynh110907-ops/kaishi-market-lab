from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kaishi_bot.agentic_strategy import (
    Action,
    Decision,
    HybridProbabilityEstimator,
    MarketObservation,
    MonteCarloConfig,
    MonteCarloGuardConfig,
    MonteCarloGuardedPolicy,
    MonteCarloProbabilityEstimator,
    PathRiskEstimate,
    ProbabilityEstimate,
)
from kaishi_bot.domain import Side


def observation(**changes) -> MarketObservation:
    values = {
        "ticker": "KXBTC15M-MC",
        "observed_at": datetime(2026, 8, 11, 12, tzinfo=UTC),
        "seconds_remaining": 30,
        "up_bid": Decimal("0.70"),
        "up_ask": Decimal("0.72"),
        "down_bid": Decimal("0.28"),
        "down_ask": Decimal("0.30"),
        "target_price": Decimal("65000"),
        "brti_price": Decimal("65020"),
        "brti_sigma_per_sqrt_second": Decimal("2"),
        "locked_sample_count": 30,
        "locked_sample_sum": Decimal("1949700"),
    }
    values.update(changes)
    return MarketObservation(**values)


class FixedEstimator:
    def __init__(self, yes: str, uncertainty: str = "0.01") -> None:
        self.yes = Decimal(yes)
        self.uncertainty = Decimal(uncertainty)

    def estimate(self, item: MarketObservation) -> ProbabilityEstimate:
        return ProbabilityEstimate(
            yes=self.yes,
            no=Decimal("1") - self.yes,
            uncertainty=self.uncertainty,
            required_remaining_average=item.required_remaining_average(),
        )


def test_monte_carlo_is_reproducible_and_responds_to_brti_distance() -> None:
    model = MonteCarloProbabilityEstimator(MonteCarloConfig(paths=2048, seed=7))

    low = model.estimate(observation(brti_price=Decimal("64980"))).yes
    high_item = observation(brti_price=Decimal("65020"))
    high = model.estimate(high_item).yes

    assert high > low
    assert model.estimate(high_item).yes == high


def test_path_risk_partitions_first_passage_outcomes() -> None:
    model = MonteCarloProbabilityEstimator(MonteCarloConfig(paths=512, seed=3))

    risk = model.assess_path_risk(observation(), Side.UP)

    assert risk.path_count == 512
    assert risk.stop_before_take_profit_probability + risk.take_profit_before_stop_probability + risk.neither_barrier_probability == pytest.approx(1.0)


def test_hybrid_keeps_xgboost_monte_carlo_and_market_as_explicit_components() -> None:
    model = HybridProbabilityEstimator(
        FixedEstimator("0.80"),
        FixedEstimator("0.60"),  # type: ignore[arg-type]
        xgboost_weight=0.5,
        monte_carlo_weight=0.3,
        market_weight=0.2,
    )

    estimate = model.estimate(observation())

    assert estimate.yes == Decimal("0.722")


class BuyPolicy:
    def decide(self, item: MarketObservation) -> Decision:
        return Decision(
            Action.BUY, Side.UP, Decimal("0.25"), Decimal("0.90"),
            Decimal("0.18"), "positive_settlement_edge",
        )


class HighStopRiskMonteCarlo(MonteCarloProbabilityEstimator):
    def assess_path_risk(self, item, side, **kwargs) -> PathRiskEstimate:
        return PathRiskEstimate(side, 0.8, 0.7, 0.1, 0.2, 1000)


def test_guard_blocks_entry_when_stop_is_likely_to_arrive_before_target() -> None:
    policy = MonteCarloGuardedPolicy(
        BuyPolicy(),
        HighStopRiskMonteCarlo(MonteCarloConfig(paths=128)),
        MonteCarloGuardConfig(maximum_stop_before_take_profit=0.35),
    )

    decision = policy.decide(observation())

    assert decision.action is Action.WAIT
    assert decision.reason == "monte_carlo_stop_risk"
    assert policy.last_risk is not None
    assert policy.last_risk.stop_before_take_profit_probability == 0.7
    assert policy.risk_evaluation_count == 1


def test_guard_clears_entry_risk_on_a_non_entry_decision() -> None:
    class WaitPolicy:
        def decide(self, item: MarketObservation) -> Decision:
            return Decision(
                Action.WAIT, Side.UP, Decimal("0"), Decimal("0.50"),
                Decimal("0"), "entry_gate",
            )

    policy = MonteCarloGuardedPolicy(
        BuyPolicy(),
        HighStopRiskMonteCarlo(MonteCarloConfig(paths=128)),
    )
    policy.decide(observation())
    assert policy.last_risk is not None

    policy.base_policy = WaitPolicy()
    policy.decide(observation(observed_at=datetime(2026, 8, 11, 12, 0, 1, tzinfo=UTC)))

    assert policy.last_risk is None
    assert policy.risk_evaluation_count == 1
