from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kaishi_bot.agentic_strategy import (
    Action,
    AgentConfig,
    AgenticPolicy,
    MarketObservation,
    PositionState,
    ProbabilityEstimate,
    SelectionConfig,
    SettlementProbabilityModel,
    TrialMetrics,
    select_robust_top_k,
    successive_halving,
)
from kaishi_bot.domain import Side


class FixedEstimator:
    def __init__(self, yes: str) -> None:
        self.yes = Decimal(yes)

    def estimate(self, observation: MarketObservation) -> ProbabilityEstimate:
        return ProbabilityEstimate(
            yes=self.yes,
            no=Decimal("1") - self.yes,
            uncertainty=Decimal("1"),
            required_remaining_average=observation.required_remaining_average(),
        )


def observation(**changes) -> MarketObservation:
    values = {
        "ticker": "KXBTC15M-TEST",
        "observed_at": datetime.now(UTC),
        "seconds_remaining": 120,
        "up_bid": Decimal("0.71"),
        "up_ask": Decimal("0.72"),
        "down_bid": Decimal("0.28"),
        "down_ask": Decimal("0.29"),
        "target_price": Decimal("65000"),
        "brti_price": Decimal("65020"),
        "brti_sigma_per_sqrt_second": Decimal("2"),
    }
    values.update(changes)
    return MarketObservation(**values)


def test_required_remaining_average_uses_locked_samples() -> None:
    item = observation(
        seconds_remaining=30,
        locked_sample_count=30,
        locked_sample_sum=Decimal("1949700"),
    )

    assert item.required_remaining_average() == Decimal("65010")


def test_probability_increases_when_brti_moves_above_target() -> None:
    model = SettlementProbabilityModel()

    low = model.estimate(observation(brti_price=Decimal("64980"))).yes
    high = model.estimate(observation(brti_price=Decimal("65020"))).yes

    assert high > low


def test_entry_requires_consecutive_confirmations() -> None:
    policy = AgenticPolicy(
        FixedEstimator("0.90"), AgentConfig(confirmation_ticks=3)
    )
    item = observation()

    assert policy.decide(item).action is Action.WAIT
    assert policy.decide(item).action is Action.WAIT
    decision = policy.decide(item)

    assert decision.action is Action.BUY
    assert decision.side is Side.UP
    assert decision.fraction == Decimal("0.25")


def test_low_bid_alone_does_not_trigger_emergency_exit() -> None:
    position = PositionState(Side.UP, Decimal("5"), Decimal("0.72"))
    policy = AgenticPolicy(FixedEstimator("0.80"))

    decision = policy.decide(
        observation(up_bid=Decimal("0.40"), up_ask=Decimal("0.41"), position=position)
    )

    assert decision.action is Action.HOLD


def test_probability_invalidation_exits_before_price_stop() -> None:
    position = PositionState(Side.UP, Decimal("5"), Decimal("0.72"))
    policy = AgenticPolicy(FixedEstimator("0.55"))

    decision = policy.decide(observation(position=position))

    assert decision.action is Action.EXIT_ALL
    assert decision.reason == "thesis_invalidated"


def test_stale_data_blocks_new_entry() -> None:
    policy = AgenticPolicy(FixedEstimator("0.95"), AgentConfig(confirmation_ticks=1))

    decision = policy.decide(observation(data_stale=True))

    assert decision.action is Action.WAIT
    assert decision.reason == "data_quality_block"


def trial(
    trial_id: str,
    *,
    returns=(0.05, 0.04, 0.03, 0.02),
    pnl=(1.0, -0.5, 0.8, 0.2),
    trades=400,
    drawdown=0.08,
) -> TrialMetrics:
    return TrialMetrics(
        trial_id=trial_id,
        fold_returns=returns,
        sharpe=1.5,
        max_drawdown=drawdown,
        cvar=-0.02,
        fee_stress_return=0.02,
        latency_stress_return=0.01,
        closed_trades=trades,
        pnl_series=pnl,
    )


def test_tournament_filters_fragile_trials_and_keeps_diversity() -> None:
    strong = trial("strong")
    correlated = trial("correlated", returns=(0.06, 0.05, 0.04, 0.03), pnl=(2, -1, 1.6, 0.4))
    diverse = trial("diverse", returns=(0.04, 0.03, 0.02, 0.01), pnl=(-0.2, 0.9, -0.1, 0.8))
    too_small = trial("too-small", trades=20)

    selected = select_robust_top_k(
        [strong, correlated, diverse, too_small],
        SelectionConfig(top_k=2, maximum_pnl_correlation=0.85),
    )

    assert len(selected) == 2
    assert "too-small" not in {item.trial_id for item in selected}
    assert {item.trial_id for item in selected} & {"strong", "correlated"}
    assert "diverse" in {item.trial_id for item in selected}


def test_successive_halving_reduces_candidates_each_stage() -> None:
    seen: list[tuple[str, float]] = []

    def evaluate(candidate: str, fraction: float) -> float:
        seen.append((candidate, fraction))
        return float(candidate.removeprefix("c"))

    ranked = successive_halving(
        [f"c{index}" for index in range(10)],
        evaluate,
        resource_fractions=(0.2, 0.5, 1.0),
        retention_fraction=0.5,
    )

    assert ranked == ["c9", "c8", "c7"]
    assert len([item for item in seen if item[1] == 0.2]) == 10
    assert len([item for item in seen if item[1] == 0.5]) == 5
    assert len([item for item in seen if item[1] == 1.0]) == 3


def test_top_k_structurally_rejects_test_metrics() -> None:
    with pytest.raises(ValueError, match="validation metrics only"):
        select_robust_top_k([replace(trial("leaked"), split="test")])


@pytest.mark.parametrize("value", [-0.1, 0, 1.1])
def test_successive_halving_rejects_invalid_retention(value: float) -> None:
    with pytest.raises(ValueError):
        successive_halving(["a"], lambda *_: 1.0, retention_fraction=value)
