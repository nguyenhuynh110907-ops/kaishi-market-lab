from datetime import UTC, datetime
from decimal import Decimal

from kaishi_bot.agentic_strategy import MarketObservation, PositionState
from kaishi_bot.domain import Side
from kaishi_bot.research_baselines import (
    PriceRuleBaseline,
    SettlementMathBaseline,
    mandatory_baseline_names,
)
from kaishi_bot.research_replay import ReplayAction


def observation(**changes):
    values = dict(
        ticker="BTC", observed_at=datetime.now(UTC), seconds_remaining=30,
        up_bid=Decimal("0.70"), up_ask=Decimal("0.72"),
        down_bid=Decimal("0.28"), down_ask=Decimal("0.30"),
        target_price=Decimal("100"), brti_price=Decimal("101"),
        brti_sigma_per_sqrt_second=Decimal("0.5"), locked_sample_count=30,
        locked_sample_sum=Decimal("2990"),
    )
    values.update(changes)
    return MarketObservation(**values)


def test_mandatory_catalog_contains_all_eight_baselines() -> None:
    assert len(mandatory_baseline_names()) == 8


def test_price_rule_uses_same_side_entry_tp_and_sl() -> None:
    baseline = PriceRuleBaseline()
    assert baseline.decide(observation()).action is ReplayAction.BUY_UP
    position = PositionState(Side.UP, Decimal("1"), Decimal("0.72"))
    assert baseline.decide(observation(
        position=position, up_bid=Decimal("0.94"), up_ask=Decimal("0.95")
    )).action is ReplayAction.EXIT_ALL


def test_settlement_math_enters_side_supported_by_required_average() -> None:
    decision = SettlementMathBaseline().decide(observation())
    assert decision.action is ReplayAction.BUY_UP
