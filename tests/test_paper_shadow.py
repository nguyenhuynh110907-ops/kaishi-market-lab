from datetime import UTC, datetime
from decimal import Decimal

from kaishi_bot.agentic_strategy import (
    AgentConfig,
    AgenticPolicy,
    PositionState,
    ProbabilityEstimate,
)
from kaishi_bot.domain import Side
from kaishi_bot.fees import FeeSchedule
from kaishi_bot.paper_shadow import PaperShadowConfig, PaperShadowRunner
from kaishi_bot.research_features import FeatureObservation
from kaishi_bot.research_store import ResearchStore


def feature(**changes):
    now = datetime(2026, 8, 11, 12, tzinfo=UTC)
    values = dict(
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
    values.update(changes)
    return FeatureObservation(**values)


class Estimator:
    def estimate(self, observation):
        return ProbabilityEstimate(
            Decimal("0.95"), Decimal("0.05"), Decimal("0.01"),
            observation.required_remaining_average(),
        )


class Broker:
    def __init__(self):
        self.decisions = []

    def apply_shadow_decision(self, decision, observation):
        self.decisions.append((decision, observation))
        return {"predicted_fill": "paper-only"}


def runner():
    broker = Broker()
    return PaperShadowRunner(
        AgenticPolicy(Estimator(), AgentConfig(confirmation_ticks=1)), broker,
        PaperShadowConfig("policy-v1", "model-v1"),
    ), broker


def test_shadow_executes_only_through_paper_broker_and_logs_versions() -> None:
    shadow, broker = runner()
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "f")
    result = shadow.on_feature(feature(), schedule)
    assert result.policy_version == "policy-v1"
    assert result.model_version == "model-v1"
    assert result.selected_action == "buy"
    assert len(broker.decisions) == 1


def test_stale_gap_waits_flat_and_protectively_exits_open_paper_position() -> None:
    shadow, broker = runner()
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "f")
    stale = feature(is_stale=True, feature_complete=True)
    flat = shadow.on_feature(stale, schedule)
    assert flat.selected_action == "wait"
    position = PositionState(Side.UP, Decimal("1"), Decimal("0.70"))
    held = shadow.on_feature(stale, schedule, position)
    assert held.selected_action == "exit_all"
    assert len(broker.decisions) == 1


def test_kill_switch_blocks_new_shadow_trade() -> None:
    shadow, broker = runner()
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "f")
    shadow.kill()
    result = shadow.on_feature(feature(), schedule)
    assert result.selected_action == "wait"
    assert result.kill_switch_active is True
    assert broker.decisions == []


def test_shadow_decision_can_be_persisted_to_research_control_plane(tmp_path) -> None:
    store = ResearchStore(tmp_path / "control.sqlite3")
    broker = Broker()
    shadow = PaperShadowRunner(
        AgenticPolicy(Estimator(), AgentConfig(confirmation_ticks=1)), broker,
        PaperShadowConfig("policy-v1", "model-v1"),
        decision_sink=store.save_shadow_decision,
    )
    try:
        schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "f")
        shadow.on_feature(feature(), schedule)
        row = store.connection.execute(
            "SELECT * FROM research_paper_shadow_decisions"
        ).fetchone()
        assert row["selected_action"] == "buy"
        assert row["policy_version"] == "policy-v1"
    finally:
        store.close()
