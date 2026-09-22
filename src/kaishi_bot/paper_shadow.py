from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Protocol

from kaishi_bot.agentic_strategy import Action, AgenticPolicy, Decision, PositionState
from kaishi_bot.domain import Side
from kaishi_bot.fees import FeeSchedule
from kaishi_bot.research_features import (
    BuiltObservation,
    FeatureObservation,
    RejectedObservation,
    ResearchObservationAdapter,
)
from kaishi_bot.research_replay import ReplayAction


class PaperShadowBroker(Protocol):
    """Narrow Paper-only port; intentionally incompatible with ProductionGateway."""

    def apply_shadow_decision(
        self, decision: Decision, observation: BuiltObservation
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class PaperShadowConfig:
    policy_version: str
    model_version: str
    stale_position_action: Action = Action.EXIT_ALL

    def __post_init__(self) -> None:
        if self.stale_position_action not in {Action.HOLD, Action.EXIT_ALL}:
            raise ValueError("stale position behavior must be HOLD or EXIT_ALL")


@dataclass(frozen=True, slots=True)
class ShadowDecisionLog:
    ticker: str
    observed_at: datetime
    dataset_version: str
    policy_version: str
    model_version: str
    selected_action: str
    selected_side: str | None
    masked_actions: tuple[str, ...]
    reason: str
    quality_state: str
    broker_result: dict[str, object] | None
    kill_switch_active: bool


class PaperShadowRunner:
    def __init__(
        self, policy: AgenticPolicy, broker: PaperShadowBroker,
        config: PaperShadowConfig,
        adapter: ResearchObservationAdapter | None = None,
        decision_sink: Callable[[ShadowDecisionLog], None] | None = None,
    ) -> None:
        self.policy = policy
        self.broker = broker
        self.config = config
        self.adapter = adapter or ResearchObservationAdapter()
        self.decision_sink = decision_sink
        self.kill_switch_active = False
        self.logs: list[ShadowDecisionLog] = []

    def kill(self) -> None:
        self.kill_switch_active = True

    def enable(self) -> None:
        self.kill_switch_active = False

    def on_feature(
        self, feature: FeatureObservation, fee_schedule: FeeSchedule | None,
        position: PositionState | None = None,
    ) -> ShadowDecisionLog:
        built = self.adapter.build(
            feature, fee_schedule=fee_schedule, position=position
        )
        if isinstance(built, RejectedObservation):
            return self._record(
                feature, Action.WAIT, None, "observation_rejected:" + ",".join(built.reasons),
                self._all_trade_actions(), None, "rejected",
            )

        observation = built.observation
        if self.kill_switch_active:
            decision = Decision(
                Action.WAIT if position is None else Action.HOLD,
                position.side if position else None, Decimal("0"), Decimal("0"),
                Decimal("0"), "paper_shadow_kill_switch",
            )
        elif observation.data_stale or observation.has_gap:
            action = (
                Action.WAIT if position is None else self.config.stale_position_action
            )
            decision = Decision(
                action, position.side if position else None,
                Decimal("1") if action is Action.EXIT_ALL else Decimal("0"),
                Decimal("0"), Decimal("0"), "data_quality_protection",
            )
        else:
            decision = self.policy.decide(observation)

        masked = self._masked_actions(decision, position)
        broker_result = None
        if decision.action not in {Action.WAIT, Action.HOLD}:
            broker_result = self.broker.apply_shadow_decision(decision, built)
        return self._record(
            feature, decision.action, decision.side, decision.reason,
            masked, broker_result,
            "stale_or_gap" if observation.data_stale or observation.has_gap else "complete",
        )

    @staticmethod
    def _all_trade_actions() -> tuple[str, ...]:
        return tuple(action.value for action in ReplayAction if action is not ReplayAction.WAIT)

    @staticmethod
    def _masked_actions(
        decision: Decision, position: PositionState | None
    ) -> tuple[str, ...]:
        masked: set[str] = set()
        if position is None:
            masked |= {ReplayAction.HOLD.value, ReplayAction.ADD.value,
                       ReplayAction.EXIT_HALF.value, ReplayAction.EXIT_ALL.value}
        else:
            masked |= {ReplayAction.BUY_UP.value, ReplayAction.BUY_DOWN.value}
        return tuple(sorted(masked))

    def _record(
        self, feature: FeatureObservation, action: Action, side: Side | None,
        reason: str, masked: tuple[str, ...], broker_result: dict[str, object] | None,
        quality_state: str,
    ) -> ShadowDecisionLog:
        item = ShadowDecisionLog(
            ticker=feature.ticker, observed_at=feature.observation_time,
            dataset_version=feature.dataset_version,
            policy_version=self.config.policy_version,
            model_version=self.config.model_version,
            selected_action=action.value,
            selected_side=side.value if side else None,
            masked_actions=masked, reason=reason, quality_state=quality_state,
            broker_result=broker_result,
            kill_switch_active=self.kill_switch_active,
        )
        self.logs.append(item)
        if self.decision_sink is not None:
            self.decision_sink(item)
        return item
