from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from kaishi_bot.agentic_strategy import Action, AgenticPolicy, MarketObservation
from kaishi_bot.domain import Side
from kaishi_bot.research_replay import ReplayAction


@dataclass(frozen=True, slots=True)
class BaselineDecision:
    action: ReplayAction
    reason: str


class ProbabilityEstimator(Protocol):
    def estimate(self, observation: MarketObservation): ...


class PriceRuleBaseline:
    def __init__(
        self, *, stop_loss: Decimal | None = Decimal("0.50"),
        take_profit: Decimal | None = Decimal("0.94"), hold_to_settlement: bool = False,
        time_dependent_stop: bool = False,
    ) -> None:
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.hold_to_settlement = hold_to_settlement
        self.time_dependent_stop = time_dependent_stop

    def decide(self, observation: MarketObservation) -> BaselineDecision:
        position = observation.position
        if observation.data_stale or observation.has_gap:
            return BaselineDecision(ReplayAction.WAIT, "data_quality_block")
        if position is None:
            if Decimal("0.68") <= observation.up_ask <= Decimal("0.77"):
                return BaselineDecision(ReplayAction.BUY_UP, "entry_band")
            if Decimal("0.68") <= observation.down_ask <= Decimal("0.77"):
                return BaselineDecision(ReplayAction.BUY_DOWN, "entry_band")
            return BaselineDecision(ReplayAction.WAIT, "outside_entry_band")
        bid = observation.bid(position.side)
        if self.take_profit is not None and bid >= self.take_profit:
            return BaselineDecision(ReplayAction.EXIT_ALL, "take_profit")
        stop = self.stop_loss
        if self.time_dependent_stop and stop is not None:
            # Tighten linearly from 50c to 65c during the final minute.
            elapsed = Decimal(max(0, 60 - observation.seconds_remaining)) / Decimal(60)
            stop = stop + Decimal("0.15") * elapsed
        if stop is not None and bid <= stop:
            return BaselineDecision(ReplayAction.EXIT_ALL, "stop_loss")
        if self.hold_to_settlement:
            return BaselineDecision(ReplayAction.HOLD, "hold_to_settlement")
        return BaselineDecision(ReplayAction.HOLD, "within_exit_band")


class SettlementMathBaseline:
    def decide(self, observation: MarketObservation) -> BaselineDecision:
        required = observation.required_remaining_average()
        if observation.data_stale or observation.has_gap or required is None:
            return BaselineDecision(ReplayAction.WAIT, "settlement_math_unavailable")
        side = Side.UP if observation.brti_price >= required else Side.DOWN
        if observation.position is None:
            return BaselineDecision(
                ReplayAction.BUY_UP if side is Side.UP else ReplayAction.BUY_DOWN,
                "required_average_edge",
            )
        if observation.position.side is not side:
            return BaselineDecision(ReplayAction.EXIT_ALL, "required_average_flipped")
        return BaselineDecision(ReplayAction.HOLD, "required_average_confirmed")


class ProbabilityThresholdBaseline:
    def __init__(
        self, estimator: ProbabilityEstimator, threshold: Decimal = Decimal("0.82")
    ) -> None:
        self.estimator = estimator
        self.threshold = threshold

    def decide(self, observation: MarketObservation) -> BaselineDecision:
        if observation.data_stale or observation.has_gap:
            return BaselineDecision(ReplayAction.WAIT, "data_quality_block")
        estimate = self.estimator.estimate(observation)
        side = Side.UP if estimate.yes >= estimate.no else Side.DOWN
        probability = estimate.for_side(side)
        if observation.position is None and probability >= self.threshold:
            return BaselineDecision(
                ReplayAction.BUY_UP if side is Side.UP else ReplayAction.BUY_DOWN,
                "probability_threshold",
            )
        if observation.position is not None and observation.position.side is not side:
            return BaselineDecision(ReplayAction.EXIT_ALL, "probability_flip")
        return BaselineDecision(
            ReplayAction.HOLD if observation.position else ReplayAction.WAIT,
            "probability_below_threshold",
        )


class AgenticPolicyBaseline:
    def __init__(self, policy: AgenticPolicy) -> None:
        self.policy = policy

    def decide(self, observation: MarketObservation) -> BaselineDecision:
        decision = self.policy.decide(observation)
        mapping = {
            Action.WAIT: ReplayAction.WAIT, Action.HOLD: ReplayAction.HOLD,
            Action.ADD: ReplayAction.ADD, Action.EXIT_HALF: ReplayAction.EXIT_HALF,
            Action.EXIT_ALL: ReplayAction.EXIT_ALL,
        }
        if decision.action is Action.BUY:
            action = ReplayAction.BUY_UP if decision.side is Side.UP else ReplayAction.BUY_DOWN
        else:
            action = mapping[decision.action]
        return BaselineDecision(action, decision.reason)


class ContextualBanditBaseline:
    """Small deterministic online contextual baseline, not an RL trainer."""

    def __init__(self, exploration: Decimal = Decimal("0.02")) -> None:
        self.exploration = exploration
        self.value = {Side.UP: Decimal("0"), Side.DOWN: Decimal("0")}
        self.count = {Side.UP: 0, Side.DOWN: 0}

    def update(self, side: Side, reward: Decimal) -> None:
        count = self.count[side] + 1
        self.value[side] += (reward - self.value[side]) / Decimal(count)
        self.count[side] = count

    def decide(self, observation: MarketObservation) -> BaselineDecision:
        if observation.data_stale or observation.has_gap:
            return BaselineDecision(ReplayAction.WAIT, "data_quality_block")
        context = (observation.brti_price - observation.target_price) / observation.target_price
        up_score = self.value[Side.UP] + context + self.exploration / Decimal(self.count[Side.UP] + 1)
        down_score = self.value[Side.DOWN] - context + self.exploration / Decimal(self.count[Side.DOWN] + 1)
        side = Side.UP if up_score >= down_score else Side.DOWN
        if observation.position is None:
            return BaselineDecision(
                ReplayAction.BUY_UP if side is Side.UP else ReplayAction.BUY_DOWN,
                "contextual_bandit",
            )
        if observation.position.side is not side:
            return BaselineDecision(ReplayAction.EXIT_ALL, "bandit_arm_flip")
        return BaselineDecision(ReplayAction.HOLD, "bandit_arm_hold")


def mandatory_baseline_names() -> tuple[str, ...]:
    return (
        "entry_68_77_sl50_tp94", "entry_68_77_hold_settlement",
        "time_dependent_sl", "settlement_math", "settlement_probability",
        "agentic_policy", "supervised_probability_threshold", "contextual_bandit",
    )
