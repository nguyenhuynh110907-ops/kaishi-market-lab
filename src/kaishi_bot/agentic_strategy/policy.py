from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Protocol

from kaishi_bot.agentic_strategy.models import (
    Action,
    AgentConfig,
    Decision,
    MarketObservation,
    ProbabilityEstimate,
)
from kaishi_bot.agentic_strategy.probability import SettlementProbabilityModel
from kaishi_bot.domain import Side


class ProbabilityEstimator(Protocol):
    def estimate(self, observation: MarketObservation) -> ProbabilityEstimate:
        """Estimate calibrated YES/NO settlement probabilities."""


class AgenticPolicy:
    """Settlement-aware policy with explicit entry and risk gates."""

    def __init__(
        self,
        estimator: ProbabilityEstimator | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.estimator = estimator or SettlementProbabilityModel(self.config)
        self._confirmations: dict[tuple[str, Side], int] = defaultdict(int)

    @staticmethod
    def _decision(
        action: Action,
        *,
        side: Side | None = None,
        fraction: Decimal = Decimal("0"),
        probability: Decimal = Decimal("0"),
        edge: Decimal = Decimal("0"),
        reason: str,
    ) -> Decision:
        return Decision(action, side, fraction, probability, edge, reason)

    def _reset_confirmations(self, ticker: str, except_side: Side | None = None) -> None:
        for key in tuple(self._confirmations):
            if key[0] == ticker and key[1] is not except_side:
                self._confirmations.pop(key, None)

    def decide(self, observation: MarketObservation) -> Decision:
        estimate = self.estimator.estimate(observation)
        if observation.data_stale or observation.has_gap:
            self._reset_confirmations(observation.ticker)
            action = Action.HOLD if observation.position else Action.WAIT
            return self._decision(action, reason="data_quality_block")
        if observation.position is not None:
            self._reset_confirmations(observation.ticker)
            return self._manage_position(observation, estimate)
        return self._consider_entry(observation, estimate)

    def _consider_entry(
        self, observation: MarketObservation, estimate: ProbabilityEstimate
    ) -> Decision:
        if observation.seconds_remaining <= self.config.no_entry_last_seconds:
            self._reset_confirmations(observation.ticker)
            return self._decision(Action.WAIT, reason="entry_cutoff")

        side = Side.UP if estimate.yes >= estimate.no else Side.DOWN
        probability = estimate.for_side(side)
        ask = observation.ask(side)
        edge = probability - ask - observation.entry_fee(side)
        probability_floor = (
            self.config.final_entry_probability
            if observation.is_final_minute
            else self.config.entry_probability
        )
        edge_floor = (
            self.config.final_entry_edge
            if observation.is_final_minute
            else self.config.entry_edge
        )
        eligible = (
            self.config.entry_min <= ask <= self.config.entry_max
            and probability >= probability_floor
            and edge >= edge_floor
        )
        if not eligible:
            self._reset_confirmations(observation.ticker)
            return self._decision(
                Action.WAIT,
                side=side,
                probability=probability,
                edge=edge,
                reason="entry_gate",
            )

        self._reset_confirmations(observation.ticker, except_side=side)
        key = (observation.ticker, side)
        self._confirmations[key] += 1
        if self._confirmations[key] < self.config.confirmation_ticks:
            return self._decision(
                Action.WAIT,
                side=side,
                probability=probability,
                edge=edge,
                reason="confirmation_pending",
            )

        self._confirmations[key] = 0
        fraction = (
            self.config.final_minute_fraction
            if observation.is_final_minute
            else self.config.exploratory_fraction
        )
        return self._decision(
            Action.BUY,
            side=side,
            fraction=fraction,
            probability=probability,
            edge=edge,
            reason="positive_settlement_edge",
        )

    def _manage_position(
        self, observation: MarketObservation, estimate: ProbabilityEstimate
    ) -> Decision:
        assert observation.position is not None
        position = observation.position
        side = position.side
        probability = estimate.for_side(side)
        bid = observation.bid(side)
        edge = probability - bid

        if bid >= self.config.final_take_profit:
            if (
                observation.is_final_minute
                and probability >= self.config.hold_to_settlement_probability
            ):
                return self._decision(
                    Action.HOLD,
                    side=side,
                    probability=probability,
                    edge=edge,
                    reason="settlement_value_exceeds_exit",
                )
            return self._decision(
                Action.EXIT_ALL,
                side=side,
                fraction=Decimal("1"),
                probability=probability,
                edge=edge,
                reason="final_take_profit",
            )

        if (
            bid >= self.config.first_take_profit
            and not position.partial_exit_taken
        ):
            return self._decision(
                Action.EXIT_HALF,
                side=side,
                fraction=Decimal("0.5"),
                probability=probability,
                edge=edge,
                reason="first_take_profit",
            )

        if probability < self.config.exit_probability:
            return self._decision(
                Action.EXIT_ALL,
                side=side,
                fraction=Decimal("1"),
                probability=probability,
                edge=edge,
                reason="thesis_invalidated",
            )

        if (
            bid <= self.config.emergency_bid
            and probability < self.config.emergency_probability
        ):
            return self._decision(
                Action.EXIT_ALL,
                side=side,
                fraction=Decimal("1"),
                probability=probability,
                edge=edge,
                reason="emergency_stop",
            )

        if (
            observation.is_final_minute
            and probability >= self.config.add_probability
            and edge >= self.config.final_entry_edge
            and position.add_count < self.config.max_add_count
        ):
            return self._decision(
                Action.ADD,
                side=side,
                fraction=self.config.add_fraction,
                probability=probability,
                edge=edge,
                reason="locked_samples_improved_edge",
            )

        return self._decision(
            Action.HOLD,
            side=side,
            probability=probability,
            edge=edge,
            reason="position_remains_valid",
        )
