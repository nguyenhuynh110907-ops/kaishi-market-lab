from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from math import exp, log, sqrt
from typing import Protocol

import numpy as np

from kaishi_bot.agentic_strategy.models import (
    Action,
    Decision,
    MarketObservation,
    ProbabilityEstimate,
)
from kaishi_bot.domain import Side


class ProbabilityEstimator(Protocol):
    def estimate(self, observation: MarketObservation) -> ProbabilityEstimate:
        """Estimate YES/NO settlement probabilities."""


class DecisionPolicy(Protocol):
    def decide(self, observation: MarketObservation) -> Decision:
        """Return the next trading decision."""


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    paths: int = 4096
    seed: int = 41
    drift_per_second: float = 0.0
    jump_probability_per_second: float = 0.002
    jump_sigma_multiplier: float = 5.0
    market_noise_sigma: float = 0.012
    market_noise_persistence: float = 0.85
    market_basis_half_life_seconds: float = 45.0

    def __post_init__(self) -> None:
        if self.paths < 128:
            raise ValueError("Monte Carlo requires at least 128 paths")
        if not 0 <= self.jump_probability_per_second <= 1:
            raise ValueError("jump probability must be between zero and one")
        if self.jump_sigma_multiplier < 0 or self.market_noise_sigma < 0:
            raise ValueError("Monte Carlo scale parameters cannot be negative")
        if not 0 <= self.market_noise_persistence < 1:
            raise ValueError("market noise persistence must be in [0, 1)")
        if self.market_basis_half_life_seconds <= 0:
            raise ValueError("market basis half-life must be positive")


@dataclass(frozen=True, slots=True)
class PathRiskEstimate:
    side: Side
    settlement_win_probability: float
    stop_before_take_profit_probability: float
    take_profit_before_stop_probability: float
    neither_barrier_probability: float
    path_count: int

    def __post_init__(self) -> None:
        probabilities = (
            self.settlement_win_probability,
            self.stop_before_take_profit_probability,
            self.take_profit_before_stop_probability,
            self.neither_barrier_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("path-risk probabilities must be between zero and one")
        barrier_total = sum(probabilities[1:])
        if abs(barrier_total - 1.0) > 1e-8:
            raise ValueError("barrier-event probabilities must sum to one")


@dataclass(frozen=True, slots=True)
class MonteCarloGuardConfig:
    stop_loss: Decimal = Decimal("0.50")
    take_profit: Decimal = Decimal("0.94")
    maximum_stop_before_take_profit: float = 0.35
    minimum_take_profit_before_stop: float = 0.20

    def __post_init__(self) -> None:
        if not Decimal("0") <= self.stop_loss < self.take_profit <= Decimal("1"):
            raise ValueError("Monte Carlo barriers must satisfy 0 <= stop < target <= 1")
        if not 0 <= self.maximum_stop_before_take_profit <= 1:
            raise ValueError("maximum stop probability must be between zero and one")
        if not 0 <= self.minimum_take_profit_before_stop <= 1:
            raise ValueError("minimum take-profit probability must be between zero and one")


@dataclass(slots=True)
class _Simulation:
    terminal_yes: np.ndarray
    yes_probability_paths: np.ndarray | None


def _future_average_variance_steps(sample_count: int) -> float:
    if sample_count <= 0:
        return 0.0
    count = float(sample_count)
    return ((count + 1.0) * (2.0 * count + 1.0)) / (6.0 * count)


def _normal_survival_approximation(threshold: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return (mean >= threshold).astype(np.float64)
    # Logistic approximation to Phi with low cost across thousands of paths.
    z = np.clip((mean - threshold) / std, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-1.702 * z))


class MonteCarloProbabilityEstimator:
    """Simulate BRTI paths and the exact 60-sample settlement average.

    The model is deliberately separate from XGBoost. It can serve as a
    standalone benchmark, a blend component, or a first-passage SL/TP guard.
    """

    def __init__(self, config: MonteCarloConfig | None = None) -> None:
        self.config = config or MonteCarloConfig()

    def _rng(self, observation: MarketObservation, purpose: str) -> np.random.Generator:
        identity = (
            f"{self.config.seed}|{purpose}|{observation.ticker}|"
            f"{observation.observed_at.isoformat()}|{observation.seconds_remaining}|"
            f"{observation.locked_sample_count}|{observation.locked_sample_sum}"
        )
        digest = hashlib.sha256(identity.encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    @staticmethod
    def _validate_settlement_clock(observation: MarketObservation) -> None:
        if observation.seconds_remaining <= 60:
            minimum_locked = max(0, 60 - observation.seconds_remaining - 1)
            if observation.locked_sample_count < minimum_locked:
                raise ValueError(
                    "final-minute observation is missing locked settlement samples"
                )

    def _simulate(
        self,
        observation: MarketObservation,
        *,
        include_probability_paths: bool,
        purpose: str,
    ) -> _Simulation:
        self._validate_settlement_clock(observation)
        rng = self._rng(observation, purpose)
        path_count = self.config.paths
        prices = np.full(path_count, float(observation.brti_price), dtype=np.float64)
        target = float(observation.target_price)
        sigma = max(
            float(observation.brti_sigma_per_sqrt_second),
            target * 1e-8,
        )
        locked_sum = np.full(path_count, float(observation.locked_sample_sum))
        locked_count = observation.locked_sample_count
        remaining_samples = 60 - locked_count
        wait_steps = max(0, observation.seconds_remaining - remaining_samples)
        total_steps = wait_steps + remaining_samples
        probability_steps = [] if include_probability_paths else None

        for step in range(total_steps):
            shocks = rng.normal(self.config.drift_per_second, sigma, path_count)
            if self.config.jump_probability_per_second:
                jumps = rng.random(path_count) < self.config.jump_probability_per_second
                shocks += jumps * rng.normal(
                    0.0,
                    sigma * self.config.jump_sigma_multiplier,
                    path_count,
                )
            prices = np.maximum(np.finfo(np.float64).eps, prices + shocks)

            if step >= wait_steps and locked_count < 60:
                locked_sum += prices
                locked_count += 1

            if probability_steps is not None:
                samples_left = 60 - locked_count
                wait_left = max(0, wait_steps - step - 1)
                if samples_left <= 0:
                    probability = (locked_sum / 60.0 >= target).astype(np.float64)
                elif locked_count:
                    required = (60.0 * target - locked_sum) / samples_left
                    std = sigma * sqrt(_future_average_variance_steps(samples_left))
                    probability = _normal_survival_approximation(required, prices, std)
                else:
                    std = sigma * sqrt(
                        wait_left + _future_average_variance_steps(samples_left)
                    )
                    probability = _normal_survival_approximation(
                        np.full(path_count, target), prices, std
                    )
                probability_steps.append(probability)

        terminal_yes = locked_sum / 60.0 >= target
        probability_matrix = (
            np.column_stack(probability_steps)
            if probability_steps
            else None
        )
        return _Simulation(terminal_yes, probability_matrix)

    def estimate(self, observation: MarketObservation) -> ProbabilityEstimate:
        simulation = self._simulate(
            observation,
            include_probability_paths=False,
            purpose="settlement",
        )
        yes_float = float(np.mean(simulation.terminal_yes))
        standard_error = sqrt(
            yes_float * (1.0 - yes_float) / self.config.paths
        )
        yes = Decimal(str(round(yes_float, 8)))
        return ProbabilityEstimate(
            yes=yes,
            no=Decimal("1") - yes,
            uncertainty=Decimal(str(round(standard_error, 8))),
            required_remaining_average=observation.required_remaining_average(),
        )

    def assess_path_risk(
        self,
        observation: MarketObservation,
        side: Side,
        *,
        stop_loss: Decimal = Decimal("0.50"),
        take_profit: Decimal = Decimal("0.94"),
    ) -> PathRiskEstimate:
        if not Decimal("0") <= stop_loss < take_profit <= Decimal("1"):
            raise ValueError("barriers must satisfy 0 <= stop < take profit <= 1")
        simulation = self._simulate(
            observation,
            include_probability_paths=True,
            purpose=f"risk:{side.value}:{stop_loss}:{take_profit}",
        )
        assert simulation.yes_probability_paths is not None
        probabilities = simulation.yes_probability_paths
        side_probabilities = probabilities if side is Side.UP else 1.0 - probabilities
        current_mid = float((observation.bid(side) + observation.ask(side)) / 2)
        current_spread = float(observation.ask(side) - observation.bid(side))

        initial_yes = float(np.mean(simulation.terminal_yes))
        initial_side_fair = initial_yes if side is Side.UP else 1.0 - initial_yes
        basis = current_mid - initial_side_fair
        elapsed = np.arange(1, side_probabilities.shape[1] + 1, dtype=np.float64)
        decay = np.exp(
            -log(2.0) * elapsed / self.config.market_basis_half_life_seconds
        )
        noise_rng = self._rng(observation, f"market-noise:{side.value}")
        noise = np.zeros(self.config.paths, dtype=np.float64)
        executable_bids = np.empty_like(side_probabilities)
        innovation_scale = self.config.market_noise_sigma * sqrt(
            1.0 - self.config.market_noise_persistence ** 2
        )
        for index in range(side_probabilities.shape[1]):
            noise = (
                self.config.market_noise_persistence * noise
                + noise_rng.normal(0.0, innovation_scale, self.config.paths)
            )
            executable_bids[:, index] = np.clip(
                side_probabilities[:, index]
                + basis * decay[index]
                - current_spread / 2.0
                + noise,
                0.0,
                1.0,
            )

        event = np.zeros(self.config.paths, dtype=np.int8)
        current_bid = float(observation.bid(side))
        if current_bid <= float(stop_loss):
            event.fill(-1)
        elif current_bid >= float(take_profit):
            event.fill(1)
        for index in range(executable_bids.shape[1]):
            unresolved = event == 0
            event[unresolved & (executable_bids[:, index] <= float(stop_loss))] = -1
            unresolved = event == 0
            event[unresolved & (executable_bids[:, index] >= float(take_profit))] = 1

        settlement_wins = (
            simulation.terminal_yes
            if side is Side.UP
            else ~simulation.terminal_yes
        )
        return PathRiskEstimate(
            side=side,
            settlement_win_probability=float(np.mean(settlement_wins)),
            stop_before_take_profit_probability=float(np.mean(event == -1)),
            take_profit_before_stop_probability=float(np.mean(event == 1)),
            neither_barrier_probability=float(np.mean(event == 0)),
            path_count=self.config.paths,
        )


class HybridProbabilityEstimator:
    """Keep XGBoost and Monte Carlo intact, then blend their predictions."""

    def __init__(
        self,
        xgboost: ProbabilityEstimator,
        monte_carlo: MonteCarloProbabilityEstimator,
        *,
        xgboost_weight: float = 0.50,
        monte_carlo_weight: float = 0.30,
        market_weight: float = 0.20,
    ) -> None:
        weights = (xgboost_weight, monte_carlo_weight, market_weight)
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("hybrid weights must be non-negative with a positive sum")
        total = sum(weights)
        self.xgboost = xgboost
        self.monte_carlo = monte_carlo
        self.weights = tuple(value / total for value in weights)

    def estimate(self, observation: MarketObservation) -> ProbabilityEstimate:
        xgb = self.xgboost.estimate(observation)
        mc = self.monte_carlo.estimate(observation)
        up_mid = float((observation.up_bid + observation.up_ask) / 2)
        down_mid = float((observation.down_bid + observation.down_ask) / 2)
        market_yes = up_mid / (up_mid + down_mid) if up_mid + down_mid else 0.5
        xgb_weight, mc_weight, market_weight = self.weights
        yes_float = (
            xgb_weight * float(xgb.yes)
            + mc_weight * float(mc.yes)
            + market_weight * market_yes
        )
        uncertainty = (
            xgb_weight * float(xgb.uncertainty)
            + mc_weight * float(mc.uncertainty)
            + market_weight * (float(observation.up_ask - observation.up_bid) / 2.0)
        )
        yes = Decimal(str(round(min(1.0, max(0.0, yes_float)), 8)))
        return ProbabilityEstimate(
            yes=yes,
            no=Decimal("1") - yes,
            uncertainty=Decimal(str(round(uncertainty, 8))),
            required_remaining_average=observation.required_remaining_average(),
        )


class MonteCarloGuardedPolicy:
    """Block BUY/ADD decisions whose simulated path risk is unacceptable."""

    def __init__(
        self,
        base_policy: DecisionPolicy,
        monte_carlo: MonteCarloProbabilityEstimator,
        config: MonteCarloGuardConfig | None = None,
    ) -> None:
        self.base_policy = base_policy
        self.monte_carlo = monte_carlo
        self.config = config or MonteCarloGuardConfig()
        self.last_risk: PathRiskEstimate | None = None
        self.risk_evaluation_count = 0

    def decide(self, observation: MarketObservation) -> Decision:
        # Risk belongs to this decision only.  Clearing it prevents a later
        # WAIT/HOLD tick from accidentally displaying a previous entry risk.
        self.last_risk = None
        decision = self.base_policy.decide(observation)
        if decision.action not in {Action.BUY, Action.ADD} or decision.side is None:
            return decision
        risk = self.monte_carlo.assess_path_risk(
            observation,
            decision.side,
            stop_loss=self.config.stop_loss,
            take_profit=self.config.take_profit,
        )
        self.last_risk = risk
        self.risk_evaluation_count += 1
        blocked_action = Action.HOLD if observation.position is not None else Action.WAIT
        if (
            risk.stop_before_take_profit_probability
            > self.config.maximum_stop_before_take_profit
        ):
            return Decision(
                blocked_action,
                decision.side,
                Decimal("0"),
                decision.probability,
                decision.edge,
                "monte_carlo_stop_risk",
            )
        if (
            risk.take_profit_before_stop_probability
            < self.config.minimum_take_profit_before_stop
        ):
            return Decision(
                blocked_action,
                decision.side,
                Decimal("0"),
                decision.probability,
                decision.edge,
                "monte_carlo_insufficient_tp_path",
            )
        return decision
