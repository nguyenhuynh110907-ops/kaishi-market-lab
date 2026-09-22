from __future__ import annotations

from decimal import Decimal
from math import erfc, sqrt

from kaishi_bot.agentic_strategy.models import (
    AgentConfig,
    MarketObservation,
    ProbabilityEstimate,
)


SQRT_TWO = sqrt(2.0)


class SettlementProbabilityModel:
    """Transparent baseline for the probability of a YES settlement.

    The model treats one-second BRTI changes as a driftless random walk.  It is
    deliberately simple: a learned/calibrated estimator can replace this class
    through the same ``estimate`` interface after an auditable baseline exists.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    @staticmethod
    def _normal_survival(threshold: float, mean: float, std: float) -> float:
        if std <= 0:
            return 1.0 if mean >= threshold else 0.0
        z = (threshold - mean) / std
        return 0.5 * erfc(z / SQRT_TWO)

    @staticmethod
    def _future_average_variance_steps(sample_count: int) -> float:
        """Variance multiplier for the mean of a random-walk price path."""

        if sample_count <= 0:
            return 0.0
        n = float(sample_count)
        return ((n + 1.0) * (2.0 * n + 1.0)) / (6.0 * n)

    def estimate(self, observation: MarketObservation) -> ProbabilityEstimate:
        minimum_sigma = (
            observation.target_price * self.config.minimum_sigma_ratio
        )
        sigma = max(observation.brti_sigma_per_sqrt_second, minimum_sigma)
        required = observation.required_remaining_average()

        if observation.locked_sample_count >= 60:
            final_average = observation.locked_sample_sum / Decimal(60)
            yes = Decimal("1") if final_average >= observation.target_price else Decimal("0")
            return ProbabilityEstimate(
                yes=yes,
                no=Decimal("1") - yes,
                uncertainty=Decimal("0"),
                required_remaining_average=None,
            )

        if required is not None:
            remaining = 60 - observation.locked_sample_count
            variance_steps = self._future_average_variance_steps(remaining)
            threshold = required
        else:
            wait_steps = max(0, observation.seconds_remaining - 60)
            variance_steps = wait_steps + self._future_average_variance_steps(60)
            threshold = observation.target_price

        std = float(sigma) * sqrt(variance_steps)
        yes_float = self._normal_survival(
            float(threshold), float(observation.brti_price), std
        )
        yes = Decimal(str(round(min(1.0, max(0.0, yes_float)), 8)))
        uncertainty = Decimal(str(round(std, 8)))
        return ProbabilityEstimate(
            yes=yes,
            no=Decimal("1") - yes,
            uncertainty=uncertainty,
            required_remaining_average=required,
        )
