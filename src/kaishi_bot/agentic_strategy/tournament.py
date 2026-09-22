from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil, sqrt


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    trial_id: str
    fold_returns: tuple[float, ...]
    sharpe: float
    max_drawdown: float
    cvar: float
    fee_stress_return: float
    latency_stress_return: float
    closed_trades: int
    pnl_series: tuple[float, ...] = ()
    split: str = "validation"

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("trial metric split must be train, validation, or test")

    @property
    def median_return(self) -> float:
        ordered = sorted(self.fold_returns)
        if not ordered:
            return float("-inf")
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    @property
    def worst_fold_return(self) -> float:
        return min(self.fold_returns, default=float("-inf"))

    @property
    def profitable_fold_ratio(self) -> float:
        if not self.fold_returns:
            return 0.0
        return sum(value > 0 for value in self.fold_returns) / len(self.fold_returns)


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    top_k: int = 5
    minimum_closed_trades: int = 300
    minimum_profitable_fold_ratio: float = 0.70
    minimum_worst_fold_return: float = -0.05
    maximum_drawdown: float = 0.15
    minimum_fee_stress_return: float = 0.0
    minimum_latency_stress_return: float = 0.0
    maximum_pnl_correlation: float = 0.85


def _percentile_ranks(values: Sequence[float], *, higher_is_better: bool) -> list[float]:
    if not values:
        return []
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(ordered):
        percentile = rank / denominator
        result[index] = percentile if higher_is_better else 1.0 - percentile
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    size = min(len(left), len(right))
    if size < 2:
        return 0.0
    a, b = left[:size], right[:size]
    mean_a, mean_b = sum(a) / size, sum(b) / size
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    variance_a = sum((x - mean_a) ** 2 for x in a)
    variance_b = sum((y - mean_b) ** 2 for y in b)
    denominator = sqrt(variance_a * variance_b)
    return numerator / denominator if denominator else 0.0


def select_robust_top_k(
    trials: Iterable[TrialMetrics], config: SelectionConfig | None = None
) -> list[TrialMetrics]:
    """Filter, rank, then select non-redundant trials.

    The function never looks at a final holdout set.  Its inputs must contain
    training/walk-forward validation metrics only.
    """

    supplied = list(trials)
    invalid = sorted({trial.split for trial in supplied if trial.split != "validation"})
    if invalid:
        raise ValueError(
            "robust top-k accepts validation metrics only; received: "
            + ", ".join(invalid)
        )
    settings = config or SelectionConfig()
    eligible = [
        trial
        for trial in supplied
        if trial.closed_trades >= settings.minimum_closed_trades
        and trial.profitable_fold_ratio >= settings.minimum_profitable_fold_ratio
        and trial.worst_fold_return >= settings.minimum_worst_fold_return
        and trial.max_drawdown <= settings.maximum_drawdown
        and trial.fee_stress_return >= settings.minimum_fee_stress_return
        and trial.latency_stress_return >= settings.minimum_latency_stress_return
    ]
    if not eligible or settings.top_k <= 0:
        return []

    metrics = (
        ([trial.median_return for trial in eligible], True, 0.30),
        ([trial.worst_fold_return for trial in eligible], True, 0.20),
        ([trial.sharpe for trial in eligible], True, 0.15),
        ([trial.max_drawdown for trial in eligible], False, 0.15),
        ([trial.cvar for trial in eligible], True, 0.10),
        (
            [min(trial.fee_stress_return, trial.latency_stress_return) for trial in eligible],
            True,
            0.10,
        ),
    )
    scores = [0.0] * len(eligible)
    for values, higher_is_better, weight in metrics:
        for index, rank in enumerate(
            _percentile_ranks(values, higher_is_better=higher_is_better)
        ):
            scores[index] += weight * rank
    ranked = [
        trial
        for _, trial in sorted(
            zip(scores, eligible),
            key=lambda item: (item[0], item[1].median_return, item[1].trial_id),
            reverse=True,
        )
    ]

    selected: list[TrialMetrics] = []
    for trial in ranked:
        if any(
            _correlation(trial.pnl_series, prior.pnl_series)
            > settings.maximum_pnl_correlation
            for prior in selected
        ):
            continue
        selected.append(trial)
        if len(selected) >= settings.top_k:
            break
    return selected


def successive_halving(
    candidate_ids: Iterable[str],
    evaluator: Callable[[str, float], float],
    *,
    resource_fractions: Sequence[float] = (0.20, 0.50, 1.00),
    retention_fraction: float = 0.20,
) -> list[str]:
    """Evaluate cheap-to-expensive stages and retain the strongest candidates."""

    remaining = list(candidate_ids)
    if not 0 < retention_fraction <= 1:
        raise ValueError("retention fraction must be in (0, 1]")
    if not resource_fractions or any(
        not 0 < fraction <= 1 for fraction in resource_fractions
    ):
        raise ValueError("resource fractions must be in (0, 1]")
    for stage, fraction in enumerate(resource_fractions):
        scored = sorted(
            ((evaluator(candidate, fraction), candidate) for candidate in remaining),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        if stage == len(resource_fractions) - 1:
            return [candidate for _, candidate in scored]
        keep = max(1, ceil(len(scored) * retention_fraction))
        remaining = [candidate for _, candidate in scored[:keep]]
    return remaining
