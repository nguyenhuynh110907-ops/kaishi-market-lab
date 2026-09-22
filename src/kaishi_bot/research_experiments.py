from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from kaishi_bot.agentic_strategy.tournament import (
    SelectionConfig,
    TrialMetrics,
    select_robust_top_k,
    successive_halving,
)


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    candidate_id: str
    model_type: str
    parameters: Mapping[str, object]

    @property
    def config_hash(self) -> str:
        raw = json.dumps(self.parameters, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class StressProfile:
    name: str
    fee_multiplier: float = 1.0
    added_latency_ms: int = 0
    visible_depth_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class TrialRunResult:
    candidate_id: str
    fold_id: str
    seed: int
    stress_profile: str
    split: str
    net_return: float
    pnl_series: tuple[float, ...]
    closed_trades: int
    maximum_drawdown: float
    sharpe: float
    cvar: float
    trajectory_hash: str
    resource_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_id: str
    selected: tuple[TrialMetrics, ...]
    runs: tuple[TrialRunResult, ...]
    finalist_hash: str
    started_at: datetime
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    dataset_version: str
    split_manifest_id: str
    code_commit: str | None = None
    artifact_root: str | None = None


class ExperimentSink(Protocol):
    def save_experiment_result(
        self, result: ExperimentResult, candidates: Sequence[CandidateConfig],
        context: ExperimentContext, runner_config: Mapping[str, object],
    ) -> None: ...


Evaluator = Callable[[CandidateConfig, str, int, StressProfile, float], TrialRunResult]


class WalkForwardExperimentRunner:
    """Multiple-seed/stress runner with a validation-only selector boundary."""

    def __init__(
        self, *, folds: Sequence[str], seeds: Sequence[int],
        stress_profiles: Sequence[StressProfile], selection: SelectionConfig | None = None,
    ) -> None:
        if not folds or not seeds or not stress_profiles:
            raise ValueError("folds, seeds, and stress profiles cannot be empty")
        self.folds = tuple(folds)
        self.seeds = tuple(seeds)
        self.stress_profiles = tuple(stress_profiles)
        self.selection = selection or SelectionConfig()

    def run(
        self, candidates: Iterable[CandidateConfig], evaluator: Evaluator, *,
        context: ExperimentContext | None = None,
        sink: ExperimentSink | None = None,
    ) -> ExperimentResult:
        if (context is None) != (sink is None):
            raise ValueError("context and sink must be supplied together")
        started_at = datetime.now(UTC)
        candidates = tuple(candidates)
        lookup = {candidate.candidate_id: candidate for candidate in candidates}
        if len(lookup) != len(candidates):
            raise ValueError("candidate IDs must be unique")

        screening_runs: list[TrialRunResult] = []

        def screen(candidate_id: str, fraction: float) -> float:
            result = evaluator(
                lookup[candidate_id], self.folds[0], self.seeds[0],
                self.stress_profiles[0], fraction,
            )
            if result.split != "train":
                raise ValueError("successive halving may use train metrics only")
            screening_runs.append(replace(result, resource_fraction=fraction))
            return result.net_return

        survivors = successive_halving(lookup, screen)
        runs: list[TrialRunResult] = list(screening_runs)
        metrics: list[TrialMetrics] = []
        for candidate_id in survivors:
            candidate_runs = [
                evaluator(lookup[candidate_id], fold, seed, stress, 1.0)
                for fold in self.folds
                for seed in self.seeds
                for stress in self.stress_profiles
            ]
            if any(item.split != "validation" for item in candidate_runs):
                raise ValueError("tournament runs must use validation metrics only")
            runs.extend(candidate_runs)
            normal = [item for item in candidate_runs if item.stress_profile == "normal"]
            fee = [item for item in candidate_runs if "fee" in item.stress_profile]
            latency = [item for item in candidate_runs if "latency" in item.stress_profile]
            source = normal or candidate_runs
            metrics.append(TrialMetrics(
                trial_id=candidate_id,
                fold_returns=tuple(item.net_return for item in source),
                sharpe=sum(item.sharpe for item in source) / len(source),
                max_drawdown=max(item.maximum_drawdown for item in source),
                cvar=min(item.cvar for item in source),
                fee_stress_return=min((item.net_return for item in fee), default=0.0),
                latency_stress_return=min(
                    (item.net_return for item in latency), default=0.0
                ),
                closed_trades=sum(item.closed_trades for item in source),
                pnl_series=tuple(value for item in source for value in item.pnl_series),
                split="validation",
            ))
        selected = tuple(select_robust_top_k(metrics, self.selection))
        freeze = json.dumps([
            {"id": item.trial_id, "config_hash": lookup[item.trial_id].config_hash}
            for item in selected
        ], sort_keys=True, separators=(",", ":"))
        result = ExperimentResult(
            str(uuid.uuid4()), selected, tuple(runs),
            hashlib.sha256(freeze.encode()).hexdigest(), started_at, datetime.now(UTC),
        )
        if sink is not None and context is not None:
            sink.save_experiment_result(
                result, candidates, context,
                {
                    "folds": self.folds,
                    "seeds": self.seeds,
                    "stress_profiles": [asdict(item) for item in self.stress_profiles],
                    "selection": asdict(self.selection),
                },
            )
        return result


def assert_final_test_is_reporting_only(
    finalists: Sequence[TrialMetrics], test_metrics: Sequence[TrialMetrics]
) -> None:
    """Validate the final report without re-running selection or changing rank."""
    if any(item.split != "validation" for item in finalists):
        raise ValueError("finalists must have been frozen from validation")
    if any(item.split != "test" for item in test_metrics):
        raise ValueError("final report metrics must be tagged test")
