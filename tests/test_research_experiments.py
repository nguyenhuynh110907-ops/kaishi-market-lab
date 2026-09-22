from kaishi_bot.agentic_strategy.tournament import SelectionConfig
from kaishi_bot.research_experiments import (
    CandidateConfig,
    ExperimentContext,
    StressProfile,
    TrialRunResult,
    WalkForwardExperimentRunner,
)
from kaishi_bot.research_store import ResearchStore


def test_experiment_runner_persists_screening_and_validation_without_test(tmp_path) -> None:
    store = ResearchStore(tmp_path / "dashboard.sqlite3")
    runner = WalkForwardExperimentRunner(
        folds=("fold-1",), seeds=(7,),
        stress_profiles=(
            StressProfile("normal"), StressProfile("fee-x1.5", fee_multiplier=1.5),
            StressProfile("latency-plus-500", added_latency_ms=500),
        ),
        selection=SelectionConfig(
            top_k=1, minimum_closed_trades=1,
            minimum_profitable_fold_ratio=0,
            minimum_worst_fold_return=-1, maximum_drawdown=1,
            minimum_fee_stress_return=-1, minimum_latency_stress_return=-1,
        ),
    )
    candidates = (
        CandidateConfig("strong", "rule", {"edge": 2}),
        CandidateConfig("weak", "rule", {"edge": 1}),
    )

    def evaluate(candidate, fold, seed, stress, fraction):
        split = "train" if fraction < 1 or stress.name == "normal" and fraction == 1 else "validation"
        # The final 100% halving call is also train; validation calls include all stresses.
        if fraction == 1 and evaluate.screen_complete:
            split = "validation"
        result = TrialRunResult(
            candidate.candidate_id, fold, seed, stress.name, split,
            float(candidate.parameters["edge"]), (1.0, 2.0), 5, 0.1, 1.0,
            -0.1, f"{candidate.candidate_id}-{split}-{stress.name}-{fraction}",
        )
        if fraction == 1 and not evaluate.screen_complete:
            evaluate.screen_complete = True
        return result

    evaluate.screen_complete = False
    try:
        result = runner.run(
            candidates, evaluate,
            context=ExperimentContext("dataset-v1", "split-v1", "abc123"),
            sink=store,
        )
        assert result.selected[0].trial_id == "strong"
        persisted_splits = {
            row[0] for row in store.connection.execute(
                "SELECT DISTINCT split FROM research_trial_runs"
            )
        }
        assert persisted_splits == {"train", "validation"}
        assert store.connection.execute(
            "SELECT COUNT(*) FROM research_trial_runs WHERE split='test'"
        ).fetchone()[0] == 0
        assert store.connection.execute(
            "SELECT COUNT(*) FROM research_experiment_candidates WHERE selected=1"
        ).fetchone()[0] == 1
    finally:
        store.close()
