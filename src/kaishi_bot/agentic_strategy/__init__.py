"""Settlement-aware strategy primitives for research and paper trading.

This package is intentionally not wired to the live runtime.  It provides a
small, testable policy surface that can later accept a learned probability
estimator without changing the execution safety boundary.
"""

from kaishi_bot.agentic_strategy.models import (
    Action,
    AgentConfig,
    Decision,
    MarketObservation,
    PositionState,
    ProbabilityEstimate,
)
from kaishi_bot.agentic_strategy.monte_carlo import (
    HybridProbabilityEstimator,
    MonteCarloConfig,
    MonteCarloGuardConfig,
    MonteCarloGuardedPolicy,
    MonteCarloProbabilityEstimator,
    PathRiskEstimate,
)
from kaishi_bot.agentic_strategy.policy import AgenticPolicy
from kaishi_bot.agentic_strategy.probability import SettlementProbabilityModel
from kaishi_bot.agentic_strategy.tournament import (
    SelectionConfig,
    TrialMetrics,
    select_robust_top_k,
    successive_halving,
)
from kaishi_bot.agentic_strategy.xgboost_model import (
    DEFAULT_FEATURE_NAMES,
    DatasetSplit,
    HistogramCalibrator,
    TrainingRow,
    XGBoostConfig,
    XGBoostProbabilityEstimator,
    XGBoostTrainer,
    chronological_ticker_split,
    observation_features,
    ticker_balanced_weights,
)

__all__ = [
    "Action",
    "AgentConfig",
    "AgenticPolicy",
    "Decision",
    "MarketObservation",
    "HybridProbabilityEstimator",
    "MonteCarloConfig",
    "MonteCarloGuardConfig",
    "MonteCarloGuardedPolicy",
    "MonteCarloProbabilityEstimator",
    "PathRiskEstimate",
    "PositionState",
    "ProbabilityEstimate",
    "SelectionConfig",
    "SettlementProbabilityModel",
    "TrialMetrics",
    "DEFAULT_FEATURE_NAMES",
    "DatasetSplit",
    "HistogramCalibrator",
    "TrainingRow",
    "XGBoostConfig",
    "XGBoostProbabilityEstimator",
    "XGBoostTrainer",
    "chronological_ticker_split",
    "observation_features",
    "ticker_balanced_weights",
    "select_robust_top_k",
    "successive_halving",
]
