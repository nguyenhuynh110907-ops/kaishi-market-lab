# Agentic strategy research module

This package contains the first auditable baseline for a future learned
strategy.  It is deliberately disconnected from live order placement.

## Components

- `MarketObservation` combines executable Kalshi quotes, BRTI, target,
  settlement samples, data-quality flags, and current position state.
- `SettlementProbabilityModel` estimates settlement probability with a simple
  driftless random-walk model.  A calibrated supervised model can later replace
  it through the same `estimate()` interface.
- `AgenticPolicy` implements confirmation-gated entry, probability-based exit,
  partial take-profit, a two-factor emergency stop, and one final-minute add.
- `select_robust_top_k` filters trials on walk-forward robustness before ranking
  and removes highly correlated policies.
- `successive_halving` supports cheap-to-expensive candidate screening.
- `xgboost_model` provides whole-ticker chronological splitting, per-ticker
  sample weights, validation-only probability calibration, artifact metadata,
  and an estimator compatible with `AgenticPolicy`.
- `monte_carlo` independently simulates BRTI paths and the exact 60-sample
  settlement average. It also estimates first passage to 50c/94c, blends the
  unchanged XGBoost estimator with Monte Carlo and market price, and can wrap a
  policy to reject BUY/ADD decisions with excessive stop-before-target risk.

The comparison surfaces remain separate: use `XGBoostProbabilityEstimator`
for the original XGBoost-only baseline, `MonteCarloProbabilityEstimator` for
Monte Carlo only, or `HybridProbabilityEstimator` for the ensemble.

```python
xgb = XGBoostProbabilityEstimator.load(artifact_directory)
mc = MonteCarloProbabilityEstimator()

xgb_only = AgenticPolicy(xgb)
mc_only = AgenticPolicy(mc)
hybrid = AgenticPolicy(HybridProbabilityEstimator(xgb, mc))
hybrid_with_path_guard = MonteCarloGuardedPolicy(hybrid, mc)
```

Install the optional trainer dependency with `pip install -e '.[ml]'`. Training
artifacts contain a UBJSON booster plus JSON metadata. Final test tickers are
recorded but never used for early stopping, calibration, or model selection.

## Safety boundary

Do not import this package from the live runtime until research capture,
chronological replay, latency/slippage simulation, walk-forward validation, and
paper-shadow acceptance criteria are implemented.  The final holdout set must
not be passed into the tournament selector.
