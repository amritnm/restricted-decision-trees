# Changelog

## 0.3.0

A large correctness and performance release. Several fixes change fitted models; see "Behaviour changes".

### New features
- **`GradientBoostingRDT(split_criterion="newton")`**: XGBoost-style regularised second-order split gain
  (`reg_lambda`, `reg_alpha`, `gamma`, hessian-based `min_child_weight`, `max_delta_step`), restricted to one
  feature per depth. Matches XGBoost >= 3.4 `exact` to float32 precision. (#15)
- **`tree_method="hist"`** with `max_bin` for `RDT`, `RandomForestRDT` and `GradientBoostingRDT`: binned split
  search that reads float32/float64 input of any layout without copying it. Equals the exact search on data with
  at most `max_bin` distinct values per column, apart from exact ties. (#11)
- **`colsample_bylevel`** for `GradientBoostingRDT` and `RandomForestRDT`, plus `level_features_`. (#12)
- **Early stopping on a user-supplied validation set**: `fit(..., eval_set=(X_val, y_val), eval_sample_weight=...)`
  with `early_stopping_rounds` and `eval_metric` (`logloss`, `auc`, `error`, `rmse`, `mae`); the model is truncated
  to `best_iteration_`, with `best_score_` and `evals_result_`. (#14)
- **Staged predictions**: `decision_function`, `staged_decision_function`, `staged_predict_proba`,
  `staged_predict`; per-tree `depth_features_` on both ensembles. (#9)
- `presort=`, `hist_subtraction=` and `max_hist_cache_mb=` parameters; `Node.n_samples`;
  `GradientBoostingRDT.depth_improvements_`. (#8, #13)

### Performance
- Presorted per-depth split search; ensembles no longer copy `X`; the GIL is released during fit and predict. (#8)
- Histogram mode: per-node histogram building, histogram subtraction, and parallel per-round passes
  (gradients, routing, score updates, prediction). About 3-5x faster per boosting round than the first hist
  implementation. (#13)

### Fixes
- Input validation: NaN/inf, shapes, sample weights (including after `class_weight`/`scale_pos_weight`),
  single-class targets for boosting. (#1)
- Split search: guarded midpoint thresholds, centred MSE sums, deterministic tie order. (#3)
- Zero-weight rows take no part in tree growth; empty forest bootstraps are redrawn. (#4)
- Random forests align every tree to the forest's classes. (#5)
- A depth's feature is chosen by size-weighted node improvement (scikit-learn's definition). (#6)
- Gradient boosting: Newton leaf values for log-loss/exponential, weighted initial score, weighted training and
  early-stopping loss, `sigmoid(2F)` for the exponential loss. (#7)
- Builds with `-ffp-contract=off` under GCC, so results are identical across platforms. (#16)

### Behaviour changes
- Boosted models fit differently (#7): larger, correct Newton steps; re-tune `learning_rate` or use early stopping.
- `max_samples` given as an int is a row count, as in scikit-learn. (#2)
- `reg_alpha`, `reg_lambda` and `gamma`/`min_impurity_decrease` act on different scales than in 0.2.0. (#6, #7)
- NaN or inf in `X` now raises `ValueError`.
- The private native `predict` returns NumPy arrays; `Node.indices` raises on bootstrapped and boosting trees. (#8)
- `GradientBoostingRDT.tol` and `validation_fraction` default to `None` (meaning 1e-4 and 0.1 on the
  internal-holdout path, 0 tolerance on the `eval_set` path). (#14)

### Infrastructure
- Test CI on Linux (x86_64 and aarch64), macOS and Windows, plus a wheel build smoke test. (#16)
