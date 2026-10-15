# Restricted Decision Trees (RDT)

A native **C++** implementation of **Restricted Decision Trees** — a tree-based
algorithm that sits between the rigidity of oblivious trees and the flexibility
of standard decision trees, exposed to Python via [nanobind](https://github.com/wjakob/nanobind).

## What is a Restricted Decision Tree?

In a **Restricted Decision Tree**, all nodes at the same depth use the **same
feature** for splitting, but each node can have a **different threshold**.
This constraint provides:

- **Structure**: more interpretable than standard decision trees — the feature
  chosen at each depth is a natural, depth-ordered importance ranking.
- **Flexibility**: more adaptable than oblivious trees (which use the same
  feature *and* threshold at every node in a depth).
- **Staged decisions**: each depth can correspond to one information source
  (e.g. a KYC vendor check, a medical test panel, a fraud-investigation
  stage) — useful when a real deployment pipeline is itself staged.

### Tree Type Comparison

| Tree Type | Feature Selection | Threshold Selection | Structure |
|-----------|------------------|-------------------|-----------|
| **Standard Decision Tree** | Different feature per node | Different threshold per node | Unrestricted |
| **Restricted Decision Tree** | **Same feature per depth** | **Different threshold per node** | Semi-structured |
| **Oblivious Decision Tree** | Same feature per depth | Same threshold per depth | Highly structured |

## Features

- ✅ Classification (binary & multi-class) and regression
- ✅ Multiple splitting criteria: Gini, entropy, MSE, MAE
- ✅ Probability predictions (`predict_proba`)
- ✅ Random Forest and Gradient Boosting ensembles (`rdt.ensemble`), with real
  cross-tree parallelism and a parameter surface matching scikit-learn/XGBoost/LightGBM
- ✅ Interactive D3.js tree visualization, plain-text tree printing, and JSON export
- ✅ Native C++ core — validated for exact output parity (predictions, per-depth
  feature choice, probabilities) against the project's original Cython
  reference implementation, and substantially faster in practice

## Install

```bash
pip install restricted-decision-trees
```

## Quick start

```python
import numpy as np
from rdt import RDT

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

model = RDT(task="classification", criterion="gini", max_depth=3)
model.fit(X, y)

model.predict(X)          # array([0., 0., 0., 1., 1., 1.])
model.predict_proba(X)    # per-class probabilities
model.depth_features_     # {0: 0} -- feature index chosen at each depth
```

For regression, pass `task="regression"` (criterion defaults to `"mse"`;
`"mae"` is also available).

### Visualizing a tree

```python
from rdt.visualize import print_tree, visualize_d3_observable, export_tree_dict

print_tree(model, feature_names=["x"])

# Interactive, collapsible D3.js visualization -- writes an HTML file you
# open in a browser (click nodes to expand/collapse, scroll/drag to zoom+pan).
visualize_d3_observable(model, y, feature_names=["x"], filename="tree.html")

# Or get the same tree structure + per-node stats as a plain dict/JSON,
# for a custom visualization of your own.
tree_dict = export_tree_dict(model, y, feature_names=["x"])
```

See `examples/basic_usage.py` and `examples/d3_visualization_demo.py` for
fuller, runnable examples.

### Ensembles

```python
from rdt.ensemble import RandomForestRDT, GradientBoostingRDT

rf = RandomForestRDT(n_estimators=100, max_depth=4, oob_score=True, n_jobs=-1)
rf.fit(X_train, y_train)
rf.predict(X_test)
rf.feature_importances_
rf.oob_score_

gb = GradientBoostingRDT(n_estimators=100, learning_rate=0.1, max_depth=3,
                          validation_fraction=0.15, n_iter_no_change=10)  # early stopping
gb.fit(X_train, y_train)
gb.predict(X_test)
```

Parameter names follow scikit-learn where a concept maps directly to a
scikit-learn parameter; where XGBoost/LightGBM use a different name for the
same concept (`colsample_bytree`, `gamma`, `min_child_weight`,
`early_stopping_rounds`, ...) that name is accepted as an alias — setting
both names to conflicting values raises. See `rdt/ensemble.py`'s module
docstring for the full mapping, including the two places where RDT's
structural constraint (every node at a depth splits on the same feature)
means a parameter can't work exactly like it does elsewhere:

- **`max_features`/`colsample_bytree`** can't be resampled *per node* the
  way scikit-learn/XGBoost do it — each tree gets one random column subset
  chosen once, before that tree is fit.
- **`reg_alpha`/`reg_lambda`** (`GradientBoostingRDT` only) are a real
  effect (post-hoc leaf-value soft-thresholding/shrinkage) but not XGBoost's
  exact regularised leaf formula — this engine is Friedman's gradient
  boosting as in scikit-learn (trees grown on the gradient with a
  squared-error criterion, then a per-leaf Newton step for `log_loss` and
  `exponential`), not XGBoost's hessian-weighted tree growth.

See `examples/ensemble_usage.py` for a fuller, runnable example.

## API reference

### `rdt.RDT(task="classification", criterion=None, max_depth=5, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0)`

**Parameters**

- `task` — `"classification"` or `"regression"`.
- `criterion` — splitting criterion. Defaults to `"gini"` for classification
  or `"mse"` for regression. Classification accepts `"gini"` or `"entropy"`;
  regression accepts `"mse"` or `"mae"`.
- `max_depth` — maximum tree depth.
- `min_samples_split` — minimum samples required to split a node.
- `min_samples_leaf` — minimum samples required in a leaf node.
- `min_impurity_decrease` — a depth is only split if its total weighted
  impurity decrease (sum over the depth's nodes of
  `node_samples / total_samples * node_improvement`, scikit-learn's
  definition and comparison, up to machine epsilon) is at least this value.

**Methods**

- `fit(X, y, sample_weight=None)` — build the tree from training data.
- `predict(X)` — predict target values.
- `predict_proba(X)` — predict class probabilities (classification only).

All estimators validate their inputs the way scikit-learn does and raise
`ValueError` for: `X` that is not 2D or is empty; NaN or infinity in `X`,
`y` or `sample_weight` (missing values are not supported — impute them
first); `y`/`sample_weight` whose length differs from `X`'s rows; negative
or all-zero weights (also after `class_weight` / `scale_pos_weight` are
applied; `scale_pos_weight` must be finite and > 0); and, at predict time, an
`X` with a different number of features than during `fit`.

**Attributes (after fitting)**

- `depth_features_` — `dict[int, int]`, the feature index chosen at each depth.
- `depth_improvements_` — `dict[int, float]`, the total weighted impurity
  decrease achieved at each depth.
- `n_features_` — number of features seen during `fit`.
- `classes_`, `n_classes_` — unique class labels and their count (classification only).
- `root` — the fitted tree's root `Node` (`.is_leaf`, `.split_feature`,
  `.split_threshold`, `.value`, `.class_counts`, `.left`, `.right`, `.indices`)
  for direct programmatic inspection.

### `rdt.ensemble.RandomForestRDT`

Bagging ensemble: `n_estimators`, `task`, `criterion`, `max_depth`,
`min_samples_split`, `min_samples_leaf` (alias `min_child_weight`),
`min_impurity_decrease` (alias `gamma`), `max_features` (alias
`colsample_bytree`), `bootstrap`, `max_samples`, `oob_score`, `class_weight`
(`'balanced'` or dict), `random_state`, `n_jobs`. `max_samples` follows
scikit-learn: an int is the number of rows drawn per tree, a float is a
fraction of `n_samples`, `None` means `n_samples`. After fitting:
`estimators_` (list of `RDT`, individually introspectable),
`feature_importances_`, `oob_score_` (if requested).

### `rdt.ensemble.GradientBoostingRDT`

Friedman gradient boosting, matching scikit-learn's
`GradientBoostingClassifier`/`Regressor`: each round grows an RDT on the
negative gradient, then sets each leaf to the loss's one-step Newton estimate
(`log_loss`, `exponential`; the leaf mean is already exact for
`squared_error`; `absolute_error`/`huber`/`quantile` keep the leaf mean of the
gradient, a documented simplification). The initial score is the weighted
prior log-odds / log class priors / weighted mean, median or quantile, and
`sample_weight` (or `scale_pos_weight`) also weights `train_score_` and the
early-stopping validation loss. Parameters: `n_estimators`, `task`, `loss` (classification:
`'log_loss'`, `'exponential'`; regression: `'squared_error'`,
`'absolute_error'`, `'huber'`, `'quantile'`), `learning_rate`, `max_depth`,
`min_samples_split`, `min_samples_leaf` (alias `min_child_weight`),
`min_impurity_decrease` (alias `gamma`), `subsample`, `max_features` (alias
`colsample_bytree`), `validation_fraction`, `n_iter_no_change` (alias
`early_stopping_rounds`), `tol`, `reg_alpha`, `reg_lambda`,
`scale_pos_weight` (binary only), `huber_delta`, `quantile_alpha`,
`random_state`, `n_jobs`. Classification needs at least two classes in
`y`. After fitting: `feature_importances_`,
`train_score_` (loss per round), `n_estimators_` (rounds actually fitted —
less than `n_estimators` if early stopping triggered).

### `rdt.visualize`

- `print_tree(model, feature_names=None)` — print an indented text
  representation of a fitted tree.
- `export_tree_dict(model, y, feature_names=None, class_names=None) -> dict` —
  export the tree as a nested dict with per-node statistics (sample count,
  impurity, target average, leaf purity).
- `visualize_d3_observable(model, y, feature_names=None, class_names=None, filename="rdt_tree_d3.html", width=1200, height=800) -> str` —
  render an interactive, collapsible D3.js tree to an HTML file (and return
  the HTML string).

## How it works

For each depth, every candidate feature is scored across **all** currently
active nodes at that depth (summing each node's best achievable impurity
improvement for that feature, weighted by the node's share of the training
samples); the feature with the highest total wins and is locked in for the
whole depth, and the per-node thresholds already found while scoring it are
reused directly as the final split points:

```
For each depth level:
  1. For every candidate feature:
       for every active node at this depth:
         find that node's best threshold for this feature
       total_improvement = sum over nodes of
                           (node samples / total samples) * node improvement
  2. Choose the feature with the highest total_improvement; lock it in.
  3. Split every node at this depth using the threshold already found
     for the winning feature (a node that feature cannot split becomes a leaf).
```

The weighted per-node term is scikit-learn's weighted impurity decrease, so a
`max_depth=1` RDT is the same stump as `DecisionTreeClassifier`/`Regressor`,
and `depth_improvements_`/`feature_importances_` are scikit-learn-style
impurity-based (MDI) importances. See `docs/ALGORITHM_DETAILS.md` for details,
including why a node the depth's feature cannot split is not deferred to a
later depth.

## Package structure

```
restricted-decision-trees/
├── README.md
├── LICENSE
├── pyproject.toml, CMakeLists.txt      # nanobind + scikit-build-core build
├── core/include/rdt/                   # header-only C++ algorithm core
├── src/bindings.cpp                    # nanobind bindings
├── python/rdt/
│   ├── __init__.py                     # RDT (sklearn-style wrapper)
│   ├── ensemble.py                     # RandomForestRDT / GradientBoostingRDT
│   └── visualize.py                    # print_tree / D3 viz / JSON export
├── tests/                              # pytest unit tests
└── examples/
    ├── basic_usage.py
    ├── ensemble_usage.py
    └── d3_visualization_demo.py
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
