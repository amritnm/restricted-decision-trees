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
  exact Newton/hessian-derived formula — this engine boosts first-order
  (Friedman-style), not second-order.

See `examples/ensemble_usage.py` for a fuller, runnable example.

## API reference

### `rdt.RDT(task="classification", criterion=None, max_depth=5, min_samples_split=2, min_samples_leaf=1)`

**Parameters**

- `task` — `"classification"` or `"regression"`.
- `criterion` — splitting criterion. Defaults to `"gini"` for classification
  or `"mse"` for regression. Classification accepts `"gini"` or `"entropy"`;
  regression accepts `"mse"` or `"mae"`.
- `max_depth` — maximum tree depth.
- `min_samples_split` — minimum samples required to split a node.
- `min_samples_leaf` — minimum samples required in a leaf node.
- `presort` — `True` (default): presorted split search; `False`: the original
  per-node sort (see [Performance notes](#performance-notes)).

**Methods**

- `fit(X, y)` — build the tree from training data.
- `predict(X)` — predict target values.
- `predict_proba(X)` — predict class probabilities (classification only).

**Attributes (after fitting)**

- `depth_features_` — `dict[int, int]`, the feature index chosen at each depth.
- `n_features_` — number of features seen during `fit`.
- `classes_`, `n_classes_` — unique class labels and their count (classification only).
- `root` — the fitted tree's root `Node` (`.is_leaf`, `.split_feature`,
  `.split_threshold`, `.value`, `.class_counts`, `.left`, `.right`,
  `.n_samples`, `.indices`) for direct programmatic inspection. `.indices`
  (the training rows that reached the node) is kept by a standalone `RDT` and
  by `RandomForestRDT(bootstrap=False)` sub-trees; sub-trees fitted on a
  bootstrap sample drop it after fitting (accessing it raises
  `AttributeError`); `.n_samples` is always available.

### `rdt.ensemble.RandomForestRDT`

Bagging ensemble: `n_estimators`, `task`, `criterion`, `max_depth`,
`min_samples_split`, `min_samples_leaf` (alias `min_child_weight`),
`min_impurity_decrease` (alias `gamma`), `max_features` (alias
`colsample_bytree`), `bootstrap`, `max_samples`, `oob_score`, `class_weight`
(`'balanced'` or dict), `random_state`, `n_jobs`, `presort`. After fitting:
`estimators_` (list of `RDT`, individually introspectable),
`feature_importances_`, `oob_score_` (if requested).

### `rdt.ensemble.GradientBoostingRDT`

First-order boosting: `n_estimators`, `task`, `loss` (classification:
`'log_loss'`, `'exponential'`; regression: `'squared_error'`,
`'absolute_error'`, `'huber'`, `'quantile'`), `learning_rate`, `max_depth`,
`min_samples_split`, `min_samples_leaf` (alias `min_child_weight`),
`min_impurity_decrease` (alias `gamma`), `subsample`, `max_features` (alias
`colsample_bytree`), `validation_fraction`, `n_iter_no_change` (alias
`early_stopping_rounds`), `tol`, `reg_alpha`, `reg_lambda`,
`scale_pos_weight` (binary only), `huber_delta`, `quantile_alpha`,
`random_state`, `n_jobs`, `presort`. After fitting: `feature_importances_`,
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

## Performance notes

- **Split search:** each feature is sorted once per fit and every depth scores
  all of its nodes in one pass over the presorted rows, instead of sorting
  every node's samples for every feature at every depth (see
  [docs/ALGORITHM_DETAILS.md](docs/ALGORITHM_DETAILS.md#complexity)).
- **Results:** the same as the per-node sort (and v0.2.0) except at exact or
  near ties — two candidate splits whose improvements are equal to within
  floating-point rounding, typically a column and an affine or mirrored copy
  of it (e.g. `x` and `1 - x`) — which may be broken the other way; on
  tie-heavy data `depth_improvements_` / `feature_importances_` can also
  differ in the last bits. `presort=False` (on `RDT`, `RandomForestRDT`,
  `GradientBoostingRDT`) uses the per-node sort and reproduces v0.2.0 bit for
  bit.
- **Memory while fitting:** the presorted orders take 4 bytes per (row,
  sorted column) — Gradient Boosting sorts a column the first time a round
  draws it and keeps it until `fit` returns; Random Forest sorts each tree's
  own sample and column subset — plus about 48 bytes per row of search state,
  and sorting needs a transient buffer of 16 bytes per row for each of up to
  8 columns sorted together (at most `n_features / 4`). This is less than
  v0.2.0's per-tree copies of `X` for the ensembles on wide data, but more than
  v0.2.0 for a single tree on tall, narrow data (e.g. 2,000,000 x 4: about
  +190 MB vs +100–130 MB) and for boosting with few rounds and a small
  `colsample_bytree`; `presort=False` avoids it.
- **No copies of `X`:** the ensembles fit and predict through each tree's
  rows / column subset in place.
- **GIL:** `fit`, `predict` and `predict_proba` release the GIL while the
  native code runs, so other Python threads (including other fits) keep
  running. One native model object (`rdt._rdt_native.*Model`) may serve
  several `predict` calls at once, but calling `fit` on it while it is in use
  by another thread (or using it while it is being fitted) raises
  `RuntimeError`; the `rdt` estimators fit a fresh native object and publish
  it only once fitted, so this does not affect them. The native
  `predict` / `predict_proba` now return NumPy arrays (they returned lists).
- `n_jobs` parallelizes across trees: Random Forest trees, and a multiclass
  boosting round's per-class trees. Binary / regression boosting fits one
  tree per round, so it does not use extra threads.

## How it works

For each depth, every candidate feature is scored across **all** currently
active nodes at that depth (summing each node's best achievable impurity
improvement for that feature); the feature with the highest total wins and is
locked in for the whole depth, and the per-node thresholds already found
while scoring it are reused directly as the final split points:

```
For each depth level:
  1. For every candidate feature:
       for every active node at this depth:
         find that node's best threshold for this feature
       total_improvement = sum of improvements across all nodes
  2. Choose the feature with the highest total_improvement; lock it in.
  3. Split every node at this depth using the threshold already found
     for the winning feature.
```

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
