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

**Methods**

- `fit(X, y)` — build the tree from training data.
- `predict(X)` — predict target values.
- `predict_proba(X)` — predict class probabilities (classification only).

**Attributes (after fitting)**

- `depth_features_` — `dict[int, int]`, the feature index chosen at each depth.
- `n_features_` — number of features seen during `fit`.
- `classes_`, `n_classes_` — unique class labels and their count (classification only).
- `root` — the fitted tree's root `Node` (`.is_leaf`, `.split_feature`,
  `.split_threshold`, `.value`, `.class_counts`, `.left`, `.right`, `.indices`)
  for direct programmatic inspection.

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
│   └── visualize.py                    # print_tree / D3 viz / JSON export
├── tests/                              # pytest unit tests
└── examples/
    ├── basic_usage.py
    └── d3_visualization_demo.py
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
