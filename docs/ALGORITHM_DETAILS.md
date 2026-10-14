# Restricted Decision Tree — Algorithm Details

## Overview

This document explains the Restricted Decision Tree (RDT) algorithm as
implemented by this package's native C++ core.

## Core concept

### The restriction constraint

In a Restricted Decision Tree:
- **All nodes at depth `d` use the SAME feature for splitting.**
- **Each node at depth `d` can use a DIFFERENT threshold.**

This is the key distinction from:
- **Standard decision trees**: each node uses any feature and threshold.
- **Oblivious decision trees**: all nodes at depth `d` use the same feature
  *and* the same threshold.

### Benefits

1. **Interpretability**: easy to see which feature matters at each depth.
2. **Regularization**: the constraint reduces overfitting relative to
   unrestricted trees.
3. **Staged decisions**: each depth can map cleanly onto one real-world
   information source or decision stage.

## Algorithm

For each depth, every candidate feature is scored across *all* currently
active nodes at that depth — summing each node's best achievable impurity
improvement for that feature — before any split is committed:

```
function BuildRestrictedTree(X, y, max_depth):
    root = Node(indices = all_samples)
    current_level = [root]

    for depth = 0 to max_depth - 1:
        if current_level is empty:
            break

        nodes_to_split = [n in current_level if n is splittable]
        if nodes_to_split is empty:
            break

        best_feature, best_thresholds = None, {}
        best_total_improvement = -infinity

        for feature in candidate_features:
            total_improvement = 0
            thresholds = {}
            for node in nodes_to_split:
                threshold, improvement = find_best_threshold(node, feature)
                if threshold is not None:
                    total_improvement += improvement
                    thresholds[node] = threshold

            if total_improvement > best_total_improvement:
                best_total_improvement = total_improvement
                best_feature, best_thresholds = feature, thresholds

        next_level = []
        for node in nodes_to_split:
            threshold = best_thresholds[node]
            node.left, node.right = split(node, best_feature, threshold)
            next_level += [node.left, node.right]

        current_level = next_level

    return root
```

### Finding the best threshold for a (node, feature) pair

```
function find_best_threshold(node, feature):
    values = sorted(unique(X[node.indices, feature]))
    candidates = midpoints of consecutive values in `values`

    best_improvement, best_threshold = -infinity, None
    for threshold in candidates:
        left, right = partition(node.indices, feature, threshold)
        if |left| < min_samples_leaf or |right| < min_samples_leaf:
            continue
        improvement = impurity(node) - weighted_average(impurity(left), impurity(right))
        if improvement > best_improvement:
            best_improvement, best_threshold = improvement, threshold

    return best_threshold, best_improvement
```

The native core sorts each `(node, feature)` pair's values once and sweeps
the split point with incrementally-updated running class counts / sums,
rather than rescanning all samples for every candidate threshold — an
`O(1)`-amortized-per-candidate scan instead of `O(n)` per candidate.

### On "two-pass"

Some descriptions of this style of algorithm (including the paper this
package's algorithm is based on) frame it as **two-pass**: a first pass picks
the winning feature per depth using *provisional* thresholds, then a second
pass discards those and re-optimizes each node's threshold given the
finalized (cascading) partitions from earlier depths.

This implementation only performs the first pass, reusing its
already-computed thresholds directly — and that turns out not to be an
approximation. Scoring a candidate feature for selection requires already
computing its best achievable threshold per node, so the "provisional"
threshold for the feature that ends up winning is, by construction, already
the exhaustive per-node optimum. By induction depth-by-depth (the root has no
ancestors to differ on, so pass one and a hypothetical pass two agree
trivially there; that makes every deeper level's node membership identical
between the two as well), a literal second pass is guaranteed to reproduce
values already on hand. Given deterministic threshold search, the two
formulations are result-equivalent — this implementation just skips the
redundant recomputation.

## Splitting criteria

### Classification

**Gini impurity**
```
Gini(S) = 1 - sum(p_i^2)
```
where `p_i` is the proportion of class `i` in set `S`. Range `[0, 1 - 1/k]`
for `k` classes; `0` is a pure node.

**Entropy**
```
Entropy(S) = -sum(p_i * log2(p_i))
```
Range `[0, log2(k)]`; more sensitive to probability changes than Gini.

### Regression

**Mean squared error (MSE)**
```
MSE(S) = mean((y_i - mean(y))^2)
```
The variance of the target values in `S`; leads to a mean prediction in leaves.

**Mean absolute error (MAE)**
```
MAE(S) = mean(|y_i - median(y)|)
```
More robust to outliers; leads to a median prediction in leaves.

## Leaf prediction

**Classification**: the majority class among the leaf's samples (ties broken
by whichever class was encountered first while scanning the leaf's samples).

**Regression**: the mean of the leaf's target values (MSE criterion) — the
implementation always uses the mean as the leaf value, independent of which
regression criterion was used to choose splits.

With `sample_weight`, counts and means are weighted. Samples whose weight is
zero take no part in growing the tree, exactly as if those rows were removed
(as in scikit-learn), so no node — in particular no leaf — has zero total
weight. A Random Forest bootstrap sample that happens to contain only
zero-weight rows is redrawn (up to 100 times, deterministically for a given
`random_state`); if none of the draws contains a positively weighted row,
`fit` raises `ValueError`.

## Prediction

```
function predict_sample(x, node):
    if node.is_leaf:
        return node.value
    if x[node.split_feature] <= node.split_threshold:
        return predict_sample(x, node.left)
    else:
        return predict_sample(x, node.right)
```

## Stopping criteria

Tree growth stops at a node when:

1. **Max depth reached**: `depth >= max_depth`.
2. **Insufficient samples**: `n_samples < min_samples_split`.
3. **Pure node**: all samples share the same target value.
4. **No valid split**: no candidate feature/threshold satisfies
   `min_samples_leaf` on both sides.

## Complexity

`O(d * m * n)` for `d` depths, `m` candidate features, `n` samples —
asymptotically the same order as a standard greedy tree's `O(d * m * n)`,
just with a higher constant factor from scoring every feature against every
node at a depth before committing to a split (versus a standard tree scoring
each node independently).

## Limitations

- **Feature must generalize across a whole depth**: a feature only useful
  for some nodes at a depth, but not others, can lose out to a feature that's
  mediocre-but-consistent everywhere at that depth.
- **No categorical feature encoding** in this implementation — all inputs
  are treated as continuous; encode categoricals (e.g. one-hot or ordinal)
  before calling `fit`.

## References and related work

- **Oblivious decision trees** — used in CatBoost; same feature *and*
  threshold per depth (a stricter special case of this constraint).
- **Standard CART** — unrestricted feature and threshold selection.
- **Structural regularization** — constraining tree shape as an implicit
  regularizer, as opposed to post-hoc pruning.
