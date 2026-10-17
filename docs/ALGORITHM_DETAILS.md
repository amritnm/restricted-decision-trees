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
improvement for that feature, weighted by the node's share of the training
samples (sample weight) — before any split is committed:

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
                    total_improvement += (node.weight / root.weight) * improvement
                    thresholds[node] = threshold

            if total_improvement > best_total_improvement:
                best_total_improvement = total_improvement
                best_feature, best_thresholds = feature, thresholds

        if best_feature is None or best_total_improvement + EPSILON < min_impurity_decrease:
            make every node in nodes_to_split a leaf
            break

        next_level = []
        for node in nodes_to_split:
            if node not in best_thresholds:
                make node a leaf        # best_feature cannot split it (see below)
                continue
            threshold = best_thresholds[node]
            node.left, node.right = split(node, best_feature, threshold)
            next_level += [node.left, node.right]

        current_level = next_level

    return root
```

`node.weight` is the node's number of training samples (or their total
`sample_weight`), so `(node.weight / root.weight) * improvement` is exactly
scikit-learn's *weighted impurity decrease* of a split,
`N_t / N * (impurity - N_t_L / N_t * left_impurity - N_t_R / N_t * right_impurity)`.
Without that weighting a node's within-node improvement would count the same
whether the node holds 10 samples or 10,000, and a feature that perfectly
separates a tiny node could win the whole depth over one that substantially
improves the large nodes. The winning total is stored per depth as
`depth_improvements_` (the basis of the ensembles' `feature_importances_`), and
`min_impurity_decrease` is compared against it: a depth is committed unless
its total weighted impurity decrease plus `EPSILON` (machine epsilon) is below
the threshold. For a single node (the root, or `max_depth=1`) this is exactly
scikit-learn's quantity and rule;
the difference is that scikit-learn applies it node by node while an RDT
applies it to the whole depth, since the depth shares one feature.

### Nodes the depth's feature cannot split

The winning feature may have no valid threshold in some node — it is constant
there, or every threshold would violate `min_samples_leaf`. Such a node
becomes a leaf at that depth, even if a feature chosen for a later depth
could have split it. This is deliberate: deferring the node and splitting it
later with depth `d+1`'s feature would place a depth-`d+1` feature at depth
`d` (breaking "all nodes at depth `d` use the same feature"), or would need a
pass-through node with a single child, which neither the tree structure nor
the per-depth feature map (`depth_features_`) can represent; it would also
break the "two-pass" equivalence argument below, which relies on every node
at a depth being split by that depth's feature. Weighting each node's
improvement by its size makes it unlikely that the depth's feature is one
that cannot split the large nodes, which is what makes freezing costly.

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

A candidate threshold between consecutive distinct values `a < b` is their
midpoint, except that — as in scikit-learn — `a` itself is used when the
midpoint is not strictly below `b` (adjacent floating-point values) or is not
finite, so `x <= threshold` always reproduces the scored partition. Tied
feature values are visited in row order, so the search performs the same
floating-point operations with every C++ standard library.

The native core sweeps the split point with incrementally-updated running
class counts / sums, rather than rescanning all samples for every candidate
threshold — an `O(1)`-amortized-per-candidate scan instead of `O(n)` per
candidate.

It also exploits the RDT structure: every node at a depth is scored on the
same candidate feature, so each feature's rows are **sorted once per fit**
(not once per node, feature and depth), and at each depth one pass over a
feature's presorted rows routes every row to its node and sweeps all of the
depth's nodes together (`core/include/rdt/presort.hpp`). Gradient Boosting
sorts each column once per `fit()` (the first time a round draws it) and
derives each round's (subsampled) order from it in linear time. This performs
exactly the same floating-point operations as the per-node sort — same
candidate boundaries, guarded midpoints, sample-count constraints, node
weighting, centred MSE sums, summation order of the running statistics
(tied feature values are visited in row order by both searches) and
tie-breaking — so both produce identical trees, on every platform. MAE keeps
the per-node path; `presort=False` selects it everywhere.

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

### Newton criterion (gradient boosting, `split_criterion="newton"`)

Each boosting round's tree can instead be grown on per-row gradients `g_i`
and hessians `h_i` of the loss (multiplied by the sample weight), as XGBoost
does. With `G`, `H` a node's sums and `T(G) = sign(G)·max(|G| - alpha, 0)`:

```
score(G, H) = T(G)^2 / (H + lambda)              (0 if H <= 0)
gain        = score(G_L, H_L) + score(G_R, H_R) - score(G, H)
```

(when `max_delta_step > 0`, `score` is evaluated at the clipped weight
`w`: `-(2 G w + (H + lambda) w^2 + 2 alpha |w|)`, as XGBoost ≥ 3.4 does;
3.0–3.3 leave out the `2 alpha |w|` term). A node's best threshold is
the valid boundary (both children `H >= min_child_weight` and `> 0`, and the
`min_samples_leaf` count rule) with the highest gain; the node splits only if
that gain is `>= gamma` and `> 1e-6`. The depth's feature is the one that
maximises `sum over splitting nodes of (gain - gamma)`; nodes it does not
split become leaves, so pruning by `gamma` happens node by node as the tree
grows (like XGBoost's `hist`; XGBoost's `exact` prunes after growing).
Leaves hold `-T(G) / (H + lambda)`, clipped to `±max_delta_step` when that is
`> 0`. The presorted search and the per-node search evaluate the same
expressions in the same order and agree bit for bit.

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
4. **No valid split on the depth's feature**: the feature selected for this
   depth has no threshold satisfying `min_samples_leaf` on both sides of this
   node (e.g. it is constant within the node). The node becomes a leaf even
   if another feature could split it — see "Nodes the depth's feature cannot
   split" above.
5. **Insufficient improvement**: the depth's total weighted impurity decrease
   is below `min_impurity_decrease` (every node at that depth becomes a leaf).

### Per-depth column sampling (`colsample_bylevel`)

In the ensembles, `colsample_bylevel < 1` restricts `candidate_features` at
each searched depth to a fresh random subset of `max(1, floor(f * m))` of the
tree's `m` columns (without replacement, ascending), drawn from a generator
seeded with `derive_seed(tree_seed, 0xB7)` before any thread starts; every
search (presorted, per-node, Newton, histogram) scores only those columns.
`level_features_` replays the draws from the seeds.

### Histogram search (`tree_method="hist"`)

Each feature is binned once per fit from the training rows with positive
weight: one bin per distinct value when there are at most `max_bin` of them,
otherwise approximately equal-count bins (boundaries only between distinct
values). Bin `b`'s upper cut is the guarded midpoint between the largest
training value in bin `b` and the smallest in bin `b + 1`; every row's bin is
the number of cuts below its value; the bins are stored in row blocks of 16
features (a row's bins of a block side by side). The tree keeps its rows
partitioned by node — each node's rows contiguous and in row order, a split
node's range divided between its children by a stable partition — so at each
depth a node's per-bin statistics (class weights; centred `w·d`, `w·d²`, `w`
for MSE; `G`, `H` for Newton) are accumulated over its own rows, for the 16
features of a block in one pass, and each node's bins are swept left to right
with the same candidate rule (between non-empty bins), `min_samples_leaf`
test, impurity / gain expressions and tie order as the exact search. Every
(node, bin) sums its rows in row order, as the one-pass-per-feature build of
earlier versions did, so the result is the same bit for bit.

With histogram subtraction (`hist_subtraction`, the default), at each depth
below the root a feature's histograms are built only for the smaller child of
each split node (by row count; the left child on a tie) and the other child's
are the parent's minus that child's, bin by bin — for the features whose
parent histograms were kept: those scored at both depths (with
`colsample_bylevel < 1` the next depth's candidates are drawn one depth early,
from the same sequence of draws), in candidate order while the kept histograms
of the current and the previous depth fit in `max_hist_cache_mb`. A child that
is not split itself (a leaf) may still be built when it is the smaller one.
MSE statistics are centred at the node's own weighted mean, which differs
between a parent and its children: below the root they are accumulated
centred at the parent's mean and re-centred to the node's before the sweep
(`Σw(y−c')² = Σw(y−c)² − 2δΣw(y−c) + δ²Σw`, `δ = c' − c`). Derived sums are
rounded differently from directly built ones (class weights excepted when
every weight is 1: counts subtract exactly), so a split score can change in
its last bits and an exact tie between two splits may be decided the other
way; `hist_subtraction=False` builds every histogram directly. A depth where
no candidate's parent histograms were kept (`max_hist_cache_mb=0`, or no
overlap with the previous depth's `colsample_bylevel` draw) is built directly
with the direct build's arithmetic (MSE centred at each node's own mean), bit
for bit. A split between a node's adjacent non-empty
bins `lo < hi` gets the threshold `split_threshold(max training value of bin
lo, min training value of bin hi)` — the exact search's threshold whenever
every bin holds one value — moved to the nearest bin boundary when a row
outside the training rows (validation, zero weight) falls inside the gap the
threshold would cut, so that for every row of `X`, `x <= threshold` exactly
when its bin is `<= split_bin`. Training scores (and the forest's OOB votes)
are routed through the bins; `X` is not read after binning.

Cost per depth: `O(m * n)` to build the histograms (`O(m * n_smaller)` with
subtraction, where `n_smaller` sums the smaller child of each split, typically
a third of `n`) plus `O(m * nodes * max_bin)` to sweep (and subtract) them;
binning costs one sort per feature over the training rows. Memory for the
subtraction cache: `nodes * sum over kept features of bins * (8 * stats + 4)`
bytes per depth (`stats` = 2 for Newton, 3 for MSE, the number of classes for
Gini / entropy), the current and the previous depth's together at most
`max_hist_cache_mb` per tree being fitted (Random Forest: up to `n_jobs`
trees at once; a multiclass boosting round: up to `min(n_jobs, n_classes)`).

## Complexity

For `d` depths, `m` candidate features and `n` samples:

- **Gini / entropy / MSE:** a one-off `O(m * n log n)` presort, then
  `O(m * n)` per depth — `O(m * n log n + d * m * n)` per tree. (Sorting every
  node's samples for every feature at every depth, as earlier versions did,
  costs `O(d * m * n log n)`.) Gradient Boosting pays the presort once per
  `fit()`; each round then costs `O(m * n)` to restrict it to the round's
  subsample plus `O(d * m * n_subsample)` for the search.
- **MAE:** the median is recomputed per candidate threshold, `O(n log n)`
  each, i.e. up to `O(d * m * n^2 log n)` per tree.

Memory during `fit` (in addition to `X`, `y` and the fitted trees):

- presorted orders: 4 bytes per (row, sorted column). A single tree and each
  Random Forest tree sort their own rows and columns; Gradient Boosting sorts
  a column over the training rows the first time a round draws it, keeps it
  until `fit` returns, and derives each round's order (4 bytes per (fit row,
  round column), freed after the round);
- search state: about 48 bytes per row (target, weight and node per row, and
  each depth's rows gathered per node), plus the nodes' row indices (4 bytes
  per row per level), as before;
- sorting: a transient 16 bytes per row per column sorted together, at most 8
  columns and at most `n_features / 4`;
- `tree_method="hist"` instead: the bins (1 or 2 bytes per value), two row
  arrays of 8 bytes per row each (view and bin-matrix row ids, double
  buffered for the partition), each depth's statistics (16 bytes per row for
  Newton, 24 for MSE, 12 for Gini / entropy), no per-node row indices (unless
  kept for `rdt.visualize`), and the subtraction caches: at most
  `max_hist_cache_mb` per tree being fitted, reused from round to round by
  gradient boosting (one cache per tree fitted at the same time, not one per
  class).

No copy of `X` is made by the ensembles at fit or predict time — each tree
reads its bootstrap rows / column subset through an index mapping (v0.2.0
copied `X[rows, columns]` for every tree and, in boosting, also `X[:, columns]`
every round).

## Limitations

- **Feature must generalize across a whole depth**: a feature only useful
  for some nodes at a depth, but not others, can lose out to a feature that's
  mediocre-but-consistent everywhere at that depth.
- **No categorical feature encoding** in this implementation — all inputs
  are treated as continuous; encode categoricals (e.g. one-hot or ordinal)
  before calling `fit`.
- **No missing-value support**: NaN or infinity in `X` raises `ValueError`
  at both `fit` and `predict`; impute (or encode missingness as an extra
  column) beforehand.

## References and related work

- **Oblivious decision trees** — used in CatBoost; same feature *and*
  threshold per depth (a stricter special case of this constraint).
- **Standard CART** — unrestricted feature and threshold selection.
- **Structural regularization** — constraining tree shape as an implicit
  regularizer, as opposed to post-hoc pruning.
