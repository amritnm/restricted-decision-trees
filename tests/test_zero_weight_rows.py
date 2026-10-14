"""Rows with sample_weight == 0 take no part in tree growth (as in
scikit-learn), and a Random Forest never fits a tree on a bootstrap sample
that holds only zero-weight rows.

Before: a split could isolate zero-weight rows into a leaf with all-zero
class counts (a predict_proba row of zeros), and an RF bootstrap drawing only
zero-weight rows produced an empty tree that dragged forest probabilities
towards 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdt import RDT
from rdt.ensemble import RandomForestRDT


def _split_nodes(node, out=None):
    out = [] if out is None else out
    if not node.is_leaf:
        out.append(node)
        _split_nodes(node.left, out)
        _split_nodes(node.right, out)
    return out


def test_zero_weight_leaf_no_longer_has_zero_probabilities():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    w = np.array([1.0, 1.0, 0.0, 1.0])
    P = RDT(max_depth=3).fit(X, y, sample_weight=w).predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_zero_weight_rows_are_equivalent_to_removing_them():
    for seed in range(6):
        rng = np.random.default_rng(200 + seed)
        n = int(rng.choice([12, 60, 250]))
        X = rng.integers(0, 6, size=(n, 3)).astype(float)
        y_clf = rng.integers(0, 3, size=n).astype(float)
        w = rng.uniform(0.1, 2.0, size=n)
        w[rng.uniform(size=n) < 0.4] = 0.0
        w[:2] = 1.0
        keep = w > 0
        for task, criterion, y in (("classification", "gini", y_clf),
                                   ("regression", "mse", y_clf + rng.normal(size=n))):
            kw = dict(task=task, criterion=criterion, max_depth=4, min_samples_leaf=2)
            full = RDT(**kw).fit(X, y, sample_weight=w)
            kept = RDT(**kw).fit(X[keep], y[keep], sample_weight=w[keep])
            assert full.depth_features_ == kept.depth_features_
            assert [nd.split_threshold for nd in _split_nodes(full.root)] == \
                [nd.split_threshold for nd in _split_nodes(kept.root)]
            np.testing.assert_allclose(full.predict(X), kept.predict(X))


def test_zero_weight_stump_matches_sklearn():
    tree = pytest.importorskip("sklearn.tree")
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([8, 30, 200]))
        X = rng.integers(0, 6, size=(n, 1)).astype(float)
        y = (rng.uniform(size=n) < 0.5).astype(float)
        y[:2] = [0.0, 1.0]
        w = rng.uniform(0.1, 2.0, size=n)
        w[rng.uniform(size=n) < 0.5] = 0.0
        w[:2] = 1.0
        sk = tree.DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        m = RDT(max_depth=1).fit(X, y, sample_weight=w)
        if sk.tree_.node_count == 1:
            assert m.root.is_leaf
        else:
            assert m.root.split_threshold == sk.tree_.threshold[0]


def _sparse_weight_problem(n=200, n_weighted=20):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(float)
    w = np.zeros(n)
    w[:n_weighted] = 1.0
    return X, y, w


def _leaves(node, out=None):
    out = [] if out is None else out
    if node.is_leaf:
        out.append(node)
    else:
        _leaves(node.left, out)
        _leaves(node.right, out)
    return out


def test_forest_redraws_bootstraps_without_weighted_rows():
    # 4-row bootstraps, 10% of rows weighted: ~2/3 of the draws hold no
    # weighted row. Those trees used to be empty: a leaf with all-zero class
    # counts (forest probability rows summing to ~0.3) or a regression value
    # of 0.
    X, y, w = _sparse_weight_problem()
    rf = RandomForestRDT(n_estimators=200, max_depth=3, max_features=None, max_samples=0.02,
                         oob_score=True, random_state=0).fit(X, y, sample_weight=w)
    for est in rf.estimators_:
        assert all(sum(leaf.class_counts) > 0 for leaf in _leaves(est.root))
    rfr = RandomForestRDT(task="regression", n_estimators=50, max_depth=3, max_features=None,
                          max_samples=0.02, random_state=0).fit(X, 100.0 + X[:, 0], sample_weight=w)
    assert rfr.predict(X).min() > 90.0
    # the redraws come from each tree's own generator: deterministic
    again = RandomForestRDT(task="regression", n_estimators=50, max_depth=3, max_features=None,
                            max_samples=0.02, random_state=0, n_jobs=2).fit(X, 100.0 + X[:, 0], sample_weight=w)
    np.testing.assert_array_equal(again.predict(X), rfr.predict(X))


def test_forest_raises_when_no_bootstrap_finds_a_weighted_row():
    X, y, w = _sparse_weight_problem(n=2000, n_weighted=1)
    with pytest.raises(ValueError, match="zero sample_weight"):
        RandomForestRDT(n_estimators=5, max_samples=1.0 / 2000, random_state=0).fit(X, y, sample_weight=w)
