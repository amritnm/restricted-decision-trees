"""RandomForestRDT class alignment across bootstrap samples.

Each tree used to derive classes_ from its own bootstrap sample, while the
forest added every tree's predict_proba row into the forest's class columns.
A bootstrap that missed a class gave a shorter row that was read past its end
(undefined behaviour) and shifted onto the wrong columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdt.ensemble import RandomForestRDT


def _leaves(node, out=None):
    out = [] if out is None else out
    if node.is_leaf:
        out.append(node)
    else:
        _leaves(node.left, out)
        _leaves(node.right, out)
    return out


def _check_forest(rf, X):
    K = len(rf.classes_)
    P = rf.predict_proba(X)
    assert P.shape == (len(X), K)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    manual = np.zeros_like(P)
    for est, cols in zip(rf.estimators_, rf._model.feature_subsets_):
        np.testing.assert_array_equal(est.classes_, rf.classes_)
        manual += est.predict_proba(np.ascontiguousarray(X[:, cols]))
        for leaf in _leaves(est.root):
            counts = np.asarray(leaf.class_counts)
            assert counts.shape == (K,)
            assert counts[np.searchsorted(rf.classes_, leaf.value)] == counts.max()
    np.testing.assert_allclose(P, manual / len(rf.estimators_), rtol=1e-12, atol=1e-15)
    return P


def test_rare_class_missing_from_some_bootstraps():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.r_[np.zeros(19), np.ones(19), np.full(2, 2.0)]
    rf = RandomForestRDT(n_estimators=50, max_depth=2, max_features=None, random_state=0).fit(X, y)
    _check_forest(rf, X)
    missing = [all(leaf.class_counts[2] == 0 for leaf in _leaves(e.root)) for e in rf.estimators_]
    assert any(missing)
    for est, m in zip(rf.estimators_, missing):
        if m:
            assert np.all(est.predict_proba(X)[:, 2] == 0.0)


def test_single_class_bootstraps_and_labels_not_0_to_k():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 4))
    y = (rng.uniform(size=2000) < 0.01).astype(float)
    rf = RandomForestRDT(n_estimators=100, max_depth=3, max_features=None, max_samples=0.03,
                         oob_score=True, random_state=0).fit(X, y)
    assert sum(sum(leaf.class_counts[1] for leaf in _leaves(e.root)) == 0 for e in rf.estimators_) > 10
    _check_forest(rf, X)
    assert 0.0 <= rf.oob_score_ <= 1.0
    # five labels, one of them on a single row
    X = rng.normal(size=(120, 5))
    y = rng.choice([-3.0, 0.5, 7.0], size=120)
    y[:2] = 11.0
    y[2] = 42.0
    rf = RandomForestRDT(n_estimators=30, max_depth=4, random_state=0, oob_score=True).fit(X, y)
    np.testing.assert_array_equal(rf.classes_, [-3.0, 0.5, 7.0, 11.0, 42.0])
    _check_forest(rf, X)


@pytest.mark.slow
@pytest.mark.parametrize("n", [10, 1000, 20000])
@pytest.mark.parametrize("p", [1, 5, 50])
@pytest.mark.parametrize("bootstrap", [True, False])
def test_forest_probabilities_across_scenarios(n, p, bootstrap):
    rng = np.random.default_rng(n + p)
    X = rng.normal(size=(n, p))
    if p > 1:
        X[:, -1] = rng.integers(0, 3, size=n)
    u = rng.uniform(size=n)
    y = np.where(u < 0.01, 2.0, np.where(X[:, 0] + rng.normal(size=n) > 1.0, 1.0, 0.0))
    y[:3] = [0.0, 1.0, 2.0]
    kw = dict(n_estimators=8 if n < 20000 else 3, max_depth=3, bootstrap=bootstrap,
              max_samples=0.1 if bootstrap else 1.0, oob_score=bootstrap, random_state=0)
    w = rng.uniform(0.1, 3.0, size=n)
    rf = RandomForestRDT(n_jobs=1, **kw).fit(X, y, sample_weight=w)
    _check_forest(rf, X)
    par = RandomForestRDT(n_jobs=2, **kw).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(par.predict_proba(X), rf.predict_proba(X))
