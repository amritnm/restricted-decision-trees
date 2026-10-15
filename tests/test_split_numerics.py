"""Numerical robustness of the per-(node, feature) split search.

- the committed threshold must reproduce the scored partition (midpoint
  overflow / +-inf / adjacent doubles used to break that);
- the MSE criterion must be shift-invariant (raw E[y^2] - E[y]^2 sums used to
  cancel catastrophically for targets with a large offset).
"""

from __future__ import annotations

import numpy as np
import pytest

from rdt import RDT, _rdt_native

Y6 = np.array([0, 0, 0, 1, 1, 1], dtype=float)


def _split_nodes(node, out=None):
    out = [] if out is None else out
    if not node.is_leaf:
        out.append(node)
        _split_nodes(node.left, out)
        _split_nodes(node.right, out)
    return out


def test_overflowing_midpoint_still_splits():
    for col in ([1e308] * 3 + [1.7e308] * 3, [-1.7e308] * 3 + [-1e308] * 3):
        X = np.array(col)[:, None]
        m = RDT(max_depth=1).fit(X, Y6)
        assert not m.root.is_leaf
        assert X[2, 0] <= m.root.split_threshold < X[3, 0]
        np.testing.assert_array_equal(m.predict(X), Y6)


def test_adjacent_doubles_commit_the_scored_split():
    a = 1.0 + 2.0 ** -52
    b = np.nextafter(a, 2.0)
    assert (a + b) / 2 == b  # the unguarded midpoint lands on b
    m = RDT(max_depth=1).fit(np.array([[a]] * 3 + [[b]] * 3), Y6)
    assert not m.root.is_leaf and m.root.split_threshold == a
    # with a third value the unguarded midpoint committed a different
    # partition (a and b left) than the one it scored (only a left)
    X = np.array([[a], [a], [a], [b], [b], [b], [5.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=float)
    m = RDT(max_depth=1).fit(X, y)
    assert sorted(m.root.left.indices) == [0, 1, 2]
    assert np.mean(m.predict(X) == y) == pytest.approx(0.75)


def test_infinite_values_via_native_module():
    # The public wrappers reject non-finite X (input validation); the core
    # itself must still commit the scored partition when given +-inf.
    for task, criterion, y in (("classification", "gini", Y6), ("regression", "mse", Y6 * 10),
                               ("regression", "mae", Y6 * 10)):
        for col, expected in (([1.0, 2.0, 3.0, np.inf, np.inf, np.inf], 3.0),
                              ([-np.inf, -np.inf, -np.inf, 1.0, 2.0, 3.0], -np.inf)):
            m = _rdt_native.RDTModel(task, criterion, 1, 2, 1, 0.0)
            m.fit(np.ascontiguousarray(np.array(col)[:, None]), y)
            assert not m.root.is_leaf and m.root.split_threshold == expected
            assert sorted(m.root.left.indices) == [0, 1, 2]


def test_midpoint_unchanged_for_ordinary_values():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4)) * np.array([1.0, 1e-3, 1e5, 7.0])
    y = (X[:, 0] + X[:, 2] / 1e5 > 0).astype(float)
    m = RDT(max_depth=4).fit(X, y)
    for node in _split_nodes(m.root):
        lo = X[np.asarray(node.left.indices), node.split_feature].max()
        hi = X[np.asarray(node.right.indices), node.split_feature].min()
        assert node.split_threshold == (lo + hi) / 2


def test_large_target_offset_does_not_change_the_split():
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.uniform(0, 1, size=(n, 3))
    signal = np.where(X[:, 1] > 0.6, 1.0, -1.0) + rng.normal(0, 0.3, n)
    ref = RDT(task="regression", max_depth=1).fit(X, signal)
    for offset in (1e6, 1e8, 1e9, 1e12, -1e9):
        m = RDT(task="regression", max_depth=1).fit(X, signal + offset)
        assert m.root.split_feature == ref.root.split_feature == 1
        assert m.root.split_threshold == ref.root.split_threshold
        assert m.depth_improvements_[0] == pytest.approx(ref.depth_improvements_[0], rel=1e-6)


def _best_stump_improvement(X, y, w):
    """Reference: best weighted-variance decrease of one split over all
    features, from dVar = (W_L W_R / W^2)(mu_L - mu_R)^2 on centred values."""
    W = w.sum()
    yc = y - np.sum(w * y) / W
    best = -np.inf
    for f in range(X.shape[1]):
        o = np.argsort(X[:, f], kind="stable")
        xs, ys, ws = X[o, f], yc[o], w[o]
        wl = np.cumsum(ws)[:-1]
        sl = np.cumsum(ws * ys)[:-1]
        wr, sr = W - wl, np.sum(ws * ys) - sl
        valid = (xs[:-1] != xs[1:]) & (wl > 0) & (wr > 0)
        if valid.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                imp = wl * wr / W**2 * (sl / wl - sr / wr) ** 2
            best = max(best, np.max(imp[valid]))
    return best


def test_mse_root_split_matches_centred_reference_under_offsets():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([10, 200, 1000]))
        p = int(rng.choice([1, 5]))
        X = rng.integers(0, 50, size=(n, p)) / 8.0
        y = np.round(X @ rng.normal(size=p) + rng.normal(size=n), 3)
        w = rng.uniform(0.1, 2, size=n) if seed % 2 else np.ones(n)
        for offset in (0.0, 2.0**30, 3e10):
            y_shifted = y + offset
            y_eff = y_shifted - offset  # what the model actually sees (Sterbenz: exact)
            best = _best_stump_improvement(X, y_eff, w)
            m = RDT(task="regression", max_depth=1).fit(X, y_shifted, sample_weight=w)
            assert m.depth_improvements_[0] == pytest.approx(best, rel=1e-9), (seed, offset)


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 10, 1000, 20000])
@pytest.mark.parametrize("p", [1, 5, 50])
def test_numerics_across_shapes(n, p):
    rng = np.random.default_rng(n + 7 * p)
    X = rng.normal(size=(n, p))
    if p > 1:
        X[:, 1] = rng.integers(0, 2, size=n)
    if p > 3:
        X[:, 3] = np.round(X[:, 3], 1)
    y = 1e9 + X[:, 0] + rng.normal(size=n)
    depth = 3 if n * p <= 50000 else 2
    reg = RDT(task="regression", max_depth=depth).fit(X, y)
    reg0 = RDT(task="regression", max_depth=depth).fit(X, y - 1e9)
    for node in _split_nodes(reg.root):
        x_left = X[np.asarray(node.left.indices), node.split_feature]
        x_right = X[np.asarray(node.right.indices), node.split_feature]
        assert x_left.max() <= node.split_threshold < x_right.min()
    if n >= 1000:
        assert reg.depth_features_ == reg0.depth_features_
        assert reg.root.split_threshold == reg0.root.split_threshold
