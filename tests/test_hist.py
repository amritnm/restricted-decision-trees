"""tree_method='hist'.

1. With at most max_bin distinct values per feature every value has its own
   bin, so hist scores the same partitions as exact: the fitted models agree
   bit for bit (predictions on data from the same value grid, depth features,
   leaves) -- or, where the two searches' different summation orders decide an
   exact tie differently, the first divergence is verified to be a tie with
   exact rational arithmetic. Thresholds are the exact search's too (the
   midpoint of the node's neighbouring values), so this holds under row
   subsampling and bootstrap and for predictions on new data from the grid.
2. The tree equals a brute-force restricted tree run on the bin indices.
3. Routing: for every row (validation / zero-weight rows included) and every
   split, bin <= split_bin exactly when x <= threshold, and a node's training
   rows are exactly the rows the raw thresholds route to it.
4. Binning: one bin per value up to max_bin, approximately equal-count bins
   beyond, uint16 above 256 bins.
5. Inputs: float32 equals float64, Fortran order / strided / memmap X equals a
   C copy, and fit does not copy X (subprocess memory test).
6. n_jobs does not change the result; staged predictions agree with the
   training scores computed through the bins.

XGBoost hist parity is in test_hist_xgboost.py.
"""

import subprocess
import sys
import textwrap
from fractions import Fraction

import numpy as np
import pytest

from rdt import RDT, _rdt_native
from rdt.ensemble import GradientBoostingRDT, RandomForestRDT


# --------------------------------------------------------------------------- exact rational tie check
def _F(v):
    return Fraction(float(v))


def _impurity(criterion, y, w):
    W = sum(w, Fraction(0))
    if W == 0:
        return Fraction(0)
    if criterion == "gini":
        tot = {}
        for yi, wi in zip(y, w):
            tot[yi] = tot.get(yi, Fraction(0)) + wi
        return 1 - sum((c / W) ** 2 for c in tot.values())
    mean = sum((wi * yi for yi, wi in zip(y, w)), Fraction(0)) / W
    return sum((wi * (yi - mean) ** 2 for yi, wi in zip(y, w)), Fraction(0)) / W


def _T(G, a):
    if G > a:
        return G - a
    if G < -a:
        return G + a
    return Fraction(0)


def _nscore(G, H, p):
    if H <= 0:
        return Fraction(0)
    if p["mds"] == 0:
        return _T(G, p["alpha"]) ** 2 / (H + p["lam"])
    w = -_T(G, p["alpha"]) / (H + p["lam"])
    if abs(w) > p["mds"]:
        w = p["mds"] if w > 0 else -p["mds"]
    return -(2 * G * w + (H + p["lam"]) * w * w + 2 * p["alpha"] * abs(w))


def _node_score(kind, idx, left, data, p, W_root):
    """Exact score a split (rows `idx`, mask `left`) adds to its depth's feature score."""
    if kind == "newton":
        g, h = data["g"], data["h"]
        G, H = sum((g[i] for i in idx), Fraction(0)), sum((h[i] for i in idx), Fraction(0))
        GL = sum((g[i] for i, l in zip(idx, left) if l), Fraction(0))
        HL = sum((h[i] for i, l in zip(idx, left) if l), Fraction(0))
        gain = _nscore(GL, HL, p) + _nscore(G - GL, H - HL, p) - _nscore(G, H, p)
        return gain - p["gamma"]
    y, w = data["y"], data["w"]
    yl = [y[i] for i, l in zip(idx, left) if l]
    wl = [w[i] for i, l in zip(idx, left) if l]
    yr = [y[i] for i, l in zip(idx, left) if not l]
    wr = [w[i] for i, l in zip(idx, left) if not l]
    W, WL, WR = sum(wl + wr, Fraction(0)), sum(wl, Fraction(0)), sum(wr, Fraction(0))
    imp = _impurity(kind, yl + yr, wl + wr) - (WL / W) * _impurity(kind, yl, wl) - (WR / W) * _impurity(kind, yr, wr)
    return W / W_root * imp


def _routes(node, X, idx):
    left = X[idx, node.split_feature] <= node.split_threshold
    return idx[left], idx[~left]


def first_divergence_is_tie(a, b, X, kind, data, p, root_rows):
    """Walk two trees fitted on the same rows level by level; at the first
    difference (depth feature or a node's partition), check with exact
    arithmetic that both choices score the same. True only if such a
    difference exists and is an exact tie (trees whose partitions agree
    everywhere are not a tie: any remaining difference is in thresholds)."""
    W_root = sum((data["w"][i] for i in root_rows), Fraction(0)) if kind != "newton" else None
    level = [(a.root, b.root, np.asarray(root_rows))]
    depth = 0
    while level:
        split = [(na, nb, idx) for na, nb, idx in level if not na.is_leaf or not nb.is_leaf]
        if not split:
            return False
        fa, fb = a.depth_features_.get(depth), b.depth_features_.get(depth)
        if fa != fb:
            # Each tree's chosen feature, scored over the depth's nodes with that
            # tree's own partitions, must total the same.
            def total(tree_side):
                s = Fraction(0)
                for na, nb, idx in split:
                    n = na if tree_side == 0 else nb
                    if not n.is_leaf:
                        left = X[idx, n.split_feature] <= n.split_threshold
                        s += _node_score(kind, list(idx), list(left), data, p, W_root)
                return s
            return total(0) == total(1)
        nxt = []
        for na, nb, idx in split:
            if na.is_leaf != nb.is_leaf:
                return False  # same feature, one node splits and the other does not: not a tie
            la = X[idx, fa] <= na.split_threshold
            lb = X[idx, fb] <= nb.split_threshold
            if not np.array_equal(la, lb):
                return _node_score(kind, list(idx), list(la), data, p, W_root) == \
                    _node_score(kind, list(idx), list(lb), data, p, W_root)
            nxt += [(na.left, nb.left, idx[la]), (na.right, nb.right, idx[~la])]
        level = nxt
        depth += 1
    return False


def _trees_equal(a, b, X):
    return a.depth_features_ == b.depth_features_ and np.array_equal(a.predict(X), b.predict(X))


# --------------------------------------------------------------------------- 1. hist == exact
def _grid_problem(rng, kind):
    n = int(rng.integers(8, 61))
    p = int(rng.integers(1, 5))
    X = rng.integers(0, int(rng.integers(2, 7)), (n, p)) / 4.0
    if kind == "gini":
        y = rng.integers(0, int(rng.integers(2, 4)), n).astype(float)
    else:
        y = rng.integers(-4, 5, n) / 2.0
    w = None
    if rng.random() < 0.5:
        w = rng.integers(1, 5, n) / 2.0
        if rng.random() < 0.3:
            w[rng.random(n) < 0.2] = 0.0
            w[0] = 1.0
    return X, y, w


@pytest.mark.parametrize("kind", ["gini", "mse", "newton"])
def test_hist_equals_exact_up_to_rational_ties(kind):
    rng = np.random.default_rng({"gini": 1, "mse": 2, "newton": 3}[kind])
    equal = ties = 0
    for _ in range(150):
        X, y, w = _grid_problem(rng, kind)
        depth = int(rng.integers(1, 5))
        msl = int(rng.choice([1, 1, 2]))
        mid = float(rng.choice([0.0, 0.0, 0.01]))
        pr = dict(lam=float(rng.choice([0.0, 1.0])), alpha=float(rng.choice([0.0, 0.5])),
                  gamma=float(rng.choice([0.0, 0.25])), mcw=float(rng.choice([0.0, 1.0])),
                  mds=float(rng.choice([0.0, 0.5])))
        h = np.ones(len(y)) if w is None else w
        models = []
        for tm in ("exact", "hist"):
            if kind == "newton":
                # a newton tree on given gradients / hessians (zero-hessian rows included)
                m = _rdt_native.RDTModel("regression", "newton", depth, 2, msl, 0.0, reg_lambda=pr["lam"],
                                         reg_alpha=pr["alpha"], gamma=pr["gamma"], min_child_weight=pr["mcw"],
                                         max_delta_step=pr["mds"], tree_method=tm)
                (m.fit if tm == "exact" else m.fit_hist)(X, y * h, hessian=h)
            else:
                m = RDT(task="classification" if kind == "gini" else "regression", criterion=kind,
                        max_depth=depth, min_samples_leaf=msl, min_impurity_decrease=mid, tree_method=tm)
                m.fit(X, y, sample_weight=w)
            models.append(m)
        a, b = models
        grid = rng.integers(0, 7, (40, X.shape[1])) / 4.0
        if _trees_equal(a, b, X) and _trees_equal(a, b, grid):
            equal += 1
            continue
        if kind == "newton":
            data = dict(g=[_F(v) for v in y * h], h=[_F(v) for v in h])
            pf = {k: _F(v) for k, v in pr.items()}
            rows = np.arange(len(y))
        else:
            data = dict(y=[_F(v) for v in y], w=[_F(v) for v in h])
            pf = None
            rows = np.flatnonzero(h > 0)
        assert first_divergence_is_tie(a, b, X, kind, data, pf, rows)
        ties += 1
    print(f"\n[hist vs exact, {kind}] equal {equal}, rational ties {ties}")
    assert equal >= 100, (equal, ties)


def _ens_problem(kind, seed, n=300, p=5):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 40, (n, p)) / 8.0
    s = X[:, 0] - X[:, 1] * X[:, 2] / 4 + rng.normal(0, 0.5, n)
    if kind == "regression":
        y = s
    elif kind == "multi":
        y = np.digitize(s, np.quantile(s, [0.33, 0.66])).astype(float)
    else:
        y = (s > np.median(s)).astype(float)
    w = rng.uniform(0.1, 3.0, n)
    w[rng.random(n) < 0.1] = 0.0
    return X, y, w


ENSEMBLES = [
    ("gb-binary", lambda tm, **k: GradientBoostingRDT(n_estimators=8, max_depth=3, subsample=0.8,
                                                      colsample_bytree=0.8, random_state=1, tree_method=tm, **k)),
    ("gb-multi", lambda tm, **k: GradientBoostingRDT(n_estimators=5, max_depth=2, random_state=2,
                                                     tree_method=tm, **k)),
    ("gb-newton", lambda tm, **k: GradientBoostingRDT(n_estimators=8, max_depth=3, split_criterion="newton",
                                                      reg_alpha=0.3, gamma=0.1, random_state=3, tree_method=tm, **k)),
    ("gb-reg-es", lambda tm, **k: GradientBoostingRDT(task="regression", loss="huber", n_estimators=30, max_depth=3,
                                                      n_iter_no_change=3, random_state=4, tree_method=tm, **k)),
    ("rf", lambda tm, **k: RandomForestRDT(n_estimators=6, max_depth=4, max_features=3, oob_score=True,
                                           random_state=5, tree_method=tm, **k)),
    ("rf-reg", lambda tm, **k: RandomForestRDT(task="regression", n_estimators=6, max_depth=4, random_state=6,
                                               tree_method=tm, **k)),
]


@pytest.mark.parametrize("name,make", ENSEMBLES, ids=[e[0] for e in ENSEMBLES])
def test_hist_ensembles_equal_exact(name, make):
    kind = "regression" if "reg" in name else "multi" if "multi" in name else "binary"
    X, y, w = _ens_problem(kind, len(name))
    a = make("exact").fit(X, y, sample_weight=w)
    b = make("hist").fit(X, y, sample_weight=w)
    Xt = np.random.default_rng(0).integers(0, 40, (200, X.shape[1])) / 8.0
    for data in (X, Xt):
        assert np.array_equal(a.predict(data), b.predict(data))
        if kind != "regression":
            assert np.array_equal(a.predict_proba(data), b.predict_proba(data))
    assert a.depth_features_ == b.depth_features_
    np.testing.assert_allclose(a.feature_importances_, b.feature_importances_, rtol=1e-9, atol=1e-15)
    if name.startswith("gb"):
        assert a.n_estimators_ == b.n_estimators_
        np.testing.assert_allclose(a.train_score_, b.train_score_, rtol=1e-12)
    if name == "rf":
        assert a.oob_score_ == b.oob_score_


def test_wide_bins_equal_exact():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 1000, (3000, 2)).astype(float)
    y = (X[:, 0] + rng.normal(0, 100, 3000) > 500).astype(float)
    bins, cuts, wide = _rdt_native._hist_bins(X, np.arange(3000, dtype=np.uint32), 1000)
    assert wide and bins.max() > 255 and all(len(c) == len(np.unique(X[:, j])) - 1 for j, c in enumerate(cuts))
    a = RDT(max_depth=3).fit(X, y)
    b = RDT(max_depth=3, tree_method="hist", max_bin=1000).fit(X, y)
    assert _trees_equal(a, b, X)


# --------------------------------------------------------------------------- 2. brute force on bin indices
def _bf_value(criterion, xb, y, w, b):
    """Weighted impurity decrease of the node split `bin <= b` (float64 NumPy)."""
    def imp(yy, ww):
        s = ww.sum()
        if criterion == "gini":
            pk = np.array([ww[yy == c].sum() for c in np.unique(yy)]) / s
            return 1 - np.sum(pk * pk)
        m = np.sum(ww * yy) / s
        return np.sum(ww * (yy - m) ** 2) / s

    left = xb <= b
    W = w.sum()
    return imp(y, w) - w[left].sum() / W * imp(y[left], w[left]) - w[~left].sum() / W * imp(y[~left], w[~left])


def _bf_best(criterion, xb, y, w, msl):
    best = None
    for b in np.unique(xb)[:-1]:
        nl = int((xb <= b).sum())
        if nl < msl or len(xb) - nl < msl:
            continue
        v = _bf_value(criterion, xb, y, w, b)
        if best is None or v > best[0]:
            best = (v, b)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_hist_tree_equals_bruteforce_on_bins(seed):
    """Guided: at each depth the native feature and each node's split bin must
    be optimal for the restricted tree on the bin indices (within rounding),
    and the threshold must be the cut after the split bin."""
    rng = np.random.default_rng(seed)
    n, p = int(rng.integers(40, 200)), int(rng.integers(1, 4))
    X = rng.standard_normal((n, p)) * rng.choice([1.0, 100.0])
    criterion = "gini" if seed % 2 else "mse"
    y = (X[:, 0] + rng.normal(0, 1, n) > 0).astype(float) if criterion == "gini" else X[:, 0] + rng.normal(0, 1, n)
    w = rng.uniform(0.5, 2.0, n)
    max_bin, depth, msl = int(rng.choice([4, 8, 16])), 3, int(rng.choice([1, 3]))
    model = RDT(task="classification" if criterion == "gini" else "regression", criterion=criterion,
                max_depth=depth, min_samples_leaf=msl, tree_method="hist", max_bin=max_bin).fit(X, y, sample_weight=w)
    bins, cuts, _ = _rdt_native._hist_bins(X, np.arange(n, dtype=np.uint32), max_bin)
    tol = 1e-9
    level = [(model.root, np.arange(n))]
    committed = 0
    for d in range(depth):
        to_split = [(nd, idx) for nd, idx in level if len(idx) >= 2 and len(np.unique(y[idx])) > 1]
        if not to_split:
            break
        totals = []
        for f in range(p):
            t, any_valid = 0.0, False
            for _, idx in to_split:
                bst = _bf_best(criterion, bins[idx, f], y[idx], w[idx], msl)
                if bst is not None:
                    t += w[idx].sum() / w.sum() * bst[0]
                    any_valid = True
            totals.append(t if any_valid else -np.inf)
        if max(totals) == -np.inf:
            assert d not in model.depth_features_
            break
        f = model.depth_features_[d]
        committed += 1
        assert totals[f] >= max(totals) - tol * max(1.0, abs(max(totals)))
        nxt = []
        for nd, idx in to_split:
            bst = _bf_best(criterion, bins[idx, f], y[idx], w[idx], msl)
            if bst is None:
                assert nd.is_leaf
                continue
            assert not nd.is_leaf and nd.split_feature == f
            got = _bf_value(criterion, bins[idx, f], y[idx], w[idx], nd.split_bin)
            assert got >= bst[0] - tol * max(1.0, abs(bst[0]))
            left = bins[idx, f] <= nd.split_bin
            # the threshold lies between the node's two bins and routes every row of X like its bin
            lo, hi = bins[idx[left], f].max(), bins[idx[~left], f].min()
            assert lo <= nd.split_bin < hi
            assert np.array_equal(X[:, f] <= nd.split_threshold, bins[:, f] <= nd.split_bin)
            nxt += [(nd.left, idx[left]), (nd.right, idx[~left])]
        level = nxt
    assert committed >= 1


# --------------------------------------------------------------------------- 3. routing
def test_routing_consistent_for_every_row_and_node():
    rng = np.random.default_rng(0)
    n = 2000
    X = np.column_stack([rng.standard_normal(n), rng.integers(0, 30, n) / 3.0, rng.exponential(1, n)])
    y = (X[:, 0] + X[:, 1] / 5 + rng.normal(0, 0.5, n) > 1).astype(float)
    w = rng.uniform(0, 1, n)
    w[rng.random(n) < 0.2] = 0.0  # zero-weight rows: binned with cuts they did not shape
    m = RDT(max_depth=5, tree_method="hist", max_bin=16).fit(X, y, sample_weight=w)
    bins, cuts, _ = _rdt_native._hist_bins(X, np.flatnonzero(w > 0).astype(np.uint32), 16)
    stack = [(m.root, np.flatnonzero(w > 0))]
    n_splits = 0
    while stack:
        node, idx = stack.pop()
        assert np.array_equal(np.sort(node.indices), idx)
        if node.is_leaf:
            continue
        n_splits += 1
        f = node.split_feature
        # every row of X, not just the node's: bin <= split_bin <=> x <= threshold
        assert np.array_equal(bins[:, f] <= node.split_bin, X[:, f] <= node.split_threshold)
        left = X[idx, f] <= node.split_threshold
        stack += [(node.left, idx[left]), (node.right, idx[~left])]
    assert n_splits > 5


def test_gb_training_scores_follow_the_raw_thresholds():
    # train_score_ is computed from scores updated through the bins; recomputing
    # it from staged predictions on X (raw thresholds) must give the same loss.
    X, y, w = _ens_problem("binary", 11, n=600)
    X = X + np.random.default_rng(1).standard_normal(X.shape) * 0.01  # many distinct values: quantile bins
    for split in ("friedman_mse", "newton"):
        m = GradientBoostingRDT(n_estimators=6, max_depth=3, tree_method="hist", max_bin=12, random_state=0,
                                split_criterion=split).fit(X, y, sample_weight=w)
        for r, F in enumerate(m.staged_decision_function(X)):
            p = np.clip(1 / (1 + np.exp(-F)), 1e-12, 1 - 1e-12)
            loss = np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w)
            np.testing.assert_allclose(loss, m.train_score_[r], rtol=1e-12)
        stages = list(m.staged_predict_proba(X))
        assert np.array_equal(stages[-1], m.predict_proba(X))


# --------------------------------------------------------------------------- 4. binning
def test_binning_rules():
    rng = np.random.default_rng(0)
    n = 10000
    X = np.column_stack([
        rng.integers(0, 50, n).astype(float),            # 50 values: one bin each
        rng.standard_normal(n),                         # continuous: equal-count bins
        np.where(rng.random(n) < 0.6, 0.0, rng.standard_normal(n)),  # a heavy tie
        np.full(n, 3.0),                                # constant: one bin
    ])
    bins, cuts, wide = _rdt_native._hist_bins(X, np.arange(n, dtype=np.uint32), 64)
    assert not wide and bins.dtype == np.uint16
    assert len(cuts[0]) == 49 and np.array_equal(np.sort(np.unique(bins[:, 0])), np.arange(50))
    counts = np.bincount(bins[:, 1])
    assert len(counts) == 64 and counts.min() > 0.8 * n / 64 and counts.max() < 1.2 * n / 64
    assert 1 < len(cuts[2]) + 1 < 64 and np.bincount(bins[:, 2]).max() >= 0.6 * n
    assert len(cuts[3]) == 0 and np.all(bins[:, 3] == 0)
    for j in range(4):  # x <= cut[b] <=> bin <= b, for every row
        for b, c in enumerate(cuts[j]):
            assert np.array_equal(X[:, j] <= c, bins[:, j] <= b)
    # rows outside the cut rows are binned with the same cuts
    sub = np.arange(0, n, 7, dtype=np.uint32)
    b2, c2, _ = _rdt_native._hist_bins(X, sub, 64)
    for j in range(4):
        assert np.array_equal(b2[:, j], np.searchsorted(c2[j], X[:, j], side="left"))


def test_validation():
    with pytest.raises(ValueError, match="tree_method"):
        RDT(tree_method="approx")
    for bad in (1, 65537, 2.5, True):
        with pytest.raises(ValueError, match="max_bin"):
            GradientBoostingRDT(tree_method="hist", max_bin=bad)
    with pytest.raises(NotImplementedError):
        RDT(task="regression", criterion="mae", tree_method="hist")
    with pytest.raises(NotImplementedError):
        RandomForestRDT(task="regression", criterion="mae", tree_method="hist")
    with pytest.raises(ValueError, match="mae"):
        _rdt_native.RDTModel("regression", "mae", 3, 2, 1, tree_method="hist").fit_hist(np.zeros((4, 1)), np.zeros(4))
    with pytest.raises(ValueError, match="float32 or float64"):
        _rdt_native.RDTModel("regression", "mse", 3, 2, 1, tree_method="hist").fit_hist(
            np.zeros((4, 1), dtype=np.int64), np.zeros(4))


# --------------------------------------------------------------------------- 5. inputs
def _fit_all(X, y):
    return [
        RDT(max_depth=4, tree_method="hist", max_bin=32).fit(X, y),
        RandomForestRDT(n_estimators=4, max_depth=3, random_state=0, tree_method="hist", max_bin=32).fit(X, y),
        GradientBoostingRDT(n_estimators=5, max_depth=3, random_state=0, tree_method="hist", max_bin=32).fit(X, y),
    ]


def _same_models(ms_a, ms_b, X):
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    for a, b in zip(ms_a, ms_b):
        assert np.array_equal(a.predict_proba(Xc), b.predict_proba(Xc))
        assert a.depth_features_ == b.depth_features_


def test_float32_and_layouts_equal_float64_c():
    rng = np.random.default_rng(0)
    X32 = rng.standard_normal((500, 6)).astype(np.float32)
    y = (X32[:, 0] + X32[:, 1] > 0).astype(float)
    ref = _fit_all(np.ascontiguousarray(X32, dtype=np.float64), y)
    _same_models(ref, _fit_all(X32, y), X32)
    _same_models(ref, _fit_all(np.asfortranarray(X32), y), X32)
    _same_models(ref, _fit_all(np.asfortranarray(X32.astype(np.float64)), y), X32)
    big = np.zeros((1000, 12), dtype=np.float32)
    big[::2, ::2] = X32
    _same_models(ref, _fit_all(big[::2, ::2], y), X32)  # strided view
    big_c = np.ascontiguousarray(X32[::-1].astype(np.float64))
    _same_models(_fit_all(big_c, y[::-1].copy()), _fit_all(X32.astype(np.float64)[::-1], y[::-1].copy()), big_c)


def test_memmap_input(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5))
    y = (X[:, 0] - X[:, 2] > 0).astype(float)
    ref = _fit_all(X, y)
    for dtype, order in ((np.float64, "C"), (np.float32, "F")):
        path = tmp_path / f"x_{np.dtype(dtype).name}_{order}.dat"
        mm = np.memmap(path, dtype=dtype, mode="w+", shape=X.shape, order=order)
        mm[:] = X
        mm.flush()
        ro = np.memmap(path, dtype=dtype, mode="r", shape=X.shape, order=order)
        got = _fit_all(ro, y)
        if dtype == np.float64:
            _same_models(ref, got, X)
        else:
            _same_models(_fit_all(X.astype(np.float32).astype(np.float64), y), got, X.astype(np.float32))


_FIT_MEMORY_SCRIPT = textwrap.dedent("""
    import resource, sys
    import numpy as np
    from rdt.ensemble import GradientBoostingRDT, RandomForestRDT

    def peak():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024

    kind, layout = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(0)
    n, p = 20000, 1000
    dtype = np.float32 if layout == "f32F" else np.float64
    X = np.empty((n, p), dtype=dtype, order="F" if layout == "f32F" else "C")
    for i in range(0, n, 500):
        X[i:i + 500] = rng.standard_normal((500, p))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    if kind == "gb":
        m = GradientBoostingRDT(n_estimators=2, max_depth=2, tree_method="hist", random_state=0, n_jobs=2)
    else:
        m = RandomForestRDT(n_estimators=2, max_depth=2, max_features=None, tree_method="hist", random_state=0)
    m.fit(X[:50], y[:50])  # warm up allocators and imports
    before = peak()
    m.fit(X, y)
    print(peak() - before, X.nbytes)
""")


@pytest.mark.parametrize("kind,layout", [("gb", "f64C"), ("rf", "f32F")])
def test_hist_fit_does_not_copy_X(kind, layout):
    pytest.importorskip("resource")
    out = subprocess.run([sys.executable, "-c", _FIT_MEMORY_SCRIPT, kind, layout],
                         capture_output=True, text=True, check=True).stdout.split()
    growth, x_bytes = int(out[0]), int(out[1])
    bins = 20000 * 1000  # uint8 bins: the one per-element allocation hist fit needs
    # A float64 copy of X alone would be x_bytes (float64) or 2 * x_bytes (float32); measured
    # growth is the bins plus ~16-20 MiB (cuts / bin extents of 1000 features, search state).
    assert growth < bins + 0.5 * x_bytes, f"fit grew peak RSS by {growth / 2**20:.0f} MiB (X is {x_bytes / 2**20:.0f} MiB)"


# --------------------------------------------------------------------------- 6. threads, staged
@pytest.mark.parametrize("name", ["gb-binary", "gb-multi", "gb-newton", "rf"])
def test_n_jobs_does_not_change_the_result(name):
    make = dict(ENSEMBLES)[name]
    kind = "multi" if "multi" in name else "binary"
    X, y, w = _ens_problem(kind, 3, n=500, p=12)
    X = X + np.random.default_rng(2).standard_normal(X.shape) * 0.05
    models = [make("hist", n_jobs=j, max_bin=24).fit(X, y, sample_weight=w) for j in (1, 2, 4)]
    for m in models[1:]:
        assert np.array_equal(models[0].predict_proba(X), m.predict_proba(X))
        assert np.array_equal(models[0].feature_importances_, m.feature_importances_)
    r = [RDT(max_depth=4, tree_method="hist", max_bin=24, n_jobs=j).fit(X, y, sample_weight=w) for j in (1, 4)]
    assert np.array_equal(r[0].predict_proba(X), r[1].predict_proba(X))
    assert r[0].depth_improvements_ == r[1].depth_improvements_
