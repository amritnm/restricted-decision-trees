"""Split-search correctness.

1. A brute-force pure-NumPy reference of the documented algorithm (per depth:
   score every feature on every node, sum the per-node best improvements,
   commit the best feature) checked against the native tree on many small
   random datasets -- ties, weights (including zeros), every criterion and
   stopping rule. The check is "guided": at each depth it verifies that the
   native choice is optimal (and is *the* reference choice whenever the best
   candidate is not within a rounding-level tolerance of the runner-up), then
   follows the native tree down, so legitimate near-tie choices cannot make
   the comparison drift.

2. The presorted search (default) against the per-(node, feature) sort search
   it replaced (``presort=False`` on the native classes), bit for bit, on
   inputs where both are guaranteed to perform the same floating-point
   operations: no tied feature values, nodes small enough that std::sort is an
   insertion sort (stable) on every major standard library, or sums that are
   exact (integer-valued targets and weights). Outside those, the two can only
   differ by summation order inside a run of tied values.
"""

import numpy as np
import pytest

from rdt import RDT, _rdt_native
from rdt.ensemble import GradientBoostingRDT, RandomForestRDT

TOL = 1e-9


# --------------------------------------------------------------------------- reference
def _impurity(criterion, y, w):
    W = float(np.sum(w))
    if W <= 0.0:
        return 0.0
    if criterion in ("gini", "entropy"):
        p = np.array([np.sum(w[y == c]) for c in np.unique(y)]) / W
        if criterion == "gini":
            return float(1.0 - np.sum(p * p))
        return float(-np.sum(p * np.log2(p + 1e-10)))
    if criterion == "mse":
        mean = np.sum(w * y) / W
        return float(np.sum(w * y * y) / W - mean * mean)
    # mae: weighted median as in the core (average the two middle values on an exact half split)
    order = np.argsort(y, kind="stable")
    ys, ws = y[order], w[order]
    cum = np.cumsum(ws)
    half = W / 2.0
    med = ys[-1]
    for i in range(len(ys)):
        if cum[i] > half:
            med = ys[i]
            break
        if cum[i] == half:
            med = (ys[i] + ys[i + 1]) / 2.0 if i + 1 < len(ys) else ys[i]
            break
    return float(np.sum(ws * np.abs(ys - med)) / W)


def _node_candidates(criterion, x, y, w, min_samples_leaf):
    """All valid (improvement, lo_value, hi_value) boundaries of one node, ascending x."""
    parent = _impurity(criterion, y, w)
    W = float(np.sum(w))
    vals = np.unique(x)
    out = []
    for a, b in zip(vals[:-1], vals[1:]):
        left = x <= a
        nl = int(left.sum())
        if nl < min_samples_leaf or len(x) - nl < min_samples_leaf:
            continue
        wl, wr = float(np.sum(w[left])), float(np.sum(w[~left]))
        weighted = (wl / W) * _impurity(criterion, y[left], w[left]) + \
                   (wr / W) * _impurity(criterion, y[~left], w[~left])
        out.append((parent - weighted, a, b))
    return out


def _close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def check_against_reference(model, X, y, w, criterion, max_depth, min_samples_split,
                            min_samples_leaf, min_impurity_decrease):
    root = model.root
    n, p = X.shape
    level = [(root, np.arange(n))]
    leaves = []
    for depth in range(max_depth):
        to_split = []
        for node, idx in level:
            if len(idx) >= min_samples_split and len(np.unique(y[idx])) > 1:
                to_split.append((node, idx))
            else:
                leaves.append((node, idx))
        if not to_split:
            level = []
            break
        totals, per_node = [], []
        for f in range(p):
            bests, total, valid = [], 0.0, False
            for _, idx in to_split:
                cands = _node_candidates(criterion, X[idx, f], y[idx], w[idx], min_samples_leaf)
                if cands:
                    best = max(c[0] for c in cands)
                    bests.append((best, cands))
                    total += best
                    valid = True
                else:
                    bests.append(None)
            totals.append(total if valid else None)
            per_node.append(bests)
        valid_totals = [t for t in totals if t is not None]
        if not valid_totals or max(valid_totals) < min_impurity_decrease - TOL:
            assert depth not in model.depth_features_
            leaves.extend(to_split)
            level = []
            break
        best_total = max(valid_totals)
        if depth not in model.depth_features_:
            # only legitimate if the best total sits at the min_impurity_decrease boundary
            assert _close(best_total, min_impurity_decrease)
            leaves.extend(to_split)
            level = []
            break
        f = model.depth_features_[depth]
        assert totals[f] is not None and totals[f] >= best_total - TOL * max(1.0, abs(best_total))
        assert _close(model.depth_improvements_[depth], totals[f])
        near = [g for g, t in enumerate(totals) if t is not None and t >= best_total - TOL * max(1.0, abs(best_total))]
        if len(near) == 1:
            assert f == near[0]
        next_level = []
        for (node, idx), nb in zip(to_split, per_node[f]):
            if nb is None:
                assert node.is_leaf
                leaves.append((node, idx))
                continue
            node_best, cands = nb
            if node.is_leaf:
                # the scored split may leave one side empty once applied at the midpoint
                thr = None
                for imp, a, b in cands:
                    if imp >= node_best - TOL * max(1.0, abs(node_best)):
                        thr = (a + b) / 2.0
                        break
                xs = X[idx, f]
                assert thr is not None and ((xs <= thr).all() or (xs > thr).all())
                leaves.append((node, idx))
                continue
            assert node.split_feature == f
            thr = node.split_threshold
            match = [c for c in cands if (c[1] + c[2]) / 2.0 == thr]
            assert match, "threshold is not a midpoint of adjacent distinct values"
            assert match[0][0] >= node_best - TOL * max(1.0, abs(node_best))
            near_t = [c for c in cands if c[0] >= node_best - TOL * max(1.0, abs(node_best))]
            if len(near_t) == 1:
                assert (near_t[0][1] + near_t[0][2]) / 2.0 == thr
            left = X[idx, f] <= thr
            next_level += [(node.left, idx[left]), (node.right, idx[~left])]
        level = next_level
    leaves.extend(level)

    for node, idx in leaves:
        assert node.is_leaf
        assert node.n_samples == len(idx)
        np.testing.assert_array_equal(np.asarray(node.indices), idx)
        yy, ww = y[idx], w[idx]
        if criterion in ("gini", "entropy"):
            labels, first = np.unique(yy, return_index=True)
            labels = labels[np.argsort(first)]  # first-encountered order
            counts = [np.sum(ww[yy == c]) for c in labels]
            best = 0
            for i in range(1, len(counts)):
                if counts[i] > counts[best]:
                    best = i
            assert node.value == labels[best]
            expect = np.zeros(len(model.classes_))
            for c, cnt in zip(labels, counts):
                expect[np.searchsorted(model.classes_, c)] = cnt
            np.testing.assert_allclose(node.class_counts, expect, rtol=1e-12, atol=1e-12)
        else:
            W = np.sum(ww)
            expect = np.sum(ww * yy) / W if W > 0 else 0.0
            assert abs(node.value - expect) <= 1e-9 * max(1.0, abs(expect))


def _random_problem(rng, task):
    n = int(rng.integers(2, 41))
    p = int(rng.integers(1, 5))
    X = np.empty((n, p))
    for j in range(p):
        kind = rng.integers(0, 4)
        if kind == 0:
            X[:, j] = rng.normal(size=n)
        elif kind == 1:
            X[:, j] = rng.integers(0, 3, size=n)
        elif kind == 2:
            X[:, j] = rng.integers(0, 2, size=n) * 2.0 - 1.0
        else:
            X[:, j] = np.round(rng.normal(size=n), 1)
    if task == "classification":
        y = rng.integers(0, int(rng.integers(2, 4)), size=n).astype(float)
    else:
        y = np.round(rng.normal(size=n) * 3, 2) if rng.uniform() < 0.5 else rng.normal(size=n)
    wk = rng.integers(0, 4)
    if wk == 0:
        w = np.ones(n)
    elif wk == 1:
        w = rng.uniform(0.2, 3.0, size=n)
    elif wk == 2:
        w = rng.integers(1, 4, size=n).astype(float)
    else:
        w = rng.uniform(0.2, 3.0, size=n)
        w[rng.uniform(size=n) < 0.25] = 0.0
    return X, y, w, wk != 0


@pytest.mark.parametrize("criterion", ["gini", "entropy", "mse", "mae"])
def test_native_tree_matches_bruteforce_reference(criterion):
    task = "classification" if criterion in ("gini", "entropy") else "regression"
    rng = np.random.default_rng({"gini": 1, "entropy": 2, "mse": 3, "mae": 4}[criterion])
    for _ in range(120):
        X, y, w, weighted = _random_problem(rng, task)
        params = dict(
            max_depth=int(rng.integers(1, 6)),
            min_samples_split=int(rng.choice([2, 2, 3, 6])),
            min_samples_leaf=int(rng.choice([1, 1, 2, 3])),
            min_impurity_decrease=float(rng.choice([0.0, 0.0, 0.01])),
        )
        model = RDT(task=task, criterion=criterion, **params).fit(X, y, w if weighted else None)
        check_against_reference(model, X, y, w, criterion, **params)


# --------------------------------------------------------------------------- presort vs per-node sort
def _walk(node, out):
    out.append((node.is_leaf, node.split_feature, node.split_threshold.hex(), node.value.hex(),
                tuple(c.hex() for c in node.class_counts), node.n_samples, tuple(node.indices)))
    if not node.is_leaf:
        _walk(node.left, out)
        _walk(node.right, out)
    return out


def _native_tree(presort, task, criterion, X, y, w, **kw):
    m = _rdt_native.RDTModel(task, criterion, kw.get("max_depth", 6), kw.get("min_samples_split", 2),
                             kw.get("min_samples_leaf", 1), kw.get("min_impurity_decrease", 0.0), presort=presort)
    m.fit(np.ascontiguousarray(X), np.ascontiguousarray(y), w)
    return m


def assert_same_tree(a, b, X):
    assert a.depth_features_ == b.depth_features_
    assert {k: v.hex() for k, v in a.depth_improvements_.items()} == \
           {k: v.hex() for k, v in b.depth_improvements_.items()}
    assert _walk(a.root, []) == _walk(b.root, [])
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def _tree_cases(rng, n, distinct):
    p = int(rng.integers(1, 6))
    if distinct:
        X = rng.normal(size=(n, p)) * rng.choice([1.0, 1e9])
    else:
        X = rng.integers(0, 3, size=(n, p)).astype(float)
        X[:, 0] = np.round(rng.normal(size=n), 1)
    w = [None, rng.uniform(0.1, 2.0, size=n), np.where(rng.uniform(size=n) < 0.2, 0.0, 1.5)][rng.integers(0, 3)]
    return X, w


@pytest.mark.parametrize("criterion", ["gini", "entropy", "mse"])
def test_presort_bitwise_equal_to_per_node_sort_without_ties(criterion):
    task = "classification" if criterion != "mse" else "regression"
    rng = np.random.default_rng(10)
    for _ in range(25):
        n = int(rng.integers(30, 600))
        X, w = _tree_cases(rng, n, distinct=True)
        y = (rng.integers(0, 4, size=n).astype(float) if task == "classification"
             else X[:, 0] * 2.5 + rng.normal(size=n))
        kw = dict(max_depth=int(rng.integers(1, 9)), min_samples_leaf=int(rng.choice([1, 5])))
        assert_same_tree(_native_tree(True, task, criterion, X, y, w, **kw),
                         _native_tree(False, task, criterion, X, y, w, **kw), X)


@pytest.mark.parametrize("criterion", ["gini", "entropy", "mse"])
def test_presort_bitwise_equal_to_per_node_sort_small_nodes_with_ties(criterion):
    # <= 16 rows: std::sort is an insertion sort (stable) in libc++, libstdc++ and MSVC,
    # so both searches see tied values in the same (row) order.
    task = "classification" if criterion != "mse" else "regression"
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(2, 17))
        X, w = _tree_cases(rng, n, distinct=False)
        y = (rng.integers(0, 3, size=n).astype(float) if task == "classification" else rng.normal(size=n))
        kw = dict(max_depth=int(rng.integers(1, 6)))
        assert_same_tree(_native_tree(True, task, criterion, X, y, w, **kw),
                         _native_tree(False, task, criterion, X, y, w, **kw), X)


def test_presort_bitwise_equal_to_per_node_sort_exact_sums_with_ties():
    # Integer targets and weights: every running sum is exact, so the order of tied rows
    # cannot matter and the two searches must agree bit for bit on heavily tied data.
    rng = np.random.default_rng(12)
    for _ in range(20):
        n = int(rng.integers(100, 2000))
        X, _ = _tree_cases(rng, n, distinct=False)
        y = rng.integers(-5, 6, size=n).astype(float)
        w = rng.integers(1, 4, size=n).astype(float) if rng.uniform() < 0.5 else None
        kw = dict(max_depth=int(rng.integers(1, 8)), min_samples_leaf=int(rng.choice([1, 10])))
        assert_same_tree(_native_tree(True, "regression", "mse", X, y, w, **kw),
                         _native_tree(False, "regression", "mse", X, y, w, **kw), X)


def _native_ensemble(cls, presort, X, y, w, **kw):
    """Fit an ensemble wrapper with its native model's presort switch forced."""
    native = {"RF": _rdt_native.RandomForestModel, "GB": _rdt_native.GradientBoostingModel}[cls]
    orig = getattr(_rdt_native, native.__name__)
    try:
        setattr(_rdt_native, native.__name__, lambda *a: orig(*a, presort=presort))
        est = (RandomForestRDT if cls == "RF" else GradientBoostingRDT)(**kw)
        return est.fit(X, y, w)
    finally:
        setattr(_rdt_native, native.__name__, orig)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ensembles_presort_bitwise_equal_to_per_node_sort_without_ties(task):
    rng = np.random.default_rng(13)
    for i in range(6):
        n, p = int(rng.integers(200, 1500)), int(rng.integers(2, 12))
        X = rng.normal(size=(n, p))
        y = (X[:, 0] + rng.normal(size=n) > 0).astype(float) if task == "classification" else X[:, 0] + rng.normal(size=n)
        if task == "classification" and i % 2:
            y += (X[:, 1] > 1.0)  # three classes
        w = rng.uniform(0.2, 2.0, size=n) if i % 3 == 0 else None
        Xte = rng.normal(size=(300, p))

        rf_kw = dict(task=task, n_estimators=4, max_depth=5, max_features=0.6, bootstrap=False,
                     random_state=i, n_jobs=2)
        a, b = _native_ensemble("RF", True, X, y, w, **rf_kw), _native_ensemble("RF", False, X, y, w, **rf_kw)
        for t, (ta, tb) in enumerate(zip(a.estimators_, b.estimators_)):
            assert_same_tree(ta, tb, np.ascontiguousarray(X[:, a._model.feature_subsets_[t]]))
        np.testing.assert_array_equal(a.predict(Xte), b.predict(Xte))
        np.testing.assert_array_equal(a.feature_importances_, b.feature_importances_)

        if task == "classification" and len(np.unique(y)) > 2 and i % 2:
            loss = "log_loss"
        else:
            loss = None
        gb_kw = dict(task=task, n_estimators=8, max_depth=4, subsample=0.7, colsample_bytree=0.6,
                     random_state=i, n_iter_no_change=3 if i % 2 else None)
        if loss:
            gb_kw["loss"] = loss
        a, b = _native_ensemble("GB", True, X, y, w, **gb_kw), _native_ensemble("GB", False, X, y, w, **gb_kw)
        np.testing.assert_array_equal(a.predict(Xte), b.predict(Xte))
        np.testing.assert_array_equal(a.train_score_, b.train_score_)
        np.testing.assert_array_equal(a.feature_importances_, b.feature_importances_)
        if task == "classification":
            np.testing.assert_array_equal(a.predict_proba(Xte), b.predict_proba(Xte))
