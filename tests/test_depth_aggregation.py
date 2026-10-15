"""Per-depth feature scoring weights each node by its share of the training
weight (scikit-learn's weighted impurity decrease N_t/N * delta_t).

The per-node improvement is a within-node quantity, so summing it unweighted
across a depth let a 10-sample node count as much as a 10,000-sample one.
References used here:
- a pure-numpy brute-force replay of the restricted algorithm (size-weighted);
- scikit-learn: a max_depth=1 RDT is a CART stump, and a one-feature RDT of
  any depth is a CART tree (every node picks its own threshold on the only
  feature), so thresholds, leaves and per-depth importances must match.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdt import RDT
from rdt.ensemble import GradientBoostingRDT, RandomForestRDT


# ---------------------------------------------------------------------------
# numpy reference
# ---------------------------------------------------------------------------

def _impurity(crit, y, w):
    W = w.sum()
    if W <= 0:
        return 0.0
    if crit in ("gini", "entropy"):
        p = np.array([w[y == c].sum() for c in np.unique(y)]) / W
        if crit == "gini":
            return 1.0 - np.sum(p * p)
        return float(-np.sum(p * np.log2(p + 1e-10)))
    if crit == "mse":
        m = np.sum(w * y) / W
        return float(np.sum(w * (y - m) ** 2) / W)
    # mae: weighted median as defined by the library (average at an exact half)
    o = np.argsort(y, kind="stable")
    ys, ws = y[o], w[o]
    half, cum, med = W / 2.0, 0.0, ys[-1]
    for i in range(len(ys)):
        cum += ws[i]
        if cum > half:
            med = ys[i]
            break
        if cum == half:
            med = (ys[i] + ys[i + 1]) / 2.0 if i + 1 < len(ys) else ys[i]
            break
    return float(np.sum(w * np.abs(y - med)) / W)


def _node_best_split(crit, x, y, w, min_samples_leaf):
    """Best (threshold, improvement) of one node on one feature, by brute
    force over every boundary between distinct values (or None)."""
    xs = np.unique(x)
    if len(xs) < 2:
        return None
    parent = _impurity(crit, y, w)
    W = w.sum()
    best = None
    for a, b in zip(xs[:-1], xs[1:]):
        left = x <= a
        nl = int(left.sum())
        if nl < min_samples_leaf or len(x) - nl < min_samples_leaf:
            continue
        wl, wr = w[left].sum(), w[~left].sum()
        imp = parent - wl / W * _impurity(crit, y[left], w[left]) - wr / W * _impurity(crit, y[~left], w[~left])
        if best is None or imp > best[1]:
            best = ((a + b) / 2.0, imp)
    return best


def _levels(root):
    level = [root]
    while level:
        yield level
        level = [c for n in level if not n.is_leaf for c in (n.left, n.right)]


def _replay_and_check(model, X, y, w, crit, max_depth, min_samples_split=2, min_samples_leaf=1,
                      min_impurity_decrease=0.0, tol=1e-9):
    """Walk the fitted tree depth by depth and check every decision against
    brute force: the depth's feature maximises the size-weighted total, each
    node's threshold is its best one for that feature, nodes that feature
    cannot split are leaves, depth_improvements_ equals the weighted total,
    and leaves hold the (weighted) class counts / means.
    Returns the per-depth reference argmax features (for exact-match stats)."""
    W = w.sum()
    ref_argmax = []
    for depth, level in enumerate(_levels(model.root)):
        splittable = [n for n in level
                      if depth < max_depth and len(n.indices) >= min_samples_split
                      and len(np.unique(y[np.asarray(n.indices)])) > 1]
        for n in level:
            if n not in splittable:
                assert n.is_leaf
        if not splittable:
            assert depth not in model.depth_features_
            break
        totals = np.full(X.shape[1], -np.inf)
        per_node = {}
        for f in range(X.shape[1]):
            tot, found = 0.0, False
            for j, n in enumerate(splittable):
                idx = np.asarray(n.indices)
                r = _node_best_split(crit, X[idx, f], y[idx], w[idx], min_samples_leaf)
                per_node[(f, j)] = r
                if r is not None:
                    tot += w[idx].sum() / W * r[1]
                    found = True
            if found:
                totals[f] = tot
        best_total = totals.max()
        ref_argmax.append(int(np.argmax(totals)) if np.isfinite(best_total) else None)
        scale = max(1.0, abs(best_total)) if np.isfinite(best_total) else 1.0
        if depth not in model.depth_features_:
            # stopped: nothing valid, or the best depth total is below the gate
            assert (not np.isfinite(best_total)) or best_total < min_impurity_decrease + tol * scale
            for n in splittable:
                assert n.is_leaf
            break
        f = model.depth_features_[depth]
        assert totals[f] >= best_total - tol * scale, (depth, f, totals)
        assert model.depth_improvements_[depth] == pytest.approx(totals[f], rel=1e-7, abs=1e-12)
        assert totals[f] >= min_impurity_decrease - tol * scale
        for j, n in enumerate(splittable):
            r = per_node[(f, j)]
            if r is None:
                assert n.is_leaf  # frozen: documented behaviour
                continue
            assert not n.is_leaf and n.split_feature == f
            idx = np.asarray(n.indices)
            left = X[idx, f] <= n.split_threshold
            imp_at_thr = (_impurity(crit, y[idx], w[idx])
                          - w[idx][left].sum() / w[idx].sum() * _impurity(crit, y[idx][left], w[idx][left])
                          - w[idx][~left].sum() / w[idx].sum() * _impurity(crit, y[idx][~left], w[idx][~left]))
            assert imp_at_thr >= r[1] - tol * max(1.0, abs(r[1]))
    # leaves
    for level in _levels(model.root):
        for n in level:
            if not n.is_leaf:
                continue
            idx = np.asarray(n.indices)
            if model.task == "classification":
                expected = [w[idx][y[idx] == c].sum() for c in model.classes_]
                np.testing.assert_allclose(n.class_counts, expected, rtol=1e-12)
            else:
                assert n.value == pytest.approx(np.sum(w[idx] * y[idx]) / w[idx].sum(), rel=1e-12)
    return ref_argmax


def _random_problem(seed, task):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([6, 15, 40, 80]))
    p = int(rng.choice([1, 2, 4]))
    kind = seed % 3
    if kind == 0:
        X = rng.normal(size=(n, p))
    elif kind == 1:
        X = rng.integers(0, 4, size=(n, p)).astype(float)
    else:
        X = np.round(rng.normal(size=(n, p)), 1)
    signal = X[:, 0] - 0.7 * X[:, -1] + rng.normal(size=n)
    if task == "classification":
        k = int(rng.choice([2, 3]))
        y = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, k + 1)[1:-1])).astype(float)
    else:
        y = signal * 2.0 + 1.0
    w = rng.uniform(0.2, 3.0, size=n) if seed % 2 else np.ones(n)
    return X, y, w



def test_matches_bruteforce_reference_on_random_small_datasets():
    # every decision (feature argmax with node-share weighting, per-node
    # thresholds, frozen nodes, depth_improvements_, the min_impurity_decrease
    # gate, leaf values) checked against a pure-numpy replay
    for crit, n_seeds in (("gini", 25), ("entropy", 15), ("mse", 25), ("mae", 10)):
        task = "classification" if crit in ("gini", "entropy") else "regression"
        exact = 0
        for seed in range(n_seeds):
            X, y, w = _random_problem(seed, task)
            if crit == "mae" and seed % 2:
                w = np.ones(len(y))  # the library's weighted median has its own tie rules
            msl = 1 + seed % 3
            mid = (0.0, 0.0, 0.01, 0.05)[seed % 4] if task == "classification" else 0.0
            m = RDT(task=task, criterion=crit, max_depth=4, min_samples_leaf=msl,
                    min_impurity_decrease=mid).fit(X, y, sample_weight=w)
            ref = _replay_and_check(m, X, y, w, crit, max_depth=4, min_samples_leaf=msl,
                                    min_impurity_decrease=mid)
            lib = [m.depth_features_[d] for d in sorted(m.depth_features_)]
            exact += lib == ref[:len(lib)]
        # exact feature-index agreement except where the reference sees an
        # exact tie (frequent for MAE on tiny nodes); every choice was already
        # checked to be optimal within rounding above
        assert exact >= 0.7 * n_seeds, (crit, exact)


def test_one_feature_tree_equals_sklearn_cart():
    # With one feature an RDT is a CART tree: thresholds, probabilities /
    # predictions and the per-depth weighted impurity decreases must match.
    tree = pytest.importorskip("sklearn.tree")
    for criterion in ("gini", "entropy", "mse"):
        task = "regression" if criterion == "mse" else "classification"
        for seed in range(6):
            rng = np.random.default_rng(seed)
            n = int(rng.choice([50, 300, 2000]))
            X = _f32_grid(rng, n, 1)
            signal = np.sin(X[:, 0] / 15.0) + 0.5 * rng.normal(size=n)
            y = (signal > np.quantile(signal, 0.8)).astype(float) if task == "classification" else 100.0 + 10.0 * signal
            w = rng.uniform(0.1, 3.0, size=n) if seed % 2 else None
            if task == "classification":
                sk = tree.DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0)
            else:
                sk = tree.DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=0)
            sk.fit(X, y, sample_weight=w)
            m = RDT(task=task, criterion=criterion, max_depth=3).fit(X, y, sample_weight=w)
            pairs = _walk_pair(m.root, sk.tree_, 0, [])
            assert len(pairs) == sk.tree_.node_count
            for node, i in pairs:
                if not node.is_leaf:
                    assert node.split_threshold == sk.tree_.threshold[i]
            if task == "classification":
                np.testing.assert_allclose(m.predict_proba(X), sk.predict_proba(X), rtol=1e-12, atol=1e-14)
            else:
                np.testing.assert_allclose(m.predict(X), sk.predict(X), rtol=1e-12)
            ref = _sk_depth_importances(sk, n if w is None else w.sum())
            assert set(ref) == set(m.depth_improvements_)
            for d, v in ref.items():
                assert m.depth_improvements_[d] == pytest.approx(v, rel=1e-8 if criterion == "entropy" else 1e-9)


def test_min_impurity_decrease_uses_sklearns_epsilon():
    # sklearn splits unless improvement + EPSILON < min_impurity_decrease.
    # The root improvement here is exactly 0.5; a threshold one ulp above it
    # still splits in sklearn (and here), and one well above does not.
    tree = pytest.importorskip("sklearn.tree")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    for mid, splits in ((0.5, True), (0.5 + 2.0**-53, True), (0.5 + 1e-9, False)):
        sk = tree.DecisionTreeClassifier(max_depth=1, min_impurity_decrease=mid).fit(X, y)
        assert (sk.tree_.node_count == 3) == splits
        assert (not RDT(max_depth=1, min_impurity_decrease=mid).fit(X, y).root.is_leaf) == splits


def _f32_grid(rng, n, p):
    # values on a 1/16 grid: exactly representable in float32 (sklearn casts X)
    # and further apart than sklearn's FEATURE_THRESHOLD
    return rng.integers(0, 1600, size=(n, p)) / 16.0


def _sk_depth_importances(sk_tree, W):
    t = sk_tree.tree_
    depth = np.zeros(t.node_count, dtype=int)
    out = {}
    for i in range(t.node_count):
        l, r = t.children_left[i], t.children_right[i]
        if l == -1:
            continue
        depth[l] = depth[r] = depth[i] + 1
        wn, wl, wr = t.weighted_n_node_samples[i], t.weighted_n_node_samples[l], t.weighted_n_node_samples[r]
        dec = wn / W * (t.impurity[i] - wl / wn * t.impurity[l] - wr / wn * t.impurity[r])
        out[depth[i]] = out.get(depth[i], 0.0) + dec
    return out


def _walk_pair(node, t, i, out):
    out.append((node, i))
    if not node.is_leaf:
        assert t.children_left[i] != -1
        _walk_pair(node.left, t, t.children_left[i], out)
        _walk_pair(node.right, t, t.children_right[i], out)
    else:
        assert t.children_left[i] == -1
    return out



def test_tiny_node_no_longer_wins_the_depth():
    # 10 rows in one root child, 10,000 in the other. Feature A separates the
    # small node perfectly (huge within-node gain), feature B the big one.
    rng = np.random.default_rng(0)
    nb, ns = 10000, 10
    f0 = np.r_[np.zeros(nb), np.ones(ns)]
    A = rng.normal(size=nb + ns)
    B = np.r_[rng.normal(size=nb), np.zeros(ns)]
    y = np.r_[5 + 5 * np.sign(B[:nb]) + rng.normal(0, 0.1, nb), 1000 + 50 * np.sign(A[nb:])]
    X = np.column_stack([f0, A, B])
    m = RDT(task="regression", max_depth=2).fit(X, y)
    assert m.depth_features_ == {0: 0, 1: 2}
    sse = np.sum((y - m.predict(X)) ** 2)
    assert sse < 30000  # was ~250,000 when depth 1 picked A


def test_low_cardinality_flag_no_longer_freezes_the_big_node():
    rng = np.random.default_rng(0)
    nb, ns = 10000, 10
    f0 = np.r_[np.zeros(nb), np.ones(ns)]
    flag = np.r_[np.zeros(nb), (np.arange(ns) % 2).astype(float)]
    B = np.r_[rng.normal(size=nb), np.zeros(ns)]
    y = np.r_[5 + 5 * np.sign(B[:nb]) + rng.normal(0, 0.1, nb), 1000 + 50 * (2 * flag[nb:] - 1)]
    X = np.column_stack([f0, flag, B])
    m = RDT(task="regression", max_depth=3).fit(X, y)
    assert m.depth_features_[1] == 2
    assert not m.root.left.is_leaf      # the 10,000-row node is split
    assert m.root.right.is_leaf         # B is constant on the 10-row node: frozen (documented)


def _leaf_nodes(node):
    return [n for lvl in _levels(node) for n in lvl if n.is_leaf]



def _telescoping_check(X, y_clf, y_reg, w, depth):
    n = len(y_clf)
    for task, crit, y in (("classification", "gini", y_clf), ("classification", "entropy", y_clf),
                          ("regression", "mse", y_reg)):
        for sw in (None, w):
            ww = np.ones(n) if sw is None else sw
            m = RDT(task=task, criterion=crit, max_depth=depth).fit(X, y, sample_weight=sw)
            W = ww.sum()
            total_decrease = _impurity(crit, y, ww) - sum(
                ww[np.asarray(l.indices)].sum() / W
                * _impurity(crit, y[np.asarray(l.indices)], ww[np.asarray(l.indices)])
                for l in _leaf_nodes(m.root))
            assert sum(m.depth_improvements_.values()) == pytest.approx(total_decrease, rel=1e-7, abs=1e-10)


def _shape_problem(n, p):
    rng = np.random.default_rng(n * 3 + p)
    X = rng.normal(size=(n, p))
    if p > 1:
        X[:, 1] = rng.integers(0, 2, size=n)
    y_clf = (X[:, 0] + rng.normal(size=n) > 1.6).astype(float)
    y_clf[:2] = [0.0, 1.0]
    return X, y_clf, 50.0 + X[:, 0] + rng.normal(size=n), rng.uniform(0.1, 2.0, size=n)


def test_depth_improvements_telescope_to_total_impurity_decrease():
    # sum of depth_improvements_ == root impurity - weighted leaf impurity
    X, y_clf, y_reg, w = _shape_problem(1000, 5)
    _telescoping_check(X, y_clf, y_reg, w, depth=4)


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 10, 1000, 20000])
@pytest.mark.parametrize("p", [1, 5, 50])
def test_depth_improvements_telescope_across_shapes(n, p):
    X, y_clf, y_reg, w = _shape_problem(n, p)
    _telescoping_check(X, y_clf, y_reg, w, depth=4 if n * p <= 50000 else 2)


def test_ensemble_importances_use_weighted_improvements():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 1.5)).astype(float)
    rf = RandomForestRDT(n_estimators=10, max_depth=3, max_features=None, bootstrap=False, random_state=0).fit(X, y)
    tree = rf.estimators_[0]
    imp = np.zeros(4)
    for d, f in tree.depth_features_.items():
        imp[f] += tree.depth_improvements_[d]
    np.testing.assert_allclose(rf.feature_importances_, imp / imp.sum(), rtol=1e-12)
    # and each tree's per-depth values are sklearn-style weighted decreases
    leaves = _leaf_nodes(tree.root)
    ww = np.ones(n)
    total_decrease = _impurity("gini", y, ww) - sum(
        len(l.indices) / n * _impurity("gini", y[np.asarray(l.indices)], ww[np.asarray(l.indices)]) for l in leaves)
    assert sum(tree.depth_improvements_.values()) == pytest.approx(total_decrease, rel=1e-9)
    gb = GradientBoostingRDT(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    assert gb.feature_importances_.sum() == pytest.approx(1.0)
    assert np.argmax(gb.feature_importances_) == 0

