"""GradientBoostingRDT(split_criterion='newton'): XGBoost-style second-order trees.

1. A float64 brute-force NumPy reference of the restricted Newton tree
   (per depth: every feature, every node, every boundary; gain
   score(L) + score(R) - score(node); a node splits only if its best gain is
   >= gamma and > 1e-6 with both children's hessian sums >= min_child_weight;
   the depth's feature maximises the sum of (gain - gamma)) against the native
   tree, on small random problems, guided: at every depth the native choice
   must be optimal (and *the* reference choice whenever the runner-up is not
   within rounding), then the check follows the native tree. Gains, leaf
   values and depth gains agree to ~1e-12.
2. The presorted and per-node searches are bitwise identical in newton mode.
3. Parameter semantics and validation.

XGBoost parity lives in test_newton_xgboost.py.
"""

import numpy as np
import pytest

from rdt import _rdt_native
from rdt.ensemble import GradientBoostingRDT

TOL = 1e-12


# --------------------------------------------------------------------------- reference
def _T(G, alpha):
    return np.sign(G) * max(abs(G) - alpha, 0.0)


def _weight(G, H, p):
    if H <= 0:
        return 0.0
    w = -_T(G, p["alpha"]) / (H + p["lam"])
    if p["mds"] > 0 and abs(w) > p["mds"]:
        w = np.copysign(p["mds"], w)
    return w


def _score(G, H, p):
    if H <= 0:
        return 0.0
    if p["mds"] == 0:
        t = _T(G, p["alpha"])
        return t * t / (H + p["lam"])
    w = _weight(G, H, p)
    return -(2 * G * w + (H + p["lam"]) * w * w + 2 * p["alpha"] * abs(w))


def _best_split(x, g, h, p, msl):
    """(gain, a, b) of the node's best valid boundary (lowest on ties), or None."""
    G, H = g.sum(), h.sum()
    parent = _score(G, H, p)
    vals = np.unique(x)
    best = None
    for a, b in zip(vals[:-1], vals[1:]):
        left = x <= a
        nl = int(left.sum())
        if nl < msl or len(x) - nl < msl:
            continue
        GL, HL = g[left].sum(), h[left].sum()
        HR = H - HL
        if not (HL > 0 and HR > 0 and HL >= p["mcw"] and HR >= p["mcw"]):
            continue
        gain = _score(GL, HL, p) + _score(G - GL, HR, p) - parent
        if best is None or gain > best[0]:
            best = (gain, a, b)
    return best


def _guarded_mid(a, b):
    t = a / 2.0 + b / 2.0
    return t if a <= t < b else a


def _close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def check_newton_tree(model, X, g, h, p, max_depth, msl, mss):
    """Guided check of a fitted native newton RDTModel against the reference."""
    root = model.root
    level = [(root, np.arange(len(X)))]
    leaves = []
    for depth in range(max_depth):
        to_split = [(nd, idx) for nd, idx in level if len(idx) >= mss]
        leaves += [(nd, idx) for nd, idx in level if len(idx) < mss]
        if not to_split:
            level = []
            break
        totals, per_node = [], []
        for f in range(X.shape[1]):
            total, gains, splits = 0.0, 0.0, []
            any_valid = False
            for _, idx in to_split:
                b = _best_split(X[idx, f], g[idx], h[idx], p, msl)
                if b is not None and b[0] >= p["gamma"] and b[0] > 1e-6:
                    total += b[0] - p["gamma"]
                    gains += b[0]
                    any_valid = True
                    splits.append(b)
                else:
                    splits.append(None)
            totals.append((total, gains) if any_valid else None)
            per_node.append(splits)
        valid = [t[0] for t in totals if t is not None]
        if not valid:
            assert depth not in model.depth_features_
            leaves += to_split
            level = []
            break
        best_total = max(valid)
        f = model.depth_features_[depth]
        assert totals[f] is not None and totals[f][0] >= best_total - TOL * max(1.0, abs(best_total))
        runner_up = sorted(valid)[-2] if len(valid) > 1 else -np.inf
        if best_total - runner_up > 1e-9 * max(1.0, abs(best_total)):
            assert f == int(np.argmax([t[0] if t is not None else -np.inf for t in totals]))
        assert _close(model.depth_improvements_[depth], totals[f][1])
        nxt = []
        for (nd, idx), b in zip(to_split, per_node[f]):
            if b is None:
                assert nd.is_leaf
                leaves.append((nd, idx))
                continue
            assert not nd.is_leaf and nd.split_feature == f
            assert nd.split_threshold == _guarded_mid(b[1], b[2])
            left = X[idx, f] <= nd.split_threshold
            nxt += [(nd.left, idx[left]), (nd.right, idx[~left])]
        level = nxt
    leaves += level
    for nd, idx in leaves:
        assert nd.is_leaf
        assert _close(nd.value, _weight(g[idx].sum(), h[idx].sum(), p))


def _newton_tree(p, max_depth, msl=1, mss=2, presort=True):
    return _rdt_native.RDTModel("regression", "newton", max_depth, mss, msl, 0.0, presort=presort,
                                reg_lambda=p["lam"], reg_alpha=p["alpha"], gamma=p["gamma"],
                                min_child_weight=p["mcw"], max_delta_step=p["mds"])


def _random_problem(rng):
    n = int(rng.integers(6, 41))
    nf = int(rng.integers(1, 5))
    X = rng.integers(0, int(rng.integers(2, 9)), (n, nf)) / 4.0  # ties
    g = rng.standard_normal(n) * rng.choice([0.1, 1.0, 10.0])
    if rng.random() < 0.3:
        g += rng.choice([-2.0, 2.0])
    h = rng.uniform(0.05, 2.0, n)
    p = dict(lam=float(rng.choice([0.0, 1.0, 5.0])), alpha=float(rng.choice([0.0, 0.0, 0.5, 3.0])),
             gamma=float(rng.choice([0.0, 0.0, 0.1, 1.0])), mcw=float(rng.choice([0.0, 1.0, 3.0])),
             mds=float(rng.choice([0.0, 0.0, 0.3])))
    return X, g, h, p


@pytest.mark.parametrize("seed", range(60))
def test_newton_tree_matches_bruteforce_reference(seed):
    rng = np.random.default_rng(seed)
    X, g, h, p = _random_problem(rng)
    depth = int(rng.integers(2, 4))
    msl = int(rng.choice([1, 1, 2]))
    mss = int(rng.choice([2, 2, 4]))
    a = _newton_tree(p, depth, msl, mss, presort=True)
    a.fit(X, g, hessian=h)
    check_newton_tree(a, X, g, h, p, depth, msl, mss)
    b = _newton_tree(p, depth, msl, mss, presort=False)
    b.fit(X, g, hessian=h)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert a.depth_features_ == b.depth_features_ and a.depth_improvements_ == b.depth_improvements_


def test_newton_tree_requires_hessian_and_regression():
    X = np.arange(8.0).reshape(-1, 1)
    t = _newton_tree(dict(lam=1.0, alpha=0.0, gamma=0.0, mcw=1.0, mds=0.0), 2)
    with pytest.raises(ValueError, match="hessian"):
        t.fit(X, np.ones(8))
    c = _rdt_native.RDTModel("classification", "newton", 2, 2, 1)
    with pytest.raises(ValueError, match="regression"):
        c.fit(X, np.ones(8), hessian=np.ones(8))


# --------------------------------------------------------------------------- presort on/off
def _gb_problem(kind, seed, n=240, p=6):
    rng = np.random.default_rng(seed)
    X = np.round(rng.standard_normal((n, p)) * 4) / 4
    s = X[:, 0] + 0.6 * X[:, 1] - 0.4 * X[:, 2] * X[:, 3] + 0.5 * rng.standard_normal(n)
    if kind == "regression":
        y = s
    elif kind == "multi":
        y = np.digitize(s, np.quantile(s, [0.3, 0.7])).astype(float)
    else:
        y = (s > 0.2).astype(float)
    w = rng.uniform(0.0, 2.0, n)
    w[rng.random(n) < 0.1] = 0.0
    return X, y, w


@pytest.mark.parametrize("kind,loss", [("binary", "log_loss"), ("multi", "log_loss"),
                                       ("binary", "exponential"), ("regression", "squared_error")])
@pytest.mark.parametrize("variant", range(4))
def test_presort_on_off_bitwise(kind, loss, variant):
    X, y, w = _gb_problem(kind, variant)
    rng = np.random.default_rng(100 + variant)
    params = dict(task="regression" if kind == "regression" else "classification", loss=loss,
                  split_criterion="newton", n_estimators=8, learning_rate=0.4, max_depth=int(rng.integers(2, 5)),
                  reg_lambda=float(rng.choice([0.0, 1.0, 10.0])), reg_alpha=float(rng.choice([0.0, 0.5])),
                  gamma=float(rng.choice([0.0, 0.2])), min_child_weight=float(rng.choice([0.0, 1.0, 5.0])),
                  max_delta_step=float(rng.choice([0.0, 0.5])), subsample=float(rng.choice([1.0, 0.7])),
                  colsample_bytree=float(rng.choice([1.0, 0.6])), min_samples_leaf=int(rng.choice([1, 3])),
                  random_state=variant, n_jobs=2)
    if kind == "binary" and variant % 2:
        params["scale_pos_weight"] = 2.5
    sw = w if variant % 2 == 0 else None
    a = GradientBoostingRDT(presort=True, **params).fit(X, y, sample_weight=sw)
    b = GradientBoostingRDT(presort=False, **params).fit(X, y, sample_weight=sw)
    assert np.array_equal(a.predict(X), b.predict(X))
    if kind != "regression":
        assert np.array_equal(a.decision_function(X), b.decision_function(X))
    assert np.array_equal(a.feature_importances_, b.feature_importances_)
    assert np.array_equal(a.train_score_, b.train_score_)
    assert a.depth_features_ == b.depth_features_


# --------------------------------------------------------------------------- semantics / API
def test_losses_without_hessian_raise():
    for loss in ("absolute_error", "huber", "quantile"):
        with pytest.raises(ValueError, match="hessian"):
            GradientBoostingRDT(task="regression", loss=loss, split_criterion="newton")
    with pytest.raises(ValueError, match="split_criterion"):
        GradientBoostingRDT(split_criterion="gini")
    m = _rdt_native.GradientBoostingModel(
        "regression", "huber", 3, 2, 1, 0.0, 5, 0.1, 1.0, _rdt_native._max_features_all(), 0.1, 0, 1e-4,
        0.0, 1.0, 1.0, 0.5, 0, 1, split_criterion="newton")
    with pytest.raises(ValueError, match="hessian"):
        m.fit(np.zeros((4, 1)), np.arange(4.0), None)


def test_parameter_defaults_and_validation():
    nt = GradientBoostingRDT(split_criterion="newton")
    assert (nt.reg_lambda, nt.reg_alpha, nt.min_impurity_decrease, nt.min_child_weight, nt.max_delta_step) == \
        (1.0, 0.0, 0.0, 1.0, 0.0)
    assert nt.min_samples_leaf == 1
    nt = GradientBoostingRDT(split_criterion="newton", min_child_weight=7.5, min_samples_leaf=3, gamma=2.0)
    assert (nt.min_child_weight, nt.min_samples_leaf, nt.min_impurity_decrease) == (7.5, 3, 2.0)
    fm = GradientBoostingRDT(min_child_weight=4)
    assert fm.min_samples_leaf == 4 and fm.reg_lambda == 0.0  # friedman: alias, unchanged default
    with pytest.raises(ValueError, match="max_delta_step"):
        GradientBoostingRDT(max_delta_step=0.5)  # friedman_mse
    for bad in (dict(max_delta_step=-1.0), dict(reg_lambda=-1.0), dict(reg_alpha=-0.1), dict(gamma=-1.0),
                dict(min_child_weight=-1.0), dict(max_delta_step=np.inf)):
        with pytest.raises(ValueError):
            GradientBoostingRDT(split_criterion="newton", **bad)


def _first_round_tree(**kw):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 16, (200, 3)) / 4.0
    y = (X[:, 0] + rng.normal(0, 1, 200) > 2).astype(float)
    m = GradientBoostingRDT(split_criterion="newton", n_estimators=1, learning_rate=1.0, max_depth=3,
                            random_state=0, **kw).fit(X, y)
    return m, X, m.decision_function(X) - m._model.init_scores_[0]


def test_min_child_weight_is_a_hessian_sum():
    # h = p(1-p) ~ 0.24 per row here, so 200 rows carry ~48 of hessian.
    m, _, leaf = _first_round_tree(min_child_weight=30.0)
    assert m.depth_features_ == [{}] and np.all(leaf == leaf[0])  # no child can reach 30
    m, _, _ = _first_round_tree(min_child_weight=5.0)
    assert 0 in m.depth_features_[0]


def test_gamma_and_max_delta_step():
    _, _, free = _first_round_tree()
    m, _, none = _first_round_tree(gamma=1e6)
    assert m.depth_features_ == [{}] and np.all(none == none[0])
    _, _, clipped = _first_round_tree(max_delta_step=0.05)
    assert np.max(np.abs(free)) > 0.05 and np.max(np.abs(clipped)) <= 0.05 + 1e-15
    _, _, shrunk = _first_round_tree(reg_lambda=1e4)
    assert np.max(np.abs(shrunk)) < 0.2 * np.max(np.abs(free))


def test_feature_importances_are_normalised_total_gain():
    # One squared_error round: g = F0 - y, h = 1, so the round's tree is the
    # native newton tree fitted on those gradients (checked against the
    # reference above); importances are its depth gains, normalised.
    X, y, _ = _gb_problem("regression", 1)
    p = dict(lam=2.0, alpha=0.5, gamma=0.1, mcw=1.0, mds=0.0)
    m = GradientBoostingRDT(task="regression", split_criterion="newton", n_estimators=1, max_depth=3,
                            reg_lambda=2.0, reg_alpha=0.5, gamma=0.1, random_state=0).fit(X, y)
    t = _newton_tree(p, 3)
    t.fit(X, m._model.init_scores_[0] - y, hessian=np.ones(len(y)))
    expected = np.zeros(X.shape[1])
    for d, f in t.depth_features_.items():
        expected[f] += t.depth_improvements_[d]
    assert m.depth_features_[0] == t.depth_features_
    np.testing.assert_allclose(m.feature_importances_, expected / expected.sum(), rtol=1e-13)
