"""tree_method='hist' with hist_subtraction (the default) against the direct build.

Histogram subtraction derives a child's histograms as parent - sibling, so its sums are rounded
differently from building them directly; everything else (bins, candidate rule, gain arithmetic,
tie order, leaf values) is shared. So:

1. A tree fitted with subtraction has the direct build's structure, and then its leaves and
   predictions are bit for bit the same, unless a split choice was a near-tie. Wherever two fits
   first differ (a depth whose feature, a threshold or a node's leaf-ness differs), both fits'
   improvements at that depth agree within 1e-12 relative: the depth's score is a max over
   features of sums of per-node maxima, which rounding errors of that size cannot move further.
   Checked on a randomized grid (RDT gini / entropy / mse, RandomForestRDT, GradientBoostingRDT
   friedman_mse and newton; binary, multiclass, regression; weights with zeros; subsample,
   colsample_bytree, colsample_bylevel, early stopping; cache budgets that keep all, some or none of
   the features). Gradient boosting is compared tree by tree up to the first tree that differs.
2. Unweighted classification sums integer counts, which subtract exactly: bitwise equal.
3. Held-out AUC agrees to 4 decimals on larger synthetic tasks.
4. Results do not depend on n_jobs; max_hist_cache_mb=0 is the direct build bit for bit; a small
   budget falls back to direct builds for the features that do not fit, and bounds the memory.
5. colsample_bylevel: the next depth's candidates are drawn one depth early, from the same sequence,
   so level_features_ does not change; features not scored at the previous depth are built
   directly.

The bitwise comparison of hist_subtraction=False against the previous build is done outside the
test suite (as for the other implementation choices).
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from rdt import RDT
from rdt.ensemble import GradientBoostingRDT, RandomForestRDT

TIE_RTOL = 1e-12


# --------------------------------------------------------------------------- tree comparison
def _levels(root):
    out, level = [], [root]
    while level:
        out.append([(n.is_leaf, n.split_feature, n.split_threshold, n.split_bin, n.n_samples) for n in level])
        level = [c for n in level if not n.is_leaf for c in (n.left, n.right)]
    return out


def _leaf_values(root):
    vals, stack = [], [root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            vals.append(n.value)
        else:
            stack += [n.right, n.left]
    return vals


def _compare_trees(root_a, imp_a, root_b, imp_b):
    """'same' (same structure: leaves must then be equal) or 'tie' (verified near-tie)."""
    la, lb = _levels(root_a), _levels(root_b)
    for d in range(max(len(la), len(lb))):
        if d < len(la) and d < len(lb) and la[d] == lb[d]:
            continue
        assert d in imp_a and d in imp_b, f"depth {d} is split in only one fit ({imp_a.get(d)}, {imp_b.get(d)})"
        a, b = imp_a[d], imp_b[d]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        assert rel <= TIE_RTOL, f"fits first differ at depth {d}, improvements {a!r} vs {b!r} (rel {rel:.3g})"
        return "tie"
    assert _leaf_values(root_a) == _leaf_values(root_b)
    return "same"


def _compare_models(kind, direct, sub):
    """Per-tree verdicts; gradient boosting stops at its first differing tree."""
    if kind == "rdt":
        return [_compare_trees(direct.root, direct.depth_improvements_, sub.root, sub.depth_improvements_)]
    if kind == "rf":
        return [_compare_trees(a.root, a.depth_improvements_, b.root, b.depth_improvements_)
                for a, b in zip(direct.estimators_, sub.estimators_)]
    out = []
    for t in range(min(len(direct.depth_features_), len(sub.depth_features_))):
        out.append(_compare_trees(direct._model._tree(t).root, direct.depth_improvements_[t],
                                  sub._model._tree(t).root, sub.depth_improvements_[t]))
        if out[-1] != "same":
            break
    return out


# --------------------------------------------------------------------------- randomized grid
def _data(rng, task, n, p, k):
    X = rng.standard_normal((n, p))
    r = rng.random()
    if r < 0.3:
        X = np.round(X * 4) / 4
    elif r < 0.45:
        X[:, : max(1, p // 2)] = rng.integers(0, 3, (n, max(1, p // 2)))
    s = X[:, 0] + 0.5 * X[:, min(1, p - 1)] * X[:, min(2, p - 1)] + 0.3 * rng.standard_normal(n)
    if task == "regression":
        y = s * (100.0 if rng.random() < 0.2 else 1.0) + (1e6 if rng.random() < 0.1 else 0.0)
    else:
        y = np.digitize(s, np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])).astype(float)
    w = None
    if rng.random() < 0.45:
        w = rng.uniform(0.1, 2.0, n)
        if rng.random() < 0.5:
            w[rng.random(n) < 0.15] = 0.0
    return X, y, w


def _scenario(i):
    rng = np.random.default_rng(11000 + i)
    kind = ["rdt", "rf", "gbf", "gbn"][i % 4]
    task = "classification" if rng.random() < 0.65 else "regression"
    k = int(rng.choice([2, 2, 3, 4])) if task == "classification" else 0
    big = rng.random() < 0.3
    n = int(rng.integers(800, 4000)) if big else int(rng.integers(40, 500))
    p = int(rng.integers(1, 12))
    X, y, w = _data(rng, task, n, p, k)
    if rng.random() < 0.3:
        X = X.astype(np.float32)
    common = dict(tree_method="hist", max_bin=int(rng.choice([4, 16, 64, 256, 300])),
                  n_jobs=int(rng.choice([1, 2, 4])),
                  max_hist_cache_mb=float(rng.choice([1024.0, 1024.0, 0.05, 0.002, 0.0])))
    depth = int(rng.integers(5, 10)) if big else int(rng.integers(1, 7))
    bylevel = float(rng.choice([1.0, 1.0, 0.5, 0.8]))
    if kind == "rdt":
        crit = str(rng.choice(["gini", "entropy"])) if task == "classification" else "mse"
        msl = int(rng.integers(1, 4))
        make = lambda sub: RDT(task=task, criterion=crit, max_depth=depth, min_samples_leaf=msl,
                               hist_subtraction=sub, **common)
    elif kind == "rf":
        kw = dict(task=task, n_estimators=int(rng.integers(2, 6)), max_depth=depth,
                  max_features=[None, "sqrt", 0.6][int(rng.integers(0, 3))], random_state=int(rng.integers(0, 100)),
                  colsample_bylevel=bylevel,
                  criterion=str(rng.choice(["gini", "entropy"])) if task == "classification" else "mse")
        make = lambda sub: RandomForestRDT(hist_subtraction=sub, **kw, **common)
    else:
        newton = kind == "gbn"
        kw = dict(task=task, loss="squared_error" if task == "regression" else "log_loss",
                  split_criterion="newton" if newton else "friedman_mse", n_estimators=int(rng.integers(2, 8)),
                  max_depth=depth, learning_rate=float(rng.choice([0.1, 0.5])),
                  subsample=float(rng.choice([1.0, 0.8, 0.5])), colsample_bytree=float(rng.choice([1.0, 0.7, 0.4])),
                  n_iter_no_change=3 if rng.random() < 0.3 else None, validation_fraction=0.2,
                  random_state=int(rng.integers(0, 100)), colsample_bylevel=bylevel)
        if newton:
            kw.update(reg_lambda=float(rng.choice([0.0, 1.0, 5.0])), reg_alpha=float(rng.choice([0.0, 0.0, 0.3])),
                      min_child_weight=float(rng.choice([0.0, 1.0, 3.0])),
                      max_delta_step=float(rng.choice([0.0, 0.0, 0.7])))
        make = lambda sub: GradientBoostingRDT(hist_subtraction=sub, **kw, **common)
    return kind, task, make, X, y, w


def _check_scenario(i):
    kind, task, make, X, y, w = _scenario(i)
    direct, sub = make(False), make(True)
    try:
        direct.fit(X, y, sample_weight=w)
    except ValueError:
        pytest.skip("scenario not fittable")
    sub.fit(X, y, sample_weight=w)
    verdicts = _compare_models(kind, direct, sub)
    if all(v == "same" for v in verdicts) and (kind != "gbf" and kind != "gbn" or
                                                len(direct.depth_features_) == len(sub.depth_features_)):
        Xc = np.ascontiguousarray(X, dtype=np.float64)
        assert np.array_equal(direct.predict(Xc), sub.predict(Xc))
        if task == "classification":
            assert np.array_equal(direct.predict_proba(Xc), sub.predict_proba(Xc))
    return verdicts


@pytest.mark.parametrize("i", range(48))
def test_subtraction_matches_direct_up_to_near_ties(i):
    _check_scenario(i)


@pytest.mark.slow
@pytest.mark.parametrize("i", range(48, 300))
def test_subtraction_matches_direct_up_to_near_ties_grid(i):
    _check_scenario(i)


# --------------------------------------------------------------------------- exact cases
def test_unweighted_classification_subtracts_exactly():
    # Class counts are integers: parent - child is exact, so the fits are identical bit for bit.
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3000, 8))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.standard_normal(3000) > 0).astype(float) + (X[:, 3] > 1)
    for crit in ("gini", "entropy"):
        a = RDT(criterion=crit, max_depth=8, tree_method="hist", hist_subtraction=False).fit(X, y)
        b = RDT(criterion=crit, max_depth=8, tree_method="hist").fit(X, y)
        assert _levels(a.root) == _levels(b.root)
        assert a.depth_improvements_ == b.depth_improvements_
        assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    f = [RandomForestRDT(n_estimators=6, max_depth=7, tree_method="hist", random_state=0, bootstrap=False,
                         max_features=None, hist_subtraction=s).fit(X, y) for s in (False, True)]
    assert np.array_equal(f[0].predict_proba(X), f[1].predict_proba(X))
    assert np.array_equal(f[0].feature_importances_, f[1].feature_importances_)


def test_zero_cache_is_the_direct_build():
    X, y = _task(4000, 10, seed=5)
    a = GradientBoostingRDT(n_estimators=8, max_depth=6, split_criterion="newton", subsample=0.8,
                            tree_method="hist", random_state=0, hist_subtraction=False).fit(X, y)
    b = GradientBoostingRDT(n_estimators=8, max_depth=6, split_criterion="newton", subsample=0.8,
                            tree_method="hist", random_state=0, max_hist_cache_mb=0).fit(X, y)
    assert np.array_equal(a.decision_function(X), b.decision_function(X))
    assert a.depth_improvements_ == b.depth_improvements_


# --------------------------------------------------------------------------- larger tasks
def _task(n, p, seed=0, kind="binary"):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, p // 2:] = np.round(X[:, p // 2:] * 3)  # some low-cardinality columns
    s = X[:, 0] - 0.7 * X[:, 1] + np.sin(2 * X[:, 2]) + 0.5 * X[:, 3] * X[:, 4] + rng.standard_normal(n)
    if kind == "regression":
        return X, s
    return X, (s > np.quantile(s, 0.7)).astype(float)


@pytest.mark.parametrize("make", [
    lambda s: GradientBoostingRDT(n_estimators=40, max_depth=7, learning_rate=0.2, split_criterion="newton",
                                  subsample=0.9, colsample_bytree=0.6, tree_method="hist", random_state=1, n_jobs=2,
                                  hist_subtraction=s),
    lambda s: GradientBoostingRDT(n_estimators=40, max_depth=6, learning_rate=0.2, subsample=0.8,
                                  tree_method="hist", random_state=2, hist_subtraction=s),
    lambda s: RandomForestRDT(n_estimators=20, max_depth=8, tree_method="hist", random_state=3, n_jobs=2,
                              hist_subtraction=s),
], ids=["gb-newton", "gb-friedman", "rf"])
def test_held_out_auc_agrees(make):
    from sklearn.metrics import roc_auc_score

    X, y = _task(30000, 24, seed=7)
    Xt, yt = _task(10000, 24, seed=8)
    aucs = [roc_auc_score(yt, make(s).fit(X, y).predict_proba(Xt)[:, 1]) for s in (False, True)]
    assert round(aucs[0], 4) == round(aucs[1], 4), aucs


@pytest.mark.parametrize("name", ["gb-newton", "gb-multi", "gb-friedman-regression", "rf", "rdt-mse"])
def test_n_jobs_does_not_change_the_result_with_subtraction(name):
    X, y = _task(3000, 12, seed=4, kind="regression" if "regression" in name or "mse" in name else "binary")
    if name == "gb-multi":
        y = np.digitize(X[:, 0] + X[:, 1], [-0.5, 0.5]).astype(float)
    w = np.random.default_rng(1).uniform(0, 2, len(y))

    def make(j):
        if name == "gb-newton":
            return GradientBoostingRDT(n_estimators=6, max_depth=8, split_criterion="newton", subsample=0.8,
                                       colsample_bylevel=0.7, tree_method="hist", random_state=0, n_jobs=j)
        if name == "gb-multi":
            return GradientBoostingRDT(n_estimators=4, max_depth=6, tree_method="hist", random_state=0, n_jobs=j)
        if name == "gb-friedman-regression":
            return GradientBoostingRDT(task="regression", n_estimators=5, max_depth=8, subsample=0.7,
                                       tree_method="hist", random_state=0, n_jobs=j)
        if name == "rf":
            return RandomForestRDT(n_estimators=6, max_depth=8, tree_method="hist", random_state=0, n_jobs=j)
        return RDT(task="regression", max_depth=9, tree_method="hist", n_jobs=j)

    fits = [make(j).fit(X, y, sample_weight=w) for j in (1, 2, 4)]
    pred = "predict" if name.endswith("regression") or name == "rdt-mse" else "predict_proba"
    for m in fits[1:]:
        assert np.array_equal(getattr(fits[0], pred)(X), getattr(m, pred)(X))
        if hasattr(m, "feature_importances_"):
            assert np.array_equal(fits[0].feature_importances_, m.feature_importances_)
        else:
            assert fits[0].depth_improvements_ == m.depth_improvements_


# --------------------------------------------------------------------------- cache budget, bylevel
def test_partial_cache_falls_back_to_direct_builds():
    # A budget that keeps only some features: those subtract, the others are built directly. The fit
    # is still the direct build's up to near-ties, and does not depend on n_jobs.
    X, y = _task(6000, 40, seed=9)
    kw = dict(n_estimators=6, max_depth=8, split_criterion="newton", tree_method="hist", random_state=0)
    direct = GradientBoostingRDT(hist_subtraction=False, **kw).fit(X, y)
    for mb in (0.02, 0.2):
        fits = [GradientBoostingRDT(max_hist_cache_mb=mb, n_jobs=j, **kw).fit(X, y) for j in (1, 3)]
        assert np.array_equal(fits[0].decision_function(X), fits[1].decision_function(X))
        assert set(_compare_models("gbn", direct, fits[0])) <= {"same", "tie"}


def test_colsample_bylevel_draws_do_not_change():
    X, y = _task(3000, 20, seed=6)
    for make in (lambda s: GradientBoostingRDT(n_estimators=5, max_depth=7, colsample_bylevel=0.3,
                                               split_criterion="newton", tree_method="hist", random_state=4,
                                               hist_subtraction=s),
                 lambda s: RandomForestRDT(n_estimators=5, max_depth=7, colsample_bylevel=0.5, tree_method="hist",
                                           random_state=4, hist_subtraction=s)):
        a, b = make(False).fit(X, y), make(True).fit(X, y)
        assert a.level_features_ == b.level_features_
        kind = "rf" if isinstance(a, RandomForestRDT) else "gbn"
        assert set(_compare_models(kind, a, b)) <= {"same", "tie"}


_CACHE_MEMORY_SCRIPT = textwrap.dedent("""
    import resource, sys
    import numpy as np
    from rdt.ensemble import GradientBoostingRDT

    def peak():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024

    rng = np.random.default_rng(0)
    n, p = 20000, 400
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    m = GradientBoostingRDT(n_estimators=1, max_depth=10, split_criterion="newton", tree_method="hist",
                            random_state=0, min_child_weight=0, max_hist_cache_mb=float(sys.argv[1]))
    m.fit(X[:200], y[:200])  # warm up allocators and imports
    before = peak()
    m.fit(X, y)
    print(peak() - before)
""")


def test_cache_budget_bounds_memory():
    pytest.importorskip("resource")

    def growth(mb):
        out = subprocess.run([sys.executable, "-c", _CACHE_MEMORY_SCRIPT, str(mb)], capture_output=True, text=True,
                             check=True).stdout.split()
        return int(out[0])

    # The full cache of this tree is ~200 MiB (depth 8 keeps ~256 nodes x 400 features x 256 bins
    # x 20 B for depth 9). A budget bounds what the cache adds to the peak (one buffer per depth,
    # freed as soon as the next depth is scored).
    direct, bounded, unbounded = growth(0), growth(32), growth(4096)
    assert bounded < direct + (32 + 16) * 2**20, (direct / 2**20, bounded / 2**20)
    assert unbounded > bounded + 100 * 2**20, (bounded / 2**20, unbounded / 2**20)


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("cls", [RDT, RandomForestRDT, GradientBoostingRDT])
def test_validation(cls):
    with pytest.raises(ValueError, match="hist_subtraction"):
        cls(tree_method="hist", hist_subtraction="yes")
    for bad in (-1, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError, match="max_hist_cache_mb"):
            cls(tree_method="hist", max_hist_cache_mb=bad)
    m = cls(tree_method="hist", hist_subtraction=np.bool_(False), max_hist_cache_mb=np.float32(8))
    assert m.hist_subtraction is False and m.max_hist_cache_mb == 8.0
