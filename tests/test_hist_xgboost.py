"""tree_method='hist' parity with XGBoost's hist at depth 1.

GradientBoostingRDT(split_criterion='newton', tree_method='hist', max_depth=1)
against xgboost tree_method='hist', max_depth=1 (a depth-1 restricted tree is
a stump: the best (feature, bin boundary) by gain, as XGBoost's root split),
base_margin = our initial score. Features take at most max_bin distinct values
(multiples of 1/16, as are labels and weights) and the cuts are checked to
match: XGBoost's bins (get_quantile_cut) must partition every feature's
training values exactly as ours do. Every round's training margins must then
agree within a few float32 ulps. Scenarios where a decision is within float32
rounding (gain ties between features or bin boundaries, gain vs gamma / the
1e-6 floor, a child's hessian vs min_child_weight) are skipped and counted.
"""

import itertools

import numpy as np
import pytest

# Parity with the newton criterion holds for XGBoost >= 3.4 (see tests/test_newton_xgboost.py).
xgb = pytest.importorskip("xgboost", minversion="3.4.0")

from rdt import _rdt_native  # noqa: E402
from rdt.ensemble import GradientBoostingRDT  # noqa: E402

# XGBoost adds each round's float32 leaf value to a float32 margin, so the
# difference grows by up to ~1 ulp per round; observed max 3.79 over 4 rounds.
ULPS = 6


def _T(G, a):
    return np.sign(G) * max(abs(G) - a, 0.0)


def _score(G, H, p):
    if H <= 0:
        return 0.0
    if p["mds"] == 0:
        t = _T(G, p["alpha"])
        return t * t / (H + p["lam"])
    w = -_T(G, p["alpha"]) / (H + p["lam"])
    if abs(w) > p["mds"]:
        w = np.copysign(p["mds"], w)
    return -(2 * G * w + (H + p["lam"]) * w * w + 2 * p["alpha"] * abs(w))


def _stump_is_near_tie(X, g, h, p):
    """float64 root split over all features; True if a decision is within float32 rounding."""
    G, H = g.sum(), h.sum()
    parent = _score(G, H, p)
    scale = max(1.0, abs(parent), abs(G))
    eps = 2e-6 * scale
    gains = []
    for f in range(X.shape[1]):
        x = X[:, f]
        for a in np.unique(x)[:-1]:
            left = x <= a
            HL = h[left].sum()
            HR = H - HL
            for hh in (HL, HR):
                if 0 < abs(hh - p["mcw"]) < 1e-5 * max(1.0, p["mcw"]):
                    return True
            if HL > 0 and HR > 0 and HL >= p["mcw"] and HR >= p["mcw"]:
                GL = g[left].sum()
                gains.append(_score(GL, HL, p) + _score(G - GL, HR, p) - parent)
    if not gains:
        return False
    gains.sort(reverse=True)
    best = gains[0]
    if best == 0.0 and parent == 0.0:
        return False
    if abs(best - 1e-6) < eps or (p["gamma"] > 0 and abs(best - p["gamma"]) < eps):
        return True
    splits = best >= p["gamma"] and best > 1e-6
    return splits and len(gains) > 1 and best - gains[1] < eps


def _gradients(kind, F, y, w):
    if kind == "binary":
        pr = 1 / (1 + np.exp(-F))
        return [((pr - y) * w, np.maximum(pr * (1 - pr), 1e-16) * w)]
    if kind == "multi":
        e = np.exp(F - F.max(axis=1, keepdims=True))
        P = e / e.sum(axis=1, keepdims=True)
        Y = np.eye(F.shape[1])[y.astype(int)]
        return [((P[:, k] - Y[:, k]) * w, np.maximum(2 * P[:, k] * (1 - P[:, k]) * w, 1e-16))
                for k in range(F.shape[1])]
    return [((F - y) * w, w.copy())]


def _cuts_match(X, d, max_bin):
    indptr, values = d.get_quantile_cut()
    _, ours, _ = _rdt_native._hist_bins(X, np.arange(len(X), dtype=np.uint32), max_bin)
    for f in range(X.shape[1]):
        v = np.unique(X[:, f])
        theirs = np.searchsorted(values[indptr[f]:indptr[f + 1]], v.astype(np.float32), side="right")
        mine = np.searchsorted(ours[f], v, side="left")
        if not np.array_equal(theirs[1:] != theirs[:-1], mine[1:] != mine[:-1]):
            return False
    return True


def run_scenario(kind, seed, p_features, lam, alpha, gamma, mcw, mds, weighted, max_bin=64, rounds=4, lr=0.5):
    """('ok', ulps) | ('near', None) | ('cuts', None)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 201))
    X = rng.integers(0, 48, (n, p_features)) / 16.0
    s = X[:, 0] - 0.5 * X[:, -1] + rng.normal(0, 0.5, n)
    if kind == "binary":
        y = (s > 0.7).astype(float)
        y[:2] = [0, 1]
    elif kind == "multi":
        y = np.digitize(s, [0.3, 1.2]).astype(float)
        y[:3] = [0, 1, 2]
    else:
        y = np.round(s * 16) / 16
    w = rng.integers(1, 33, n) / 16.0 if weighted else None
    task = "regression" if kind == "reg" else "classification"
    loss = "squared_error" if kind == "reg" else "log_loss"
    m = GradientBoostingRDT(task=task, loss=loss, split_criterion="newton", tree_method="hist", max_bin=max_bin,
                            n_estimators=rounds, learning_rate=lr, max_depth=1, reg_lambda=lam, reg_alpha=alpha,
                            gamma=gamma, min_child_weight=mcw, max_delta_step=mds, random_state=0)
    m.fit(X, y, sample_weight=w)
    stages = list(m.staged_predict(X)) if kind == "reg" else list(m.staged_decision_function(X))
    init = np.asarray(m._model.init_scores_)
    p = dict(lam=lam, alpha=alpha, gamma=gamma, mcw=mcw, mds=mds)
    ew = np.ones(n) if w is None else w
    prev = np.tile(init, (n, 1)) if kind == "multi" else np.full(n, init[0])
    for r in range(rounds):
        if any(_stump_is_near_tie(X, g, h, p) for g, h in _gradients(kind, prev, y, ew)):
            return "near", None
        prev = stages[r]

    params = dict(tree_method="hist", max_bin=max_bin, max_depth=1, eta=lr, reg_lambda=lam, reg_alpha=alpha,
                  gamma=gamma, min_child_weight=mcw, max_delta_step=mds, nthread=1)
    if kind == "binary":
        params["objective"] = "binary:logistic"
        bm = np.full(n, init[0])
    elif kind == "multi":
        params.update(objective="multi:softprob", num_class=3)
        bm = np.tile(init, (n, 1))
    else:
        params["objective"] = "reg:squarederror"
        bm = np.full(n, init[0])
    d = xgb.DMatrix(X, label=y, weight=w, base_margin=bm)
    booster = xgb.train(params, d, rounds)
    if not _cuts_match(X, d, max_bin):
        return "cuts", None
    worst = 0.0
    for r in range(1, rounds + 1):
        theirs = booster.predict(d, output_margin=True, iteration_range=(0, r))
        ours = stages[r - 1]
        ulp = np.spacing(np.maximum(np.abs(ours), 1.0).astype(np.float32)).astype(np.float64)
        worst = max(worst, float(np.max(np.abs(theirs - ours) / ulp)))
    return "ok", worst


QUICK = [
    ("binary", 0, 1, 1.0, 0.0, 0.0, 1.0, 0.0, False),
    ("binary", 1, 3, 0.0, 5.0, 0.5, 0.0, 0.7, True),
    ("multi", 2, 2, 1.0, 0.0, 0.0, 1.0, 0.0, True),
    ("reg", 3, 3, 100.0, 0.0, 0.5, 50.0, 0.0, False),
]


@pytest.mark.parametrize("case", QUICK, ids=lambda c: f"{c[0]}-p{c[2]}")
def test_xgboost_hist_depth1_parity_quick(case):
    status, ulps = run_scenario(*case)
    if status != "ok":
        pytest.skip(f"not comparable: {status}")
    assert ulps <= ULPS


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["binary", "multi", "reg"])
def test_xgboost_hist_depth1_parity_grid(kind):
    grid = itertools.product([0.0, 1.0, 100.0], [0.0, 5.0], [0.0, 0.5], [0.0, 1.0, 50.0], [0.0, 0.7],
                             [False, True], [1, 3])
    results = [run_scenario(kind, 100 * i + nf, nf, lam, alpha, gamma, mcw, mds, weighted)
               for i, (lam, alpha, gamma, mcw, mds, weighted, nf) in enumerate(grid)]
    ok = [u for s, u in results if s == "ok"]
    counts = {s: sum(1 for st, _ in results if st == s) for s in ("ok", "near", "cuts")}
    print(f"\n[xgboost hist depth-1 parity {kind}] {counts}, max {max(ok):.2f} float32 ulps")
    assert max(ok) <= ULPS
    assert counts["cuts"] == 0
    assert counts["ok"] >= 0.8 * len(results)


def test_cuts_match_xgboost_up_to_max_bin_distinct_values():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 48, 500) / 16.0, rng.integers(0, 5, 500) / 2.0])
    d = xgb.DMatrix(X, label=rng.random(500))
    xgb.train(dict(tree_method="hist", max_bin=64, max_depth=1), d, 1)
    assert _cuts_match(X, d, 64)


def _non_dyadic_scenario(kind, seed):
    """('ok', worst bound ratio) | ('near', None) | ('cuts', None): float32 feature values with arbitrary
    mantissas, float32 weights, random regularisation. Each round's root feature and row partition must
    equal XGBoost's exactly, and the margins must agree within a float32 error bound (XGBoost evaluates
    the link and each row's gradient / hessian in float32 and stores leaves and margins in float32)."""
    rng = np.random.default_rng(seed)
    n, p_feat, max_bin = int(rng.integers(60, 300)), int(rng.integers(1, 5)), int(rng.choice([16, 64, 256]))
    cols = []
    for _ in range(p_feat):
        vals = np.unique(rng.standard_normal(int(rng.integers(2, max_bin + 1))).astype(np.float32)
                         * np.float32(rng.choice([0.01, 1.0, 300.0])))
        cols.append(vals[rng.integers(0, len(vals), n)].astype(np.float64))
    X = np.column_stack(cols)
    s = X[:, 0] / (X[:, 0].std() + 1e-30) + rng.normal(0, 0.7, n)
    if kind == "binary":
        y = (s > rng.normal(0, 0.5)).astype(float)
        y[:2] = [0, 1]
    elif kind == "multi":
        y = np.digitize(s, np.quantile(s, [0.35, 0.7])).astype(float)
        y[:3] = [0, 1, 2]
    else:
        y = (s * 1.7 + 0.3).astype(np.float32).astype(float)
    w = rng.uniform(0.2, 3.0, n).astype(np.float32).astype(float) if rng.random() < 0.5 else None
    p = dict(lam=float(rng.choice([0.0, 1.0, 7.5])), alpha=float(rng.choice([0.0, 0.8])),
             gamma=float(rng.choice([0.0, 0.3])), mcw=float(rng.choice([0.0, 1.0, 4.0])),
             mds=float(rng.choice([0.0, 0.5])))
    rounds, lr = int(rng.integers(2, 6)), float(rng.choice([0.3, 1.0]))
    m = GradientBoostingRDT(task="regression" if kind == "reg" else "classification",
                            loss="squared_error" if kind == "reg" else "log_loss", split_criterion="newton",
                            tree_method="hist", max_bin=max_bin, n_estimators=rounds, learning_rate=lr, max_depth=1,
                            reg_lambda=p["lam"], reg_alpha=p["alpha"], gamma=p["gamma"], min_child_weight=p["mcw"],
                            max_delta_step=p["mds"], random_state=0).fit(X, y, sample_weight=w)
    stages = list(m.staged_predict(X)) if kind == "reg" else list(m.staged_decision_function(X))
    init = np.asarray(m._model.init_scores_)
    k_out = 3 if kind == "multi" else 1
    params = dict(tree_method="hist", max_bin=max_bin, max_depth=1, eta=lr, reg_lambda=p["lam"], reg_alpha=p["alpha"],
                  gamma=p["gamma"], min_child_weight=p["mcw"], max_delta_step=p["mds"], nthread=1)
    params.update({"binary": dict(objective="binary:logistic"), "reg": dict(objective="reg:squarederror"),
                   "multi": dict(objective="multi:softprob", num_class=3)}[kind])
    d = xgb.DMatrix(X, label=y, weight=w, base_margin=np.tile(init, (n, 1)) if kind == "multi" else np.full(n, init[0]))
    booster = xgb.train(params, d, rounds)
    if not _cuts_match(X, d, max_bin):
        return "cuts", None
    leaves = booster.predict(d, pred_leaf=True).reshape(n, -1)
    roots = [int(t.split("[f", 1)[1].split("<", 1)[0]) if t.startswith("0:[") else -1
             for t in booster.get_dump()]
    ew = np.ones(n) if w is None else w
    eps = float(np.finfo(np.float32).eps)
    prev = np.tile(init, (n, 1)) if kind == "multi" else np.full(n, init[0])
    bound = np.zeros((n, k_out))
    worst = 0.0
    for r in range(rounds):
        gh = _gradients(kind, prev, y, ew)
        if any(_stump_is_near_tie(X, g, h, p) for g, h in gh):
            return "near", None
        ours = stages[r].reshape(n, k_out)
        theirs = booster.predict(d, output_margin=True, iteration_range=(0, r + 1)).reshape(n, k_out)
        assert [m.depth_features_[r * k_out + k].get(0, -1) for k in range(k_out)] == roots[r * k_out:(r + 1) * k_out]
        for k in range(k_out):
            inc = (ours[:, k] - prev.reshape(n, k_out)[:, k]) / lr
            lid = leaves[:, r * k_out + k]
            # the same partition: one increment per XGBoost leaf, different between leaves
            tol = 1e-9 * max(1.0, float(np.abs(inc).max()))
            means = [inc[lid == leaf].mean() for leaf in np.unique(lid)]
            assert all(np.ptp(inc[lid == leaf]) <= tol for leaf in np.unique(lid))
            assert len(means) == 1 or abs(means[0] - means[1]) > tol
            g, h = gh[k]
            for leaf in np.unique(lid):
                rows = lid == leaf
                wr, wl = ew[rows], abs(inc[rows][0])
                if kind == "reg":
                    dg, dh = 2 * eps * (np.abs(g[rows]) + wr * np.abs(y[rows])).sum(), eps * wr.sum()
                else:
                    dg = dh = 2 * eps * wr.sum()
                bnd = 4 * (dg + wl * dh) / max(h[rows].sum() + p["lam"], 1e-300) + 2 * eps * max(wl, 1.0)
                bound[rows, k] += lr * bnd + eps * np.maximum(np.abs(ours[rows, k]), 1.0)
        worst = max(worst, float(np.max(np.abs(theirs - ours) / bound)))
        prev = stages[r]
    return "ok", worst


@pytest.mark.parametrize("kind", ["binary", "multi", "reg"])
def test_xgboost_hist_depth1_partitions_non_dyadic(kind):
    results = [_non_dyadic_scenario(kind, 1000 + i) for i in range(25)]
    ok = [u for s, u in results if s == "ok"]
    assert sum(s == "cuts" for s, _ in results) == 0
    assert len(ok) >= 20 and max(ok) <= 1.0, (len(ok), max(ok))
