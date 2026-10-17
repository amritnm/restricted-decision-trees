"""Parity of GradientBoostingRDT(split_criterion='newton') with XGBoost.

XGBoost ``tree_method='exact'`` on a single feature (with one feature a
restricted tree is an ordinary depth-wise tree), depth 1-3, ``base_margin`` =
our initial score, ``eta`` = learning_rate, no sampling. Every round's margins
must agree within a few float32 ulps (XGBoost computes gradients, hessians,
leaf weights and margins in float32). Features are multiples of 1/16 (as are
labels and weights) so float32 and float64 hold the same values and midpoints.

Scenarios are skipped (and counted) when the comparison is not meaningful:

- tie: two candidate thresholds of a node have the same gain (XGBoost scans
  from the right and keeps the highest threshold, ours keeps the lowest), or a
  gain is within float32 rounding of the runner-up, of gamma or of the 1e-6
  floor, or a child's hessian sum is within float32 rounding of
  min_child_weight. Detected with a float64 reference on the gradients our
  model actually used.
- prune: gamma > 0 and XGBoost kept a split whose gain is below gamma. XGBoost
  'exact' prunes by gamma after growing (bottom-up, so such a split survives
  when a split below it reaches gamma); this implementation applies gamma to
  each node as the tree grows, as XGBoost 'hist' does.

Needs XGBoost >= 3.4.0 (skipped otherwise). The newton mode follows the
current XGBoost formulas; XGBoost 3.0-3.3 differ in two corners, so the
comparison would fail there for reasons that are not bugs here:

- max_delta_step > 0 with reg_alpha > 0: 3.0-3.3 score a child at its
  (clipped) weight w as -(2 G w + (H + lambda) w^2), without the
  2 * alpha * |w| term (so even an unclipped weight then scores
  (G^2 - alpha^2) / (H + lambda) rather than T(G)^2 / (H + lambda)); 3.4 adds
  the term, as here.
- a node that becomes a leaf with a hessian sum below min_child_weight (only
  the root can: every other node is a child of a valid split): 3.0-3.3 give
  it weight 0, 3.4 (and this implementation) -T(G) / (H + lambda).
"""

import itertools
import json

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost", minversion="3.4.0")

from rdt.ensemble import GradientBoostingRDT  # noqa: E402

ULPS = 4  # observed max ~1.5 float32 ulps of max(1, |margin|)


# --------------------------------------------------------------------------- float64 reference (tie detection)
def _T(G, a):
    return np.sign(G) * max(abs(G) - a, 0.0)


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


def _tie_flags(x, g, h, p, depth):
    """Grow the single-feature tree in float64. Returns 'tie' if a node's best
    threshold has an exact tie (|gap| <= 1e-9 of the node's scale), 'near' if a
    decision is within float32 rounding (2e-6 of the scale), else None."""
    level = [np.arange(len(x))]
    for _ in range(depth):
        nxt = []
        for idx in level:
            if len(idx) < 2:
                continue
            xs, gs, hs = x[idx], g[idx], h[idx]
            G, H = gs.sum(), hs.sum()
            parent = _score(G, H, p)
            vals = np.unique(xs)
            cands = []
            for a in vals[:-1]:
                left = xs <= a
                GL, HL = gs[left].sum(), hs[left].sum()
                HR = H - HL
                for hh in (HL, HR):
                    if 0 < abs(hh - p["mcw"]) < 1e-5 * max(1.0, p["mcw"]):
                        return "near"
                if not (HL > 0 and HR > 0 and HL >= p["mcw"] and HR >= p["mcw"]):
                    continue
                cands.append((_score(GL, HL, p) + _score(G - GL, HR, p) - parent, a))
            if not cands:
                continue
            best = max(cands, key=lambda c: c[0])  # first max = lowest threshold
            scale = max(1.0, abs(parent), abs(G))
            eps = 2e-6 * scale
            if best[0] == 0.0 and parent == 0.0:
                continue  # every score is exactly 0 (|G| <= alpha everywhere): no split in either
            splits = best[0] >= p["gamma"] and best[0] > 1e-6
            if abs(best[0] - 1e-6) < eps or (p["gamma"] > 0 and abs(best[0] - p["gamma"]) < eps):
                return "near"
            if splits:
                runner_up = max((c[0] for c in cands if c is not best), default=-np.inf)
                if best[0] - runner_up <= 1e-9 * scale:
                    return "tie"
                if best[0] - runner_up < eps:
                    return "near"
                left = xs <= best[1]
                nxt += [idx[left], idx[~left]]
        level = nxt
    return None


def _gradients(kind, F, y, w):
    """Per-output (g, h) for margins F, as the newton mode computes them."""
    if kind in ("binary", "spw"):
        p = 1 / (1 + np.exp(-F))
        return [((p - y) * w, np.maximum(p * (1 - p), 1e-16) * w)]
    if kind == "exponential":
        t = 2 * y - 1
        e = np.exp(-t * F)
        return [(-t * e * w, e * w)]
    if kind == "multi":
        e = np.exp(F - F.max(axis=1, keepdims=True))
        P = e / e.sum(axis=1, keepdims=True)
        Y = np.eye(F.shape[1])[y.astype(int)]
        return [((P[:, k] - Y[:, k]) * w, np.maximum(2 * P[:, k] * (1 - P[:, k]) * w, 1e-16))
                for k in range(F.shape[1])]
    return [((F - y) * w, w.copy())]


# --------------------------------------------------------------------------- scenarios
def _data(kind, seed, weighted):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(40, 161))
    x = rng.integers(0, 48, n) / 16.0
    if kind in ("binary", "spw", "exponential"):
        y = (rng.random(n) < 1 / (1 + np.exp(-2 * (x - 1.5)))).astype(float)
        y[:2] = [0, 1]
    elif kind == "multi":
        y = np.clip(np.floor(x + rng.normal(0, 0.6, n)), 0, 2)
        y[:3] = [0, 1, 2]
    else:
        y = np.round((np.sin(2 * x) + rng.normal(0, 0.3, n)) * 16) / 16
    w = rng.integers(1, 33, n) / 16.0 if weighted else None
    return x, y, w


def _exp_obj(w):
    def obj(predt, dtrain):
        t = 2 * dtrain.get_label() - 1
        e = np.exp(-t * predt.astype(np.float64))
        return -t * e * w, e * w
    return obj


def run_scenario(kind, seed, depth, lam, alpha, gamma, mcw, mds, weighted, rounds=3, lr=0.3, audit=False):
    """Returns ('ok', max_ulps) or ('tie'|'near'|'prune', None). audit=True
    compares flagged scenarios too and returns (flag or 'ok', max_ulps)."""
    x, y, w = _data(kind, seed, weighted)
    X = x.reshape(-1, 1)
    n = len(x)
    spw = 2.5 if kind == "spw" else None
    task = "regression" if kind == "reg" else "classification"
    loss = {"reg": "squared_error", "exponential": "exponential"}.get(kind, "log_loss")
    m = GradientBoostingRDT(task=task, loss=loss, split_criterion="newton", n_estimators=rounds, learning_rate=lr,
                            max_depth=depth, reg_lambda=lam, reg_alpha=alpha, gamma=gamma, min_child_weight=mcw,
                            max_delta_step=mds, scale_pos_weight=spw, random_state=0)
    m.fit(X, y, sample_weight=w)
    stages = list(m.staged_predict(X)) if kind == "reg" else list(m.staged_decision_function(X))
    init = np.asarray(m._model.init_scores_)

    # Tie detection on the gradients each of our rounds used.
    ew = np.ones(n) if w is None else w.copy()
    if spw is not None:
        ew = ew * np.where(y == 1, spw, 1.0)
    p = dict(lam=lam, alpha=alpha, gamma=gamma, mcw=mcw, mds=mds)
    prev = np.tile(init, (n, 1)) if kind == "multi" else np.full(n, init[0])
    flag = None
    for r in range(rounds):
        for g, h in _gradients(kind, prev, y, ew):
            flag = flag or _tie_flags(x, g, h, p, depth)
        if flag and not audit:
            return flag, None
        prev = stages[r]

    params = dict(tree_method="exact", max_depth=depth, eta=lr, reg_lambda=lam, reg_alpha=alpha, gamma=gamma,
                  min_child_weight=mcw, max_delta_step=mds, nthread=1)
    obj = None
    if kind in ("binary", "spw"):
        params["objective"] = "binary:logistic"
        if spw is not None:
            params["scale_pos_weight"] = spw
        bm = np.full(n, init[0])
    elif kind == "multi":
        params.update(objective="multi:softprob", num_class=3)
        bm = np.tile(init, (n, 1))
    elif kind == "exponential":
        obj = _exp_obj(np.ones(n) if w is None else w)
        bm = np.full(n, init[0])
    else:
        params["objective"] = "reg:squarederror"
        bm = np.full(n, init[0])
    d = xgb.DMatrix(X, label=y, weight=w, base_margin=bm)
    booster = xgb.train(params, d, rounds, obj=obj)

    if gamma > 0:
        for tree in booster.get_dump(dump_format="json", with_stats=True):
            stack = [json.loads(tree)]
            while stack:
                node = stack.pop()
                if "children" in node:
                    if node["gain"] < gamma:
                        flag = flag or "prune"
                        if not audit:
                            return flag, None
                    stack += node["children"]

    worst = 0.0
    for r in range(1, rounds + 1):
        theirs = booster.predict(d, output_margin=True, iteration_range=(0, r))
        ours = stages[r - 1]
        ulp = np.spacing(np.maximum(np.abs(ours), 1.0).astype(np.float32)).astype(np.float64)
        worst = max(worst, float(np.max(np.abs(theirs - ours) / ulp)))
    return flag or "ok", worst


KINDS = ["binary", "multi", "reg", "spw", "exponential"]


def _check(results):
    ok = [u for status, u in results if status == "ok"]
    assert ok, "every scenario was skipped"
    assert max(ok) <= ULPS, f"margin differs by {max(ok):.2f} float32 ulps"
    return ok


QUICK = [  # (kind, seed, depth, lam, alpha, gamma, mcw, mds, weighted)
    ("binary", 0, 1, 1.0, 0.0, 0.0, 1.0, 0.0, False),
    ("binary", 1, 3, 0.0, 5.0, 0.5, 0.0, 0.7, True),
    ("multi", 2, 2, 1.0, 0.0, 0.0, 1.0, 0.0, True),
    ("multi", 3, 3, 100.0, 0.0, 0.0, 0.0, 0.7, False),
    ("reg", 4, 2, 1.0, 5.0, 0.5, 1.0, 0.0, True),
    ("reg", 5, 3, 0.0, 0.0, 0.0, 50.0, 0.7, False),
    ("spw", 6, 2, 1.0, 0.0, 0.0, 1.0, 0.0, True),
    ("exponential", 7, 3, 1.0, 0.0, 0.0, 1.0, 0.0, False),
]


@pytest.mark.parametrize("case", QUICK, ids=lambda c: f"{c[0]}-d{c[2]}")
def test_xgboost_exact_parity_quick(case):
    status, ulps = run_scenario(*case)
    if status != "ok":
        pytest.skip(f"not comparable: {status}")
    assert ulps <= ULPS


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_xgboost_exact_parity_grid(kind):
    grid = itertools.product([0.0, 1.0, 100.0, 2000.0], [0.0, 5.0, 100.0], [0.0, 0.5], [0.0, 1.0, 50.0],
                             [0.0, 0.7], [False, True])
    results = []
    for i, (lam, alpha, gamma, mcw, mds, weighted) in enumerate(grid):
        for depth in (1, 2, 3):
            results.append(run_scenario(kind, 1000 * depth + i, depth, lam, alpha, gamma, mcw, mds, weighted))
    ok = _check(results)
    counts = {s: sum(1 for st, _ in results if st == s) for s in ("ok", "tie", "near", "prune")}
    print(f"\n[xgboost parity {kind}] {counts}, max {max(ok):.2f} float32 ulps")
    assert counts["ok"] >= 0.6 * len(results)
