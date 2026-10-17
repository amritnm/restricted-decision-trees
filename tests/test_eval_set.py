"""GradientBoostingRDT early stopping on a user-supplied eval_set.

1. Truncation: a fit that stops at best round k is bit-for-bit the model
   fit(n_estimators=k) without early stopping -- predictions, probabilities,
   decision function, every staged output, train_score_, importances,
   depth_features_ / level_features_ -- across randomized scenarios (losses,
   friedman_mse / newton, exact with presort on / off and hist, subsample,
   colsample_bytree / colsample_bylevel, sample_weight, scale_pos_weight,
   eval weights, every metric; the full grid is behind -m slow).
2. Metrics: evals_result_ equals scikit-learn / NumPy recomputed from the
   staged outputs on the eval set (logloss and AUC weighted / unweighted with
   ties, error, rmse, mae; binary, multiclass, regression) to ~1e-12.
3. Stopping: the run stops at best + early_stopping_rounds (or n_estimators),
   ties and gains within tol are not improvements, AUC is maximised.
4. Back-compat inside the suite: without eval_set nothing changes (defaults
   resolve to the old tol / validation_fraction; an eval_set without a
   patience only records the metric). The bitwise comparison against the
   base build lives in dev/ (it needs two builds).
5. Validation errors, warnings, determinism, n_jobs invariance, eval set
   input layouts, no copy of a float32 eval set in hist mode, GIL release.
"""

import subprocess
import sys
import textwrap
import threading
import time
import warnings

import numpy as np
import pytest
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from rdt.ensemble import GradientBoostingRDT

from _peak_rss import PEAK_RSS, require_peak_rss  # tests/_peak_rss.py


# --------------------------------------------------------------------------- data / scenarios
def _make(rng, task, n_classes, n, p, quantize):
    X = rng.standard_normal((n, p))
    if quantize:
        X = np.round(X * quantize) / quantize  # tied feature values -> tied scores
    s = X[:, 0] + 0.6 * X[:, min(1, p - 1)] * X[:, min(2, p - 1)] + 1.0 * rng.standard_normal(n)
    if task == "regression":
        return X, s
    cuts = np.quantile(s, np.linspace(0, 1, n_classes + 1)[1:-1])
    return X, np.digitize(s, cuts).astype(float)


def _scenario(i):
    """Random (params, data, eval) for scenario i; the metric is valid for the task."""
    rng = np.random.default_rng(31000 + i)
    kind = rng.choice(["binary", "binary", "multi", "regression"])
    task = "regression" if kind == "regression" else "classification"
    n_classes = 2 if kind == "binary" else (int(rng.integers(3, 5)) if kind == "multi" else 0)
    n, m, p = int(rng.integers(120, 400)), int(rng.integers(60, 300)), int(rng.integers(1, 7))
    quantize = int(rng.choice([0, 2, 4]))
    X, y = _make(rng, task, n_classes, n, p, quantize)
    Xv, yv = _make(rng, task, n_classes, m, p, quantize)
    if kind == "binary":
        loss = str(rng.choice(["log_loss", "log_loss", "exponential"]))
    elif kind == "multi":
        loss = "log_loss"
    else:
        loss = str(rng.choice(["squared_error", "squared_error", "huber", "absolute_error", "quantile"]))
    newton = loss in ("log_loss", "exponential", "squared_error") and rng.random() < 0.5
    tree_method = str(rng.choice(["exact", "exact", "hist"]))
    params = dict(task=task, loss=loss, split_criterion="newton" if newton else "friedman_mse",
                  n_estimators=int(rng.integers(15, 60)), learning_rate=float(rng.choice([0.3, 0.5, 1.0])),
                  max_depth=int(rng.integers(1, 5)), subsample=float(rng.choice([1.0, 0.7])),
                  colsample_bytree=[None, 0.6][int(rng.integers(0, 2))],
                  colsample_bylevel=float(rng.choice([1.0, 1.0, 0.5])), presort=bool(rng.random() < 0.6),
                  tree_method=tree_method, random_state=int(rng.integers(0, 1000)),
                  n_jobs=int(rng.choice([1, 2])))
    if tree_method == "hist":
        params["max_bin"] = int(rng.choice([8, 32, 256]))
        if rng.random() < 0.4:
            X = X.astype(np.float32)
        if rng.random() < 0.4:
            Xv = np.asfortranarray(Xv.astype(np.float32))
    w = None
    if rng.random() < 0.4:
        w = rng.uniform(0.2, 2.0, n)
        if rng.random() < 0.3:
            w[rng.random(n) < 0.1] = 0.0
    if kind == "binary" and rng.random() < 0.25:
        params["scale_pos_weight"] = float(rng.choice([0.5, 3.0]))
    wv = rng.uniform(0.1, 3.0, m) if rng.random() < 0.5 else None
    if wv is not None and rng.random() < 0.3:
        wv[rng.random(m) < 0.15] = 0.0
    metrics = {"binary": [None, "logloss", "auc", "error"], "multi": [None, "logloss", "error"],
               "regression": [None, "rmse", "mae"]}[kind]
    params["eval_metric"] = metrics[int(rng.integers(0, len(metrics)))]
    params["tol"] = [None, None, 0.0, 1e-3][int(rng.integers(0, 4))]
    patience = int(rng.integers(1, 7))
    return params, patience, (X, y, w), (Xv, yv, wv)


def _expected_stop(values, patience, tol, maximize, n_estimators):
    """(best round, rounds run) by the documented rule, from the recorded scores."""
    best, best_r, since = None, 0, 0
    for r, v in enumerate(values):
        if r == 0 or (v > best + tol if maximize else v < best - tol):
            best, best_r, since = v, r, 0
        else:
            since += 1
        if since >= patience:
            return best_r, r + 1
    assert len(values) == n_estimators
    return best_r, len(values)


def _dense(X):
    return np.ascontiguousarray(X, dtype=np.float64)


def _assert_same_model(a, b, X, Xv, task):
    """Bitwise equality of everything a fitted model exposes."""
    assert a.n_estimators_ == b.n_estimators_
    assert np.array_equal(a.train_score_, b.train_score_)
    assert np.array_equal(a.feature_importances_, b.feature_importances_)
    assert a.depth_features_ == b.depth_features_
    assert a.depth_improvements_ == b.depth_improvements_
    assert a.level_features_ == b.level_features_
    for Z in (_dense(X), _dense(Xv)):
        assert np.array_equal(a.predict(Z), b.predict(Z))
        sa, sb = list(a.staged_predict(Z)), list(b.staged_predict(Z))
        assert len(sa) == len(sb) == a.n_estimators_
        assert all(np.array_equal(u, v) for u, v in zip(sa, sb))
        if task == "classification":
            assert np.array_equal(a.predict_proba(Z), b.predict_proba(Z))
            assert np.array_equal(a.decision_function(Z), b.decision_function(Z))
            for fa, fb in ((a.staged_predict_proba, b.staged_predict_proba),
                           (a.staged_decision_function, b.staged_decision_function)):
                sa, sb = list(fa(Z)), list(fb(Z))
                assert len(sa) == len(sb) and all(np.array_equal(u, v) for u, v in zip(sa, sb))


def _fit_es(params, patience, data, ev):
    X, y, w = data
    Xv, yv, wv = ev
    m = GradientBoostingRDT(early_stopping_rounds=patience, **params)
    return m.fit(X, y, sample_weight=w, eval_set=(Xv, yv), eval_sample_weight=wv)


def _reference(params, k, data):
    X, y, w = data
    ref = {key: v for key, v in params.items() if key not in ("eval_metric", "tol")}
    ref["n_estimators"] = k
    return GradientBoostingRDT(**ref).fit(X, y, sample_weight=w)


def _check_truncation(i, hist_subtraction=True):
    params, patience, data, ev = _scenario(i)
    params["hist_subtraction"] = hist_subtraction  # only read by tree_method='hist'
    m = _fit_es(params, patience, data, ev)
    metric = params["eval_metric"] or ("logloss" if params["task"] == "classification" else "rmse")
    values = m.evals_result_[metric]
    tol = 0.0 if params["tol"] is None else params["tol"]
    best, run = _expected_stop(values, patience, tol, metric == "auc", params["n_estimators"])
    assert (m.best_iteration_, len(values)) == (best, run)
    assert m.n_estimators_ == best + 1 and m.best_score_ == values[best]
    ref = _reference(params, best + 1, data)
    _assert_same_model(m, ref, data[0], ev[0], params["task"])
    return run < params["n_estimators"]


# --------------------------------------------------------------------------- 1. truncation
@pytest.mark.parametrize("hist_subtraction", [True, False], ids=["subtraction", "direct"])
@pytest.mark.parametrize("i", range(12))
def test_truncated_model_equals_fit_with_best_n_estimators(i, hist_subtraction):
    _check_truncation(i, hist_subtraction)


@pytest.mark.slow
@pytest.mark.parametrize("hist_subtraction", [True, False], ids=["subtraction", "direct"])
def test_truncation_grid(hist_subtraction):
    stopped = [_check_truncation(i, hist_subtraction) for i in range(12, 212)]
    assert sum(stopped) >= 150, f"only {sum(stopped)} of {len(stopped)} scenarios stopped before n_estimators"


@pytest.mark.parametrize("tree_method", ["exact", "hist"])
def test_truncated_model_reproduced_by_an_unstopped_fit_prefix(tree_method):
    """Rounds 0..best are the first rounds of a full-length fit: its staged
    stage best+1 equals the truncated model."""
    params, patience, data, ev = _scenario(3)
    params.update(tree_method=tree_method, n_estimators=80, learning_rate=1.0)
    m = _fit_es(params, 2, data, ev)
    assert m.n_estimators_ < 80
    full = _reference(params, 80, data)
    Z = _dense(ev[0])
    stages = list(full.staged_predict(Z))
    assert np.array_equal(stages[m.n_estimators_ - 1], m.predict(Z))


# --------------------------------------------------------------------------- 2. metric values
def _metric_problem(kind, seed, quantize=2, n=300, m=250, p=4):
    rng = np.random.default_rng(seed)
    task = "regression" if kind == "regression" else "classification"
    k = {"binary": 2, "multi": 3, "regression": 0}[kind]
    X, y = _make(rng, task, k, n, p, quantize)
    Xv, yv = _make(rng, task, k, m, p, quantize)
    wv = rng.uniform(0.0, 3.0, m)
    wv[rng.random(m) < 0.1] = 0.0
    return X, y, Xv, yv, wv


def _recompute(model, metric, Xv, yv, wv):
    Z = _dense(Xv)
    if metric == "logloss":
        return [log_loss(yv, pr, sample_weight=wv, labels=model.classes_) for pr in model.staged_predict_proba(Z)]
    if metric == "auc":
        return [roc_auc_score(yv, d, sample_weight=wv) for d in model.staged_decision_function(Z)]
    if metric == "error":
        return [np.average(pr != yv, weights=wv) for pr in model.staged_predict(Z)]
    if metric == "rmse":
        return [np.sqrt(mean_squared_error(yv, pr, sample_weight=wv)) for pr in model.staged_predict(Z)]
    return [mean_absolute_error(yv, pr, sample_weight=wv) for pr in model.staged_predict(Z)]


METRIC_CASES = [
    ("binary", "log_loss", "logloss"), ("binary", "exponential", "logloss"), ("multi", "log_loss", "logloss"),
    ("binary", "log_loss", "auc"), ("binary", "exponential", "auc"),
    ("binary", "log_loss", "error"), ("multi", "log_loss", "error"),
    ("regression", "squared_error", "rmse"), ("regression", "huber", "rmse"),
    ("regression", "squared_error", "mae"), ("regression", "absolute_error", "mae"),
]


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("tree_method", ["exact", "hist"])
@pytest.mark.parametrize("kind,loss,metric", METRIC_CASES, ids=lambda v: str(v))
def test_evals_result_equals_sklearn_on_staged_outputs(kind, loss, metric, tree_method, weighted):
    X, y, Xv, yv, wv = _metric_problem(kind, seed=hash((kind, loss, metric)) % 1000)
    wv = wv if weighted else None
    task = "regression" if kind == "regression" else "classification"
    m = GradientBoostingRDT(task=task, loss=loss, n_estimators=25, max_depth=2, learning_rate=0.5,
                            subsample=0.8, random_state=1, tree_method=tree_method, max_bin=16, eval_metric=metric)
    m.fit(X, y, eval_set=(Xv, yv), eval_sample_weight=wv)
    assert m.best_iteration_ is None and m.best_score_ is None  # no patience: recorded only
    assert list(m.evals_result_) == [metric] and len(m.evals_result_[metric]) == 25 == m.n_estimators_
    np.testing.assert_allclose(m.evals_result_[metric], _recompute(m, metric, Xv, yv, wv), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_auc_with_heavy_ties(weighted):
    """Two-valued features and depth-1 trees: every round has few distinct
    scores, each shared by positives and negatives (half credit per tied pair)."""
    rng = np.random.default_rng(7)
    n, m = 400, 300
    X = rng.integers(0, 2, (n, 2)).astype(float)
    y = ((X[:, 0] + rng.random(n)) > 0.9).astype(float)
    Xv = rng.integers(0, 2, (m, 2)).astype(float)
    yv = ((Xv[:, 0] + rng.random(m)) > 0.9).astype(float)
    wv = rng.uniform(0.0, 2.0, m) if weighted else None
    gb = GradientBoostingRDT(n_estimators=8, max_depth=1, learning_rate=0.3, random_state=0, eval_metric="auc")
    gb.fit(X, y, eval_set=[(Xv, yv)], eval_sample_weight=wv)
    stages = list(gb.staged_decision_function(Xv))
    assert max(len(np.unique(d)) for d in stages) <= 4  # heavily tied
    expected = [roc_auc_score(yv, d, sample_weight=wv) for d in stages]
    np.testing.assert_allclose(gb.evals_result_["auc"], expected, rtol=1e-12, atol=1e-12)
    # Hand check of the tie rule on the first round's scores.
    d = stages[0]
    w = np.ones(m) if wv is None else wv
    pos, neg = yv == 1, yv == 0
    num = sum(w[i] * w[j] * (1.0 if d[i] > d[j] else 0.5 if d[i] == d[j] else 0.0)
              for i in np.flatnonzero(pos) for j in np.flatnonzero(neg))
    assert gb.evals_result_["auc"][0] == pytest.approx(num / (w[pos].sum() * w[neg].sum()), rel=1e-12)


def test_logloss_clips_confident_wrong_predictions_like_sklearn():
    """Saturated probabilities (raw scores past +-37, where predict_proba
    rounds to exactly 0 / 1): -log is clipped at float64 eps, as log_loss."""
    X = np.repeat(np.arange(6.0), 40)[:, None]
    y = (X[:, 0] >= 3).astype(float)
    Xv = np.arange(6.0)[:, None]
    yv = 1.0 - (Xv[:, 0] >= 3)  # every eval label wrong
    gb = GradientBoostingRDT(n_estimators=3, max_depth=1, learning_rate=20.0, split_criterion="newton",
                             reg_lambda=0.0, min_child_weight=0.0, random_state=0)
    gb.fit(X, y, eval_set=(Xv, yv))
    assert np.abs(gb.decision_function(Xv)).min() > 37
    expected = [log_loss(yv, pr, labels=[0, 1]) for pr in gb.staged_predict_proba(Xv)]
    np.testing.assert_allclose(gb.evals_result_["logloss"], expected, rtol=1e-12, atol=1e-12)
    assert gb.evals_result_["logloss"][-1] == pytest.approx(-np.log(np.finfo(float).eps), rel=1e-12)


def test_patience_does_not_change_the_recorded_metric():
    X, y, Xv, yv, wv = _metric_problem("binary", seed=5)
    kw = dict(n_estimators=60, max_depth=3, learning_rate=0.8, random_state=2, eval_metric="logloss")
    rec = GradientBoostingRDT(**kw).fit(X, y, eval_set=(Xv, yv), eval_sample_weight=wv)
    es = GradientBoostingRDT(early_stopping_rounds=3, **kw).fit(X, y, eval_set=(Xv, yv), eval_sample_weight=wv)
    run = len(es.evals_result_["logloss"])
    assert run < 60
    assert es.evals_result_["logloss"] == rec.evals_result_["logloss"][:run]


# --------------------------------------------------------------------------- 3. stopping rule
def test_stops_at_best_plus_patience_and_truncates():
    X, y, Xv, yv, _ = _metric_problem("binary", seed=11)
    gb = GradientBoostingRDT(n_estimators=200, max_depth=3, learning_rate=1.0, random_state=0,
                             early_stopping_rounds=5).fit(X, y, eval_set=(Xv, yv))
    vals = gb.evals_result_["logloss"]
    assert len(vals) == gb.best_iteration_ + 5 + 1 < 200
    assert gb.best_score_ == vals[gb.best_iteration_] == min(vals)
    assert all(v >= gb.best_score_ for v in vals[gb.best_iteration_ + 1:])
    assert gb.n_estimators_ == gb.best_iteration_ + 1 == len(gb.train_score_) == len(gb.depth_features_)
    assert len(list(gb.staged_predict_proba(Xv))) == gb.n_estimators_


def test_ties_are_not_improvements():
    """A perfectly separated eval set: error 0 from the first round on, so the
    best stays round 0 and the run stops after the patience."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (300, 2))
    y = (X[:, 0] > 0).astype(float)
    Xv = np.column_stack([np.r_[np.full(30, -0.9), np.full(30, 0.9)], rng.uniform(-1, 1, 60)])
    yv = (Xv[:, 0] > 0).astype(float)
    gb = GradientBoostingRDT(n_estimators=50, max_depth=1, learning_rate=0.5, random_state=0,
                             early_stopping_rounds=4, eval_metric="error").fit(X, y, eval_set=(Xv, yv))
    assert gb.evals_result_["error"] == [0.0] * 5
    assert gb.best_iteration_ == 0 and gb.best_score_ == 0.0 and gb.n_estimators_ == 1


def test_gains_within_tol_are_not_improvements():
    X, y, Xv, yv, _ = _metric_problem("binary", seed=13)
    kw = dict(n_estimators=150, max_depth=2, learning_rate=0.2, random_state=0)
    vals = GradientBoostingRDT(**kw).fit(X, y, eval_set=(Xv, yv)).evals_result_["logloss"]
    for tol in (0.0, 1e-3, 1e-2):
        gb = GradientBoostingRDT(early_stopping_rounds=3, tol=tol, **kw).fit(X, y, eval_set=(Xv, yv))
        best, run = _expected_stop(vals, 3, tol, False, 150)
        assert (gb.best_iteration_, len(gb.evals_result_["logloss"])) == (best, run)
    assert _expected_stop(vals, 3, 1e-2, False, 150) != _expected_stop(vals, 3, 0.0, False, 150)


def test_auc_is_maximised():
    X, y, Xv, yv, wv = _metric_problem("binary", seed=17)
    gb = GradientBoostingRDT(n_estimators=120, max_depth=3, learning_rate=1.0, random_state=0, eval_metric="auc",
                             early_stopping_rounds=6).fit(X, y, eval_set=(Xv, yv), eval_sample_weight=wv)
    vals = gb.evals_result_["auc"]
    assert gb.best_score_ == max(vals) == vals[gb.best_iteration_]
    assert vals.index(max(vals)) == gb.best_iteration_  # first occurrence: later ties do not count


def test_runs_to_n_estimators_when_patience_never_runs_out():
    X, y, Xv, yv, _ = _metric_problem("regression", seed=19)
    gb = GradientBoostingRDT(task="regression", n_estimators=30, max_depth=2, learning_rate=0.3, random_state=0,
                             early_stopping_rounds=30).fit(X, y, eval_set=(Xv, yv))
    vals = gb.evals_result_["rmse"]
    assert len(vals) == 30
    assert gb.best_iteration_ == int(np.argmin(vals)) and gb.n_estimators_ == gb.best_iteration_ + 1
    full = GradientBoostingRDT(task="regression", n_estimators=gb.n_estimators_, max_depth=2, learning_rate=0.3,
                               random_state=0).fit(X, y)
    assert np.array_equal(gb.predict(Xv), full.predict(Xv))


# --------------------------------------------------------------------------- 4. back-compat
def _holdout_kwargs():
    return dict(n_estimators=80, max_depth=2, learning_rate=0.5, random_state=4, n_iter_no_change=3)


def test_without_eval_set_the_defaults_resolve_to_the_old_values():
    X, y, Xv, yv, _ = _metric_problem("binary", seed=23)
    a = GradientBoostingRDT(**_holdout_kwargs()).fit(X, y)
    b = GradientBoostingRDT(validation_fraction=0.1, tol=1e-4, **_holdout_kwargs()).fit(X, y)
    c = GradientBoostingRDT(early_stopping_rounds=3, **{**_holdout_kwargs(), "n_iter_no_change": None}).fit(X, y)
    assert a.n_estimators_ < 80  # the internal holdout stopped it
    for other in (b, c):
        _assert_same_model(a, other, X, Xv, "classification")
    assert a.validation_fraction == 0.1 and a.tol is None
    assert a.best_iteration_ is None and a.best_score_ is None and a.evals_result_ is None


@pytest.mark.parametrize("tree_method", ["exact", "hist"])
def test_eval_set_without_patience_only_records(tree_method):
    X, y, Xv, yv, wv = _metric_problem("multi", seed=29)
    kw = dict(n_estimators=12, max_depth=3, learning_rate=0.5, random_state=4, subsample=0.8, tree_method=tree_method)
    plain = GradientBoostingRDT(**kw).fit(X, y)
    rec = GradientBoostingRDT(**kw).fit(X, y, eval_set=(Xv, yv), eval_sample_weight=wv)
    _assert_same_model(plain, rec, X, Xv, "classification")
    assert len(rec.evals_result_["logloss"]) == 12 and rec.best_iteration_ is None


def test_refit_without_eval_set_clears_the_eval_attributes():
    X, y, Xv, yv, _ = _metric_problem("binary", seed=31)
    gb = GradientBoostingRDT(n_estimators=40, max_depth=2, learning_rate=1.0, random_state=0, early_stopping_rounds=2)
    gb.fit(X, y, eval_set=(Xv, yv))
    assert gb.best_iteration_ is not None
    gb.fit(X, y)  # internal holdout now
    assert gb.best_iteration_ is None and gb.best_score_ is None and gb.evals_result_ is None


# --------------------------------------------------------------------------- 5. validation
def _small():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] > 0).astype(float)
    return X, y


@pytest.mark.parametrize("bad,match", [
    (lambda X, y: [(X, y), (X, y)], "single validation set"),
    (lambda X, y: [], "single validation set"),
    (lambda X, y: {"X": X}, "tuple"),
    (lambda X, y: [(X, y, y)], "pair"),
    (lambda X, y: (X[:, :2], y), "features"),
    (lambda X, y: (X[:, 0], y), "2D"),
    (lambda X, y: (X[:0], y[:0]), "0 samples"),
    (lambda X, y: (X, y[:-1]), "samples"),
    (lambda X, y: (X, np.c_[y, y]), "1D"),
    (lambda X, y: (np.where(np.arange(X.size).reshape(X.shape) == 5, np.nan, X), y), "NaN"),
    (lambda X, y: (X, np.where(np.arange(y.size) == 3, np.inf, y)), "NaN or infinity"),
    (lambda X, y: (X, np.where(np.arange(y.size) == 3, 7.0, y)), "not seen"),
])
@pytest.mark.parametrize("tree_method", ["exact", "hist"])
def test_bad_eval_set_raises(bad, match, tree_method):
    X, y = _small()
    with pytest.raises(ValueError, match=match):
        GradientBoostingRDT(n_estimators=3, tree_method=tree_method, early_stopping_rounds=2).fit(
            X, y, eval_set=bad(X, y))


def test_bad_eval_weights_and_metric_combinations_raise():
    X, y = _small()
    gb = GradientBoostingRDT(n_estimators=3, early_stopping_rounds=2)
    with pytest.raises(ValueError, match="without eval_set"):
        gb.fit(X, y, eval_sample_weight=np.ones(80))
    for w, match in ((np.ones(79), "eval_sample_weight"), (-np.ones(80), "non-negative"),
                     (np.zeros(80), "non-zero"), (np.full(80, np.nan), "NaN"), (np.ones((80, 2)), "1D")):
        with pytest.raises(ValueError, match=match):
            gb.fit(X, y, eval_set=(X, y), eval_sample_weight=w)
    with pytest.raises(ValueError, match="eval_metric"):
        GradientBoostingRDT(eval_metric="rmse")
    with pytest.raises(ValueError, match="eval_metric"):
        GradientBoostingRDT(task="regression", eval_metric="auc")
    with pytest.raises(ValueError, match="eval_metric"):
        GradientBoostingRDT(eval_metric="merror")
    with pytest.raises(ValueError, match="needs an eval_set"):
        GradientBoostingRDT(n_estimators=3, eval_metric="auc").fit(X, y)
    y3 = np.digitize(X[:, 0], [-0.5, 0.5]).astype(float)
    with pytest.raises(ValueError, match="binary"):
        GradientBoostingRDT(n_estimators=3, eval_metric="auc").fit(X, y3, eval_set=(X, y3))
    with pytest.raises(ValueError, match="both classes"):  # one class in y_val
        GradientBoostingRDT(n_estimators=3, eval_metric="auc").fit(X, y, eval_set=(X[y == 1], y[y == 1]))
    w = (y == 1).astype(float)  # positive weight on one class only
    with pytest.raises(ValueError, match="both classes"):
        GradientBoostingRDT(n_estimators=3, eval_metric="auc").fit(X, y, eval_set=(X, y), eval_sample_weight=w)
    for patience in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="int >= 1"):
            GradientBoostingRDT(n_estimators=3, early_stopping_rounds=patience).fit(X, y, eval_set=(X, y))
    for tol in (-1e-3, np.inf, np.nan):
        with pytest.raises(ValueError, match="tol"):
            GradientBoostingRDT(n_estimators=3, early_stopping_rounds=2, tol=tol).fit(X, y, eval_set=(X, y))


def test_validation_fraction_warning_only_when_set_explicitly():
    X, y = _small()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GradientBoostingRDT(n_estimators=5, early_stopping_rounds=2).fit(X, y, eval_set=(X, y))
        GradientBoostingRDT(n_estimators=5, validation_fraction=0.3).fit(X, y, eval_set=(X, y))  # no patience
        GradientBoostingRDT(n_estimators=5, validation_fraction=0.3, n_iter_no_change=2).fit(X, y)  # holdout
    with pytest.warns(UserWarning, match="validation_fraction=0.3 is ignored"):
        GradientBoostingRDT(n_estimators=5, validation_fraction=0.3, early_stopping_rounds=2).fit(
            X, y, eval_set=(X, y))


def test_eval_set_and_weight_list_forms():
    X, y, Xv, yv, wv = _metric_problem("binary", seed=37)
    kw = dict(n_estimators=30, max_depth=2, learning_rate=0.8, random_state=0, early_stopping_rounds=3)
    a = GradientBoostingRDT(**kw).fit(X, y, eval_set=(Xv, yv), eval_sample_weight=wv)
    b = GradientBoostingRDT(**kw).fit(X, y, eval_set=[(Xv, yv)], eval_sample_weight=[wv])
    c = GradientBoostingRDT(**kw).fit(X, y, eval_set=[[Xv.tolist(), yv.tolist()]], eval_sample_weight=list(wv))
    assert a.evals_result_ == b.evals_result_ == c.evals_result_
    assert a.best_iteration_ == b.best_iteration_ == c.best_iteration_


def test_training_weights_do_not_apply_to_the_eval_set():
    X, y, Xv, yv, _ = _metric_problem("binary", seed=41)
    w = np.random.default_rng(0).uniform(0.1, 3.0, len(y))
    gb = GradientBoostingRDT(n_estimators=10, max_depth=2, random_state=0, scale_pos_weight=4.0)
    gb.fit(X, y, sample_weight=w, eval_set=(Xv, yv))
    np.testing.assert_allclose(gb.evals_result_["logloss"], _recompute(gb, "logloss", Xv, yv, None), rtol=1e-12)


# --------------------------------------------------------------------------- determinism, threads, inputs
def test_deterministic():
    params, patience, data, ev = _scenario(5)
    a, b = _fit_es(params, patience, data, ev), _fit_es(params, patience, data, ev)
    assert a.evals_result_ == b.evals_result_ and a.best_iteration_ == b.best_iteration_
    _assert_same_model(a, b, data[0], ev[0], params["task"])


@pytest.mark.parametrize("kind,tree_method", [("binary", "hist"), ("multi", "exact"), ("multi", "hist"),
                                              ("regression", "exact")])
def test_n_jobs_does_not_change_the_result(kind, tree_method):
    # More eval rows than one row block (16384, for_blocks), so the blocks run in parallel.
    X, y, Xv, yv, wv = _metric_problem(kind, seed=43, n=600, m=40000, p=6, quantize=4)
    task = "regression" if kind == "regression" else "classification"
    fits = [GradientBoostingRDT(task=task, n_estimators=40, max_depth=3, learning_rate=0.8, random_state=0,
                                tree_method=tree_method, n_jobs=j, early_stopping_rounds=4,
                                eval_metric="error" if kind == "multi" else None).fit(
        X, y, eval_set=(Xv, yv), eval_sample_weight=wv) for j in (1, 2, 4)]
    for other in fits[1:]:
        assert other.evals_result_ == fits[0].evals_result_
        assert other.best_iteration_ == fits[0].best_iteration_
        _assert_same_model(fits[0], other, X, Xv[:500], task)


@pytest.mark.parametrize("hist_subtraction", [True, False], ids=["subtraction", "direct"])
@pytest.mark.parametrize("tree_method", ["exact", "hist"])
def test_eval_scores_are_the_staged_predictions_bitwise(tree_method, hist_subtraction):
    # The eval rows are scored in parallel row blocks (several blocks of 16384
    # here); each round's metric must be that of the serial staged
    # predictions, bit for bit. Unweighted RMSE / MAE are row-order sums of
    # exactly representable terms' roundings, reproduced here with cumsum.
    X, y, Xv, yv, _ = _metric_problem("regression", seed=59, n=800, m=40000, p=5, quantize=0)
    for metric in ("rmse", "mae"):
        m = GradientBoostingRDT(task="regression", n_estimators=15, max_depth=4, learning_rate=0.3, subsample=0.8,
                                random_state=3, tree_method=tree_method, hist_subtraction=hist_subtraction,
                                n_jobs=4, eval_metric=metric).fit(X, y, eval_set=(Xv, yv))
        expected = []
        for pred in m.staged_predict(_dense(Xv)):
            d = yv - pred
            terms = d * d if metric == "rmse" else np.abs(d)
            mean = np.cumsum(terms)[-1] / float(len(yv))
            expected.append(float(np.sqrt(mean)) if metric == "rmse" else float(mean))
        assert m.evals_result_[metric] == expected


@pytest.mark.parametrize("tree_method", ["exact", "hist"])
def test_eval_set_layouts_and_dtypes_agree(tree_method):
    X, y, Xv, yv, wv = _metric_problem("binary", seed=47)
    Xv32 = Xv.astype(np.float32)  # values exactly representable both ways
    kw = dict(n_estimators=25, max_depth=3, learning_rate=0.8, random_state=0, tree_method=tree_method,
              early_stopping_rounds=3)
    base = GradientBoostingRDT(**kw).fit(X, y, eval_set=(Xv32.astype(np.float64), yv), eval_sample_weight=wv)
    big = np.zeros((Xv.shape[0] * 2, Xv.shape[1] * 3), dtype=np.float32)
    big[::2, ::3] = Xv32
    for layout in (Xv32, np.asfortranarray(Xv32), big[::2, ::3], np.asfortranarray(Xv32.astype(np.float64))):
        other = GradientBoostingRDT(**kw).fit(X, y, eval_set=(layout, yv), eval_sample_weight=wv)
        assert other.evals_result_ == base.evals_result_ and other.best_iteration_ == base.best_iteration_


def test_hist_eval_set_memmap(tmp_path):
    X, y, Xv, yv, _ = _metric_problem("binary", seed=53)
    path = tmp_path / "xv.f32"
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=Xv.shape, order="F")
    mm[:] = Xv
    mm.flush()
    ro = np.memmap(path, dtype=np.float32, mode="r", shape=Xv.shape, order="F")
    kw = dict(n_estimators=20, max_depth=2, random_state=0, tree_method="hist", early_stopping_rounds=3)
    a = GradientBoostingRDT(**kw).fit(X, y, eval_set=(ro, yv))
    b = GradientBoostingRDT(**kw).fit(X, y, eval_set=(np.asarray(Xv.astype(np.float32), dtype=np.float64), yv))
    assert a.evals_result_ == b.evals_result_


_EVAL_MEMORY_SCRIPT = PEAK_RSS + textwrap.dedent("""
    import sys
    import numpy as np
    from rdt.ensemble import GradientBoostingRDT

    rng = np.random.default_rng(0)
    m, p = 40000, 1000                                   # X_val: 160 MB of float32, Fortran order
    Xv = np.empty((m, p), dtype=np.float32, order="F")
    for i in range(0, m, 500):
        Xv[i:i + 500] = rng.standard_normal((500, p))
    yv = (Xv[:, 0] + Xv[:, 1] > 0).astype(float)
    X = np.ascontiguousarray(Xv[:300], dtype=np.float32)
    y = yv[:300].copy()
    gb = GradientBoostingRDT(n_estimators=3, max_depth=2, tree_method="hist", random_state=0,
                             early_stopping_rounds=2)
    gb.fit(X[:50], y[:50], eval_set=(Xv[:50], yv[:50]))  # warm up allocators and imports
    before = peak()
    gb.fit(X, y, eval_set=(Xv, yv))
    print(peak() - before, Xv.nbytes)
""")


def test_hist_float32_eval_set_is_not_copied():
    require_peak_rss()
    out = subprocess.run([sys.executable, "-c", _EVAL_MEMORY_SCRIPT], capture_output=True, text=True,
                         check=True).stdout.split()
    growth, xv_bytes = int(out[0]), int(out[1])
    # A float64 copy would be 2 * xv_bytes, a float32 one xv_bytes; the eval
    # state is a few doubles per eval row (~1 MB here).
    assert growth < 0.25 * xv_bytes, f"fit grew peak RSS by {growth / 2**20:.0f} MiB (X_val is {xv_bytes / 2**20:.0f} MiB)"


def test_fit_with_eval_set_releases_the_gil():
    """Most of this fit is routing / scoring the large eval set (AUC sorts it
    every round); a Python thread keeps running throughout."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 10))
    y = (X[:, 0] + rng.standard_normal(2000) > 0).astype(float)
    Xv = rng.standard_normal((100000, 10))
    yv = (Xv[:, 0] + rng.standard_normal(100000) > 0).astype(float)
    gb = GradientBoostingRDT(n_estimators=40, max_depth=4, random_state=0, eval_metric="auc",
                             early_stopping_rounds=1000)
    ticks, stop = [0], threading.Event()

    def spin():
        while not stop.is_set():
            ticks[0] += 1
            time.sleep(0.0005)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    time.sleep(0.05)
    start_ticks = ticks[0]
    t0 = time.perf_counter()
    gb.fit(X, y, eval_set=(Xv, yv))
    elapsed = time.perf_counter() - t0
    during = ticks[0] - start_ticks
    stop.set()
    t.join()
    assert len(gb.evals_result_["auc"]) == 40
    # With the GIL held for the whole native call the counter cannot advance at all.
    assert during >= 10 or elapsed < 0.05, f"background thread ran {during} times during a {elapsed:.2f}s fit"
