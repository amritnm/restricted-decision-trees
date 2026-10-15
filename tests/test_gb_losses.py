"""GradientBoostingRDT losses: leaf line search, initial scores, weights, links.

With a single feature an RDT is an ordinary CART tree (each node picks its own
threshold on the only feature), so scikit-learn's GradientBoostingClassifier /
GradientBoostingRegressor with criterion='squared_error' is an exact reference:
same init, same trees, same leaf values, same learning rate -> the same raw
scores after every round.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rdt.ensemble import GradientBoostingRDT


def _logit(p):
    return np.log(p / (1.0 - p))


def _sklearn_gb(kind, **kw):
    """sklearn GradientBoosting{Classifier,Regressor} with squared-error trees.

    criterion='squared_error' pins the tree criterion on older releases (whose
    default was friedman_mse -- same argmax per node, but explicit is clearer);
    newer releases deprecate the parameter because it is always that.
    """
    ensemble = pytest.importorskip("sklearn.ensemble")
    cls = getattr(ensemble, f"GradientBoosting{kind}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return cls(criterion="squared_error", **kw)


def _sk_fit(est, X, y, w=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return est.fit(X, y, sample_weight=w)


def _grid_feature(rng, n):
    # exactly representable in float32 (sklearn casts X) and well separated
    return (rng.integers(0, 4000, size=n) / 16.0)[:, None]


def _binary_problem(seed, n, prior):
    rng = np.random.default_rng(seed)
    X = _grid_feature(rng, n)
    z = (X[:, 0] - 125.0) / 40.0
    p = 1.0 / (1.0 + np.exp(-(_logit(prior) + 1.5 * z)))
    y = (rng.uniform(size=n) < p).astype(float)
    y[:2] = [0.0, 1.0]
    return rng, X, y



# ---------------------------------------------------------------------------
# scikit-learn parity, round by round (one feature: an RDT is a CART tree)
# ---------------------------------------------------------------------------

def test_binary_scores_match_sklearn_every_round():
    for loss, prior, weighted, max_depth in (
            ("log_loss", 0.01, False, 1), ("log_loss", 0.01, True, 1), ("log_loss", 0.2, False, 2),
            ("log_loss", 0.2, True, 2), ("log_loss", 0.5, True, 1), ("exponential", 0.2, True, 2)):
        rng, X, y = _binary_problem(int(prior * 1000) + max_depth, 3000, prior)
        w = rng.uniform(0.2, 3.0, size=len(y)) if weighted else None
        lr = 0.1 if max_depth == 1 else 0.3
        sk = _sk_fit(_sklearn_gb("Classifier", loss=loss, n_estimators=20, learning_rate=lr,
                                 max_depth=max_depth, random_state=0), X, y, w)
        staged = list(sk.staged_decision_function(X))
        for k in (1, 2, 5, 20):
            gb = GradientBoostingRDT(loss=loss, n_estimators=k, learning_rate=lr, max_depth=max_depth,
                                     random_state=0).fit(X, y, sample_weight=w)
            raw = staged[k - 1].ravel()
            # exponential: the raw score is half the log-odds in both libraries
            expected = 1.0 / (1.0 + np.exp(-(2.0 * raw if loss == "exponential" else raw)))
            np.testing.assert_allclose(gb.predict_proba(X)[:, 1], expected, rtol=1e-9, atol=1e-12)


def test_multiclass_and_regression_match_sklearn_every_round():
    rng = np.random.default_rng(4)
    n = 2000
    X = _grid_feature(rng, n)
    y = np.digitize(X[:, 0] + rng.normal(scale=30.0, size=n), np.quantile(X[:, 0], [0.25, 0.5, 0.75])).astype(float)
    y[y == 2] = np.where(rng.uniform(size=(y == 2).sum()) < 0.9, 1.0, 2.0)  # a rare class
    for w in (None, rng.uniform(0.2, 3.0, size=n)):
        sk = _sk_fit(_sklearn_gb("Classifier", n_estimators=10, learning_rate=0.2, max_depth=2, random_state=0),
                     X, y, w)
        staged = list(sk.staged_predict_proba(X))
        for k in (1, 10):
            gb = GradientBoostingRDT(n_estimators=k, learning_rate=0.2, max_depth=2, random_state=0).fit(
                X, y, sample_weight=w)
            np.testing.assert_allclose(gb.predict_proba(X), staged[k - 1], rtol=1e-9, atol=1e-12)
    yr = 1000.0 + np.sin(X[:, 0] / 20.0) * 5 + rng.normal(size=n)
    w = rng.uniform(0.2, 3.0, size=n)
    sk = _sk_fit(_sklearn_gb("Regressor", n_estimators=15, learning_rate=0.2, max_depth=2, random_state=0), X, yr, w)
    gb = GradientBoostingRDT(task="regression", n_estimators=15, learning_rate=0.2, max_depth=2,
                             random_state=0).fit(X, yr, sample_weight=w)
    np.testing.assert_allclose(gb.predict(X), sk.predict(X), rtol=1e-12)


def test_rare_event_leaf_steps_are_newton_steps():
    # Original report: with a ~1% prior the mean residual is p(1-p) ~ 100x too
    # small a step, so probabilities stayed glued to the prior.
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 10, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-7.5 + 0.5 * x)))).astype(float)
    X = x[:, None]
    gb = GradientBoostingRDT(n_estimators=1, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
    prior = y.mean()
    thr = _sk_fit(_sklearn_gb("Classifier", n_estimators=1, learning_rate=1.0, max_depth=1), X, y) \
        .estimators_[0, 0].tree_.threshold[0]
    for mask in (x <= thr, x > thr):
        r = y[mask] - prior
        newton = r.sum() / (prior * (1 - prior) * mask.sum())
        step = _logit(gb.predict_proba(X[mask][:1])[0, 1]) - _logit(prior)
        assert step == pytest.approx(newton, rel=1e-9)


# ---------------------------------------------------------------------------
# Initial scores honour sample_weight / scale_pos_weight
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Initial scores, train_score_ and early stopping honour the weights
# ---------------------------------------------------------------------------

def _constant_X(n):
    return np.zeros((n, 1))  # no split possible: isolates init + leaf values


def _weighted_quantile_ref(y, w, rank):
    # first sorted value whose cumulative weight, rescaled to sum to n, exceeds `rank`
    o = np.argsort(y, kind="stable")
    cum = np.cumsum(w[o]) * len(y) / w.sum()
    return y[o][np.argmax(cum > rank)]


def test_weighted_init_is_the_weighted_prior():
    y = np.r_[np.ones(100), np.zeros(900)]
    X = _constant_X(len(y))
    for loss in ("log_loss", "exponential"):
        for pos_w in (1.0, 9.0, 0.3):
            w = np.where(y == 1, pos_w, 1.0)
            wprior = (w * y).sum() / w.sum()
            for kwargs, extra in ((dict(sample_weight=w), {}), ({}, dict(scale_pos_weight=pos_w))):
                gb = GradientBoostingRDT(loss=loss, n_estimators=1, learning_rate=1e-12, random_state=0, **extra)
                # exponential: also checks the sigmoid(2F) link (0.2.0 gave 0.25 for a 0.1 prior)
                assert gb.fit(X, y, **kwargs).predict_proba(X[:1])[0, 1] == pytest.approx(wprior, rel=1e-9)
    # multiclass now starts from the weighted log class priors (0.2.0: uniform)
    rng = np.random.default_rng(0)
    y = rng.choice([0.0, 1.0, 2.0, 5.0], p=[0.6, 0.3, 0.09, 0.01], size=3000)
    w = rng.uniform(0.1, 4.0, size=len(y))
    prior = np.array([w[y == c].sum() for c in (0.0, 1.0, 2.0, 5.0)]) / w.sum()
    gb = GradientBoostingRDT(n_estimators=1, learning_rate=1e-12, random_state=0).fit(
        _constant_X(len(y)), y, sample_weight=w)
    np.testing.assert_allclose(gb.predict_proba(X[:1])[0], prior, rtol=1e-9)


def test_regression_init_is_weighted():
    rng = np.random.default_rng(1)
    n = 501
    y = rng.lognormal(size=n) * 10
    w = rng.uniform(0.0, 3.0, size=n)
    X = _constant_X(n)
    expected = {"squared_error": np.sum(w * y) / w.sum(), "huber": np.sum(w * y) / w.sum(),
                "absolute_error": _weighted_quantile_ref(y, w, n / 2.0),
                "quantile": _weighted_quantile_ref(y, w, 0.8 * (n - 1))}
    for loss, value in expected.items():
        gb = GradientBoostingRDT(task="regression", loss=loss, n_estimators=1, learning_rate=1e-12,
                                 quantile_alpha=0.8, random_state=0).fit(X, y, sample_weight=w)
        assert gb.predict(X[:1])[0] == pytest.approx(value, rel=1e-9), loss


def test_train_score_is_the_weighted_training_loss():
    rng = np.random.default_rng(5)
    n = 1000
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(size=n) > 1.0).astype(float)
    w = np.where(y == 1, 5.0, 1.0) * rng.uniform(0.5, 1.5, size=n)
    gb = GradientBoostingRDT(n_estimators=7, max_depth=2, random_state=0).fit(X, y, sample_weight=w)
    p = gb.predict_proba(X)[:, 1]
    ll = -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum()
    assert gb.train_score_[-1] == pytest.approx(ll, rel=1e-9)


def test_early_stopping_with_class_balancing_weights():
    # Practical symptom of the unweighted init / validation loss: with
    # class-balancing weights on an ~80/20 problem with a weak signal, the
    # unweighted holdout loss got worse almost immediately and early
    # stopping fired after n_iter_no_change + 1 rounds (11 here on 0.2.0).
    datasets = pytest.importorskip("sklearn.datasets")
    X, y = datasets.make_classification(n_samples=6000, n_features=40, n_informative=2, n_redundant=0,
                                        n_clusters_per_class=1, weights=[0.2, 0.8], flip_y=0.4,
                                        class_sep=0.1, random_state=0)
    y = y.astype(float)
    w = (len(y) / (2.0 * np.bincount(y.astype(int))))[y.astype(int)]   # 'balanced'
    gb = GradientBoostingRDT(n_estimators=200, max_depth=3, learning_rate=0.1, validation_fraction=0.1,
                             n_iter_no_change=10, random_state=0).fit(X, y, sample_weight=w)
    assert gb.n_estimators_ > 11


def test_early_stopping_ignores_zero_weight_rows():
    # The holdout loss is the weighted loss: flipping the labels of rows with
    # weight 0 changes neither the stopping round nor train_score_.
    datasets = pytest.importorskip("sklearn.datasets")
    X, y = datasets.make_classification(n_samples=3000, n_features=8, weights=[0.7, 0.3], flip_y=0.1,
                                        random_state=0)
    y = y.astype(float)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, size=len(y))
    w[rng.uniform(size=len(y)) < 0.3] = 0.0
    y_flipped = np.where(w == 0, 1.0 - y, y)
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.2, validation_fraction=0.2, n_iter_no_change=5,
              random_state=0)
    a = GradientBoostingRDT(**kw).fit(X, y, sample_weight=w)
    b = GradientBoostingRDT(**kw).fit(X, y_flipped, sample_weight=w)
    assert a.n_estimators_ == b.n_estimators_ < 300
    np.testing.assert_allclose(a.train_score_, b.train_score_, rtol=1e-12)


def test_zero_weight_training_split_or_holdout_raises():
    # The early-stopping split can leave all the weight on one side. With no
    # weight on the holdout, early stopping used to fire after
    # n_iter_no_change + 1 rounds; with none on the training rows the model
    # was a clipped constant. scikit-learn raises in both cases.
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(float)
    for validation_fraction, expected in ((0.5, "training rows"), (0.05, "validation rows")):
        messages = []
        for rs in range(40):
            w = np.zeros(n)
            w[rng.choice(n, 2, replace=False)] = 1.0   # only two weighted rows
            try:
                GradientBoostingRDT(n_estimators=20, max_depth=2, n_iter_no_change=3,
                                    validation_fraction=validation_fraction, random_state=rs).fit(
                    X, y, sample_weight=w)
            except ValueError as e:
                assert "zero total sample_weight" in str(e)
                messages.append(str(e))
        assert any(expected in m for m in messages), (validation_fraction, messages[:2])
    # weight on both sides: fits normally
    w = np.ones(n)
    GradientBoostingRDT(n_estimators=5, n_iter_no_change=3, validation_fraction=0.2, random_state=0).fit(
        X, y, sample_weight=w)


# ---------------------------------------------------------------------------
# Property sweeps (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("prior", [0.01, 0.2, 0.5])
@pytest.mark.parametrize("weights", ["none", "random", "zeros"])
@pytest.mark.parametrize("multiclass", [False, True])
def test_log_loss_train_loss_decreases_monotonically(prior, weights, multiclass):
    rng = np.random.default_rng(int(prior * 100))
    n = 2000
    X = rng.normal(size=(n, 5))
    X[:, 4] = rng.integers(0, 3, size=n)
    z = X[:, 0] + 0.5 * X[:, 1] * X[:, 4] + rng.normal(size=n)
    if multiclass:
        y = np.where(z > np.quantile(z, 1 - prior), 2.0, np.where(z > np.quantile(z, 0.5), 1.0, 0.0))
    else:
        y = (z > np.quantile(z, 1 - prior)).astype(float)
    w = None
    if weights != "none":
        w = rng.uniform(0.2, 3.0, size=n)
        if weights == "zeros":
            w[rng.uniform(size=n) < 0.3] = 0.0
    gb = GradientBoostingRDT(n_estimators=40, max_depth=3, learning_rate=0.05, random_state=0).fit(
        X, y, sample_weight=w)
    assert np.all(np.diff(gb.train_score_) <= 1e-12)
    np.testing.assert_allclose(gb.predict_proba(X).sum(axis=1), 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("n", [10, 1000, 20000])
@pytest.mark.parametrize("p", [1, 5, 50])
@pytest.mark.parametrize("config", ["plain", "subsample", "colsample", "early_stopping", "weights_zeros",
                                    "scale_pos_weight", "exponential", "multiclass"])
def test_gb_properties_across_scenarios(n, p, config):
    rng = np.random.default_rng(n + p)
    X = rng.normal(size=(n, p))
    if p > 1:
        X[:, -1] = rng.integers(0, 2, size=n)
    z = X[:, 0] + rng.normal(size=n)
    y = (z > np.quantile(z, 0.9)).astype(float)
    y[:2] = [0.0, 1.0]
    w = None
    kw = dict(n_estimators=8, max_depth=3 if n * p < 10**6 else 2, learning_rate=0.1, random_state=1)
    if config == "subsample":
        kw["subsample"] = 0.6
    elif config == "colsample":
        kw["colsample_bytree"] = 0.5
    elif config == "early_stopping":
        kw.update(n_estimators=30, validation_fraction=0.2, n_iter_no_change=3)
    elif config == "weights_zeros":
        w = rng.uniform(0.0, 2.0, size=n)
        w[rng.uniform(size=n) < 0.3] = 0.0
        w[:2] = 1.0
    elif config == "scale_pos_weight":
        kw["scale_pos_weight"] = 9.0
    elif config == "exponential":
        kw["loss"] = "exponential"
    elif config == "multiclass":
        y = np.digitize(z, np.quantile(z, [0.6, 0.97])).astype(float)
        y[:3] = [0.0, 1.0, 2.0]
    gb = GradientBoostingRDT(**kw).fit(X, y, sample_weight=w)
    P = gb.predict_proba(X)
    assert np.all(np.isfinite(P)) and np.all((P >= 0) & (P <= 1))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    par = GradientBoostingRDT(**{**kw, "n_jobs": 2}).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(par.predict_proba(X), P)
