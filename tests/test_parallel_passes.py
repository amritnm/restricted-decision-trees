"""The per-row passes of gradient boosting run in row blocks on the model's threads.

Per round: gradients / hessians, the gather of the round's rows, routing every row to its leaf of the
new tree and the score update are per-row work; the training and early-stopping losses compute each
row's loss in parallel and sum them serially in row order. Prediction (predict, predict_proba,
decision_function, staged predictions) routes row blocks in parallel, each row adding the trees in
round order. None of it may change a result: these tests fit and predict on enough rows for several
blocks (16384 rows each) and compare n_jobs=1 with n_jobs>1 bit for bit, for the training scores,
early stopping, importances and every prediction method. (The comparison with the previous, serial
build is done outside the test suite.)
"""

import numpy as np
import pytest

from rdt.ensemble import GradientBoostingRDT

N = 40000


def _data(kind, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 8))
    X[:, 4:] = np.round(X[:, 4:] * 2)
    s = X[:, 0] - X[:, 1] * X[:, 2] + 0.5 * X[:, 3] + rng.standard_normal(N)
    w = rng.uniform(0.0, 2.0, N)
    w[rng.random(N) < 0.05] = 0.0
    if kind == "regression":
        return X, s, w
    if kind == "multi":
        return X, np.digitize(s, np.quantile(s, [0.3, 0.7])).astype(float), w
    return X, (s > 0.5).astype(float), w


CASES = {
    "newton-binary": ("binary", dict(split_criterion="newton", subsample=0.8, colsample_bytree=0.8)),
    "newton-exponential": ("binary", dict(split_criterion="newton", loss="exponential")),
    "friedman-binary": ("binary", dict(subsample=0.7)),  # per-leaf Newton step on the gradients
    "friedman-regression": ("regression", dict(task="regression", reg_lambda=0.5, reg_alpha=0.01)),
    "newton-regression": ("regression", dict(task="regression", split_criterion="newton")),
    "multiclass": ("multi", dict(split_criterion="newton")),
}


# The exact search runs the same per-round passes; two cases suffice for it.
@pytest.mark.parametrize("case,tree_method", [(c, "hist") for c in CASES] +
                         [("newton-binary", "exact"), ("friedman-regression", "exact")])
def test_n_jobs_does_not_change_any_result(case, tree_method):
    kind, kw = CASES[case]
    X, y, w = _data(kind)
    fits = [GradientBoostingRDT(n_estimators=6, max_depth=4, n_iter_no_change=2, validation_fraction=0.2,
                                tol=0.0, tree_method=tree_method, random_state=3, n_jobs=j, **kw).fit(X, y,
                                                                                                    sample_weight=w)
            for j in (1, 3)]
    a, b = fits
    assert np.array_equal(a.train_score_, b.train_score_)
    assert a.n_estimators_ == b.n_estimators_
    assert np.array_equal(a.feature_importances_, b.feature_importances_)
    assert np.array_equal(a.predict(X), b.predict(X))
    if kind != "regression":
        assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
        assert np.array_equal(a.decision_function(X), b.decision_function(X))
        for sa, sb in zip(a.staged_decision_function(X), b.staged_decision_function(X)):
            assert np.array_equal(sa, sb)
    else:
        for sa, sb in zip(a.staged_predict(X), b.staged_predict(X)):
            assert np.array_equal(sa, sb)


def test_prediction_threads_match_serial_prediction():
    # Two identical fits (the fit does not depend on n_jobs), predicting serially and on 4 threads.
    X, y, w = _data("binary", seed=1)
    a, b = [GradientBoostingRDT(n_estimators=20, max_depth=5, split_criterion="newton", tree_method="hist",
                                random_state=0, n_jobs=j).fit(X, y, sample_weight=w) for j in (1, 4)]
    Xp = np.random.default_rng(2).standard_normal((300000, 8))
    assert np.array_equal(a.predict_proba(Xp), b.predict_proba(Xp))
    assert np.array_equal(a.decision_function(Xp), b.decision_function(Xp))
    for sa, sb in zip(a.staged_decision_function(Xp[:50000]), b.staged_decision_function(Xp[:50000])):
        assert np.array_equal(sa, sb)
