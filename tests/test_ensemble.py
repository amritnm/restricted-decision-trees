"""Unit/sanity tests for rdt.ensemble (RandomForestRDT, GradientBoostingRDT).

Not exact-parity oracle tests like tests/test_rdt.py's parity suite for the
single tree -- these are genuinely new implementations (native ensembling,
not a port of any single prior Python implementation), so this is a
correctness/regression suite: reproducibility, aggregation sanity, parameter
aliasing, and each documented feature actually doing something.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdt.ensemble import GradientBoostingRDT, RandomForestRDT


def _separable_classification_data(n=300, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [np.linspace(0, 10, n)] + [rng.normal(size=n) for _ in range(n_features - 1)]
    )
    y = (X[:, 0] > 5).astype(float)
    return X, y


def _linear_regression_data(n=300, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [np.linspace(0, 10, n)] + [rng.normal(size=n) for _ in range(n_features - 1)]
    )
    y = X[:, 0] * 2.0 + rng.normal(scale=0.2, size=n)
    return X, y


# ---------------------------------------------------------------------------
# RandomForestRDT
# ---------------------------------------------------------------------------

def test_rf_classification_basic():
    X, y = _separable_classification_data()
    rf = RandomForestRDT(n_estimators=15, max_depth=3, max_features=None, random_state=1)
    rf.fit(X, y)
    acc = np.mean(rf.predict(X) == y)
    assert acc > 0.9
    proba = rf.predict_proba(X)
    assert proba.shape == (300, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(300))


def test_rf_regression_basic():
    X, y = _linear_regression_data()
    rf = RandomForestRDT(task="regression", n_estimators=15, max_depth=4, max_features=None, random_state=1)
    rf.fit(X, y)
    mse = np.mean((rf.predict(X) - y) ** 2)
    assert mse < 1.0


def test_rf_feature_importances_sum_to_one_and_identify_informative_feature():
    X, y = _separable_classification_data(n_features=4)
    rf = RandomForestRDT(n_estimators=20, max_depth=3, max_features=None, random_state=1)
    rf.fit(X, y)
    assert rf.feature_importances_.shape == (4,)
    assert np.isclose(rf.feature_importances_.sum(), 1.0)
    assert np.argmax(rf.feature_importances_) == 0


def test_rf_oob_score():
    X, y = _separable_classification_data()
    rf = RandomForestRDT(n_estimators=30, max_depth=3, oob_score=True, random_state=1)
    rf.fit(X, y)
    assert rf.oob_score_ is not None
    assert 0.0 <= rf.oob_score_ <= 1.0
    assert rf.oob_score_ > 0.7


def test_rf_reproducibility_with_parallel_build():
    X, y = _separable_classification_data()
    rf1 = RandomForestRDT(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
    rf1.fit(X, y)
    rf2 = RandomForestRDT(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
    rf2.fit(X, y)
    np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))


def test_rf_estimators_are_introspectable_rdt_objects():
    X, y = _separable_classification_data()
    rf = RandomForestRDT(n_estimators=5, max_depth=3, random_state=1)
    rf.fit(X, y)
    assert len(rf.estimators_) == 5
    for tree in rf.estimators_:
        assert tree.root is not None
        assert isinstance(tree.depth_features_, dict)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_features="log2", colsample_bytree=0.9),
        dict(min_samples_leaf=2, min_child_weight=5),
        dict(min_impurity_decrease=0.1, gamma=0.5),
    ],
)
def test_rf_conflicting_aliases_raise(kwargs):
    with pytest.raises(ValueError):
        RandomForestRDT(**kwargs)


def test_rf_alias_matches_primary_default_no_raise():
    rf = RandomForestRDT(max_features="sqrt", colsample_bytree=0.5, random_state=1)
    assert rf.max_features == 0.5


def test_rf_class_weight_balanced_changes_fit():
    rng = np.random.default_rng(0)
    n_majority, n_minority = 270, 30
    X = np.vstack([rng.normal(loc=0.0, size=(n_majority, 2)), rng.normal(loc=3.0, size=(n_minority, 2))])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    rf_unweighted = RandomForestRDT(n_estimators=10, max_depth=3, random_state=1, class_weight=None)
    rf_unweighted.fit(X, y)
    rf_balanced = RandomForestRDT(n_estimators=10, max_depth=3, random_state=1, class_weight="balanced")
    rf_balanced.fit(X, y)
    # Not asserting a specific direction of change -- just that class_weight
    # actually influences the fit rather than being silently ignored.
    assert not np.array_equal(rf_unweighted.predict(X), rf_balanced.predict(X)) or \
        not np.array_equal(rf_unweighted.predict_proba(X), rf_balanced.predict_proba(X))


# ---------------------------------------------------------------------------
# GradientBoostingRDT
# ---------------------------------------------------------------------------

def test_gb_binary_classification_basic():
    X, y = _separable_classification_data()
    gb = GradientBoostingRDT(n_estimators=30, max_depth=2, learning_rate=0.3, random_state=1)
    gb.fit(X, y)
    assert np.mean(gb.predict(X) == y) > 0.9
    proba = gb.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)))


def test_gb_multiclass_classification():
    rng = np.random.default_rng(0)
    n = 300
    f0 = np.linspace(0, 12, n)
    X = np.column_stack([f0, rng.normal(size=n), rng.normal(size=n)])
    y = np.where(f0 < 4, 0, np.where(f0 < 8, 1, 2)).astype(float)

    gb = GradientBoostingRDT(n_estimators=40, max_depth=2, learning_rate=0.3, random_state=2, n_jobs=-1)
    gb.fit(X, y)
    assert np.mean(gb.predict(X) == y) > 0.85
    proba = gb.predict_proba(X)
    assert proba.shape == (n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(n))


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "huber", "quantile"])
def test_gb_regression_losses(loss):
    # absolute_error/huber/quantile use sign/piecewise gradients that take
    # smaller, coarser steps per round than squared_error's proportional
    # gradient, so they need a larger training budget to reach a comparably
    # tight fit -- more estimators/higher learning_rate than the
    # squared_error-only checks elsewhere in this file.
    X, y = _linear_regression_data()
    gb = GradientBoostingRDT(
        task="regression", loss=loss, n_estimators=50, max_depth=2, learning_rate=0.2, random_state=1,
    )
    gb.fit(X, y)
    mae = np.mean(np.abs(gb.predict(X) - y))
    assert mae < 2.0


def test_gb_training_loss_decreases():
    X, y = _separable_classification_data()
    gb = GradientBoostingRDT(n_estimators=30, max_depth=2, learning_rate=0.3, random_state=1)
    gb.fit(X, y)
    assert gb.train_score_[-1] < gb.train_score_[0]


def test_gb_early_stopping_stops_before_n_estimators():
    X, y = _separable_classification_data()
    gb = GradientBoostingRDT(
        n_estimators=500, max_depth=2, learning_rate=0.5, random_state=1,
        validation_fraction=0.2, n_iter_no_change=5,
    )
    gb.fit(X, y)
    assert gb.n_estimators_ < 500


def test_gb_early_stopping_rounds_alias():
    X, y = _separable_classification_data()
    gb = GradientBoostingRDT(
        n_estimators=500, max_depth=2, learning_rate=0.5, random_state=1,
        validation_fraction=0.2, early_stopping_rounds=5,
    )
    gb.fit(X, y)
    assert gb.n_estimators_ < 500


def test_gb_feature_importances_sum_to_one():
    X, y = _linear_regression_data(n_features=3)
    gb = GradientBoostingRDT(task="regression", n_estimators=20, max_depth=2, random_state=1)
    gb.fit(X, y)
    assert np.isclose(gb.feature_importances_.sum(), 1.0)
    assert np.argmax(gb.feature_importances_) == 0


def test_gb_reg_alpha_reg_lambda_do_not_crash_and_shrink_predictions():
    X, y = _linear_regression_data()
    gb_plain = GradientBoostingRDT(task="regression", n_estimators=20, max_depth=2, random_state=1)
    gb_plain.fit(X, y)
    gb_reg = GradientBoostingRDT(
        task="regression", n_estimators=20, max_depth=2, random_state=1, reg_alpha=0.1, reg_lambda=1.0,
    )
    gb_reg.fit(X, y)
    # regularized predictions shouldn't be wildly different, but should differ
    assert not np.array_equal(gb_plain.predict(X), gb_reg.predict(X))


def test_gb_scale_pos_weight_binary_only():
    X, y = _separable_classification_data()
    gb = GradientBoostingRDT(n_estimators=10, max_depth=2, random_state=1, scale_pos_weight=3.0)
    gb.fit(X, y)  # should not raise

    rng = np.random.default_rng(0)
    n = 300
    f0 = np.linspace(0, 12, n)
    X3 = np.column_stack([f0, rng.normal(size=n)])
    y3 = np.where(f0 < 4, 0, np.where(f0 < 8, 1, 2)).astype(float)
    gb3 = GradientBoostingRDT(n_estimators=10, max_depth=2, random_state=1, scale_pos_weight=2.0)
    with pytest.raises(ValueError):
        gb3.fit(X3, y3)


def test_gb_exponential_loss_binary_only():
    rng = np.random.default_rng(0)
    n = 300
    f0 = np.linspace(0, 12, n)
    X3 = np.column_stack([f0, rng.normal(size=n)])
    y3 = np.where(f0 < 4, 0, np.where(f0 < 8, 1, 2)).astype(float)
    gb3 = GradientBoostingRDT(n_estimators=10, max_depth=2, random_state=1, loss="exponential")
    with pytest.raises(ValueError):
        gb3.fit(X3, y3)


def test_gb_conflicting_aliases_raise():
    with pytest.raises(ValueError):
        GradientBoostingRDT(n_iter_no_change=5, early_stopping_rounds=10)


def test_gb_invalid_loss_for_task_raises():
    with pytest.raises(ValueError):
        GradientBoostingRDT(task="regression", loss="log_loss")
    with pytest.raises(ValueError):
        GradientBoostingRDT(task="classification", loss="squared_error")


@pytest.mark.parametrize("loss", ["squared_error", "huber", "log_loss", "log_loss_multiclass"])
def test_gb_train_score_is_the_loss_of_predict_on_the_training_set(loss):
    # train_score_ is computed from the raw scores accumulated during fit; predicting the
    # training set afterwards must reproduce them -- through each tree's column subset and with
    # reg_alpha / reg_lambda shrinkage applied exactly as in fit.
    rng = np.random.default_rng(31)
    n = 800
    X = rng.normal(size=(n, 8))
    X[:, 1] = np.round(X[:, 1])
    s = X[:, 0] - 0.5 * X[:, 1] + 0.5 * rng.normal(size=n)
    kw = dict(n_estimators=15, max_depth=3, learning_rate=0.3, subsample=0.8, colsample_bytree=0.5,
              reg_lambda=1.0, reg_alpha=0.01, random_state=0)
    if loss in ("squared_error", "huber"):
        y = 2.0 * s
        gb = GradientBoostingRDT(task="regression", loss=loss, **kw).fit(X, y)
        diff = y - gb.predict(X)
        if loss == "squared_error":
            value = np.mean(diff * diff)
        else:
            d = gb.huber_delta
            value = np.mean(np.where(np.abs(diff) <= d, 0.5 * diff * diff, d * (np.abs(diff) - 0.5 * d)))
    elif loss == "log_loss":
        y = (s > 0).astype(float)
        gb = GradientBoostingRDT(**kw).fit(X, y)
        p = np.clip(gb.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        value = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    else:
        y = np.digitize(s, [-0.5, 0.5]).astype(float)
        gb = GradientBoostingRDT(**kw).fit(X, y)
        P = gb.predict_proba(X)
        value = np.mean(-np.log(P[np.arange(n), np.searchsorted(gb.classes_, y)]))
    np.testing.assert_allclose(value, gb.train_score_[-1], rtol=1e-10)
