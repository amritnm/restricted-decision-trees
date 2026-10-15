"""Input validation for RDT / RandomForestRDT / GradientBoostingRDT.

The native core indexes y/sample_weight by row and X by fit-time column
numbers without bounds checks, and sorts feature values with ``<`` (undefined
behaviour once NaN is present), so invalid inputs used to be accepted silently
and produce garbage (or read out of bounds).
"""

from __future__ import annotations

import numpy as np
import pytest

from rdt import RDT, _rdt_native
from rdt.ensemble import GradientBoostingRDT, RandomForestRDT


def _models():
    return [RDT(max_depth=2), RandomForestRDT(n_estimators=3, max_depth=2, random_state=0),
            GradientBoostingRDT(n_estimators=3, max_depth=2, random_state=0)]


def _data(n=60, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] > 0).astype(float)
    y[:2] = [0.0, 1.0]
    return X, y


def test_nan_in_X_no_longer_collapses_the_tree():
    # NaN <=> positive label: the fit used to "succeed" with the root turned
    # into a leaf (NaN scored as a split point, NaN threshold).
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=1000) < 0.3).astype(float)
    x0 = rng.uniform(size=1000)
    x0[y == 1] = np.nan
    X = np.column_stack([x0, rng.normal(size=1000)])
    for model in _models():
        with pytest.raises(ValueError, match="NaN or infinity"):
            model.fit(X, y)


def test_non_finite_values_raise_at_fit_and_predict():
    X, y = _data()
    for bad in (np.nan, np.inf, -np.inf):
        for model in _models():
            X_bad = X.copy()
            X_bad[7, 1] = bad
            with pytest.raises(ValueError, match="NaN or infinity"):
                model.fit(X_bad, y)
            y_bad = y.copy()
            y_bad[5] = bad
            with pytest.raises(ValueError, match="NaN or infinity"):
                model.fit(X, y_bad)
            w = np.ones(len(y))
            w[4] = bad
            with pytest.raises(ValueError, match="NaN or infinity"):
                model.fit(X, y, sample_weight=w)
            model.fit(X, y)
            with pytest.raises(ValueError, match="NaN or infinity"):
                model.predict(X_bad)
            with pytest.raises(ValueError, match="NaN or infinity"):
                model.predict_proba(X_bad)


def test_wrong_number_of_features_at_predict_raises():
    X, y = _data(p=5)
    # only the last column matters: predicting on fewer columns used to read
    # other rows' values silently
    y = (X[:, 4] > 0).astype(float)
    for model in _models():
        model.fit(X, y)
        for p in (4, 6):
            with pytest.raises(ValueError, match="features"):
                model.predict(np.zeros((3, p)))
            with pytest.raises(ValueError, match="features"):
                model.predict_proba(np.zeros((3, p)))


def test_mismatched_lengths_and_bad_shapes_raise():
    X, y = _data(n=50)
    for model in _models():
        with pytest.raises(ValueError, match="y has"):
            model.fit(X, y[:20])
        with pytest.raises(ValueError, match="sample_weight has"):
            model.fit(X, y, sample_weight=np.ones(10))
        with pytest.raises(ValueError, match="2D"):
            model.fit(X[:, 0], y)
        with pytest.raises(ValueError, match="1D"):
            model.fit(X, y[:, None])
        with pytest.raises(ValueError, match="0 samples"):
            model.fit(X[:0], y[:0])


def test_invalid_effective_weights_raise():
    X, y = _data(n=80)
    neg = np.ones(len(y))
    neg[:10] = -1.0
    for model in _models():
        with pytest.raises(ValueError, match="non-zero"):
            model.fit(X, y, sample_weight=np.zeros(len(y)))
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X, y, sample_weight=neg)
    # weights produced by class_weight / scale_pos_weight are validated too
    for cw, msg in (({0.0: 0.0, 1.0: 0.0}, "non-zero"), ({0.0: np.nan, 1.0: 1.0}, "NaN"),
                    ({0.0: -2.0, 1.0: 1.0}, "non-negative")):
        with pytest.raises(ValueError, match=msg):
            RandomForestRDT(n_estimators=2, class_weight=cw, random_state=0).fit(X, y)
    for spw in (np.nan, np.inf, -1.0, 0.0):
        with pytest.raises(ValueError, match="scale_pos_weight"):
            GradientBoostingRDT(n_estimators=2, scale_pos_weight=spw, random_state=0).fit(X, y)


def test_gb_single_class_raises():
    # GB classification with one class wrote past the end of a buffer.
    X = np.random.default_rng(0).normal(size=(200, 3))
    for loss in ("log_loss", "exponential"):
        with pytest.raises(ValueError, match="at least 2 classes"):
            GradientBoostingRDT(n_estimators=5, loss=loss, random_state=0).fit(X, np.zeros(200))
    # single-class RDT / RF and regression stay supported
    for model in (RDT(max_depth=2), RandomForestRDT(n_estimators=3, random_state=0)):
        np.testing.assert_allclose(model.fit(X, np.full(200, 3.0)).predict_proba(X), 1.0)
    GradientBoostingRDT(task="regression", n_estimators=3, random_state=0).fit(X, np.zeros(200))


def test_native_module_guards_shapes():
    # Defence in depth for direct users of the private extension module.
    X, y = _data(n=40)
    model = _rdt_native.RDTModel("classification", "gini", 3, 2, 1, 0.0)
    with pytest.raises(ValueError):
        model.fit(X, y[:10])
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.ascontiguousarray(X[:, :2]))


def test_valid_inputs_still_accepted():
    X, y = _data(n=80, p=3)
    ref = RDT(max_depth=3).fit(X, y).predict(X)
    np.testing.assert_array_equal(RDT(max_depth=3).fit(np.asfortranarray(X), y.astype(int)).predict(X), ref)
    np.testing.assert_array_equal(RDT(max_depth=3).fit(X.tolist(), y.tolist()).predict(X.tolist()), ref)
    # huge but finite values (their sum overflows) are not mistaken for inf
    Xh = np.array([[1e308], [1e308], [1e308], [-1e308], [1.5e308], [1.7e308]])
    yh = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    for model in _models():
        assert model.fit(Xh, yh).predict(Xh).shape == (6,)


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 10, 1000, 20000])
@pytest.mark.parametrize("p", [1, 5, 50])
def test_validation_across_shapes(n, p):
    rng = np.random.default_rng(n * 100 + p)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] > 0).astype(float)
    y[:2] = [0.0, 1.0]
    w = rng.uniform(0, 2, size=n)
    w[: n // 4] = 0.0
    w[1] = 1.0
    for model in _models():
        model.fit(X, y, sample_weight=w)
        assert np.all(np.isfinite(model.predict_proba(X)))
        X_nan = X.copy()
        X_nan[rng.integers(n), rng.integers(p)] = np.nan
        with pytest.raises(ValueError):
            model.predict(X_nan)
        with pytest.raises(ValueError):
            model.predict(np.zeros((3, p + 1)))
