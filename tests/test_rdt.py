"""Unit tests for the rdt.RDT Python wrapper (not parity -- see test_parity.py)."""

import numpy as np
import pytest

from rdt import RDT


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        RDT(task="not_a_task")


def test_invalid_criterion_for_classification_raises():
    with pytest.raises(ValueError):
        RDT(task="classification", criterion="mse")


def test_invalid_criterion_for_regression_raises():
    with pytest.raises(ValueError):
        RDT(task="regression", criterion="gini")


def test_default_criterion_per_task():
    assert RDT(task="classification").criterion == "gini"
    assert RDT(task="regression").criterion == "mse"


def test_predict_before_fit_raises():
    model = RDT()
    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_predict_proba_on_regression_raises():
    model = RDT(task="regression")
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_fit_predict_roundtrip_classification():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    model = RDT(task="classification", max_depth=3)
    model.fit(X, y)

    preds = model.predict(X)
    np.testing.assert_array_equal(preds, y)

    proba = model.predict_proba(X)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(6))

    assert model.n_classes_ == 2
    np.testing.assert_array_equal(model.classes_, [0.0, 1.0])
    assert model.depth_features_ == {0: 0}
    assert model.root is not None
    assert model.root.is_leaf is False


def test_fit_predict_roundtrip_regression():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([10.0, 12.0, 50.0, 52.0])

    model = RDT(task="regression", max_depth=1)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds[0] == pytest.approx(11.0)
    assert preds[2] == pytest.approx(51.0)


def test_root_node_tree_structure_is_navigable():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = RDT(task="classification", max_depth=1)
    model.fit(X, y)

    root = model.root
    assert root.is_leaf is False
    assert root.split_feature == 0
    assert root.left.is_leaf is True
    assert root.right.is_leaf is True
    assert root.left.value == 0.0
    assert root.right.value == 1.0
