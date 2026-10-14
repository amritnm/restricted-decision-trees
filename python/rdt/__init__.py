"""Native C++ implementation of Restricted Decision Trees (RDT).

Sklearn-style wrapper around the nanobind-compiled ``_rdt_native.RDTModel``,
matching the public surface of ``rdt_optimal_cython.OptimalRDTCython`` so it
can be used as a near drop-in replacement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _rdt_native

__all__ = ["RDT"]


class RDT:
    """Restricted Decision Tree, backed by a native C++ core.

    Parameters
    ----------
    task : str, default='classification'
        'classification' or 'regression'
    criterion : str, optional
        Splitting criterion. If None, defaults to 'gini' for classification
        or 'mse' for regression.
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    min_impurity_decrease : float, default=0.0
        A depth's winning feature is only committed if its total impurity
        improvement meets this threshold; otherwise every node at that depth
        becomes a leaf. Matches sklearn's parameter of the same name (and is
        XGBoost/LightGBM's gamma/min_split_loss).
    """

    def __init__(
        self,
        task: str = "classification",
        criterion: Optional[str] = None,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ):
        self.task = task.lower()
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")

        if criterion is None:
            self.criterion = "gini" if self.task == "classification" else "mse"
        else:
            self.criterion = criterion.lower()

        if self.task == "classification" and self.criterion not in ("gini", "entropy"):
            raise ValueError("For classification, criterion must be 'gini' or 'entropy'")
        if self.task == "regression" and self.criterion not in ("mse", "mae"):
            raise ValueError("For regression, criterion must be 'mse' or 'mae'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.root = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.depth_features_ = {}
        self.depth_improvements_ = {}

        self._model = None

    @classmethod
    def _wrap_native(cls, model, *, task: str, criterion: str, max_depth: int,
                      min_samples_split: int, min_samples_leaf: int,
                      min_impurity_decrease: float, n_features: int) -> "RDT":
        """Wrap an already-fitted native RDTModel (e.g. one sub-tree of an
        ensemble) for introspection -- .root, .depth_features_, .predict(),
        rdt.visualize compatibility -- without re-fitting."""
        self = cls(
            task=task, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self._model = model
        self.n_features_ = n_features
        if self.task == "classification":
            self.classes_ = np.array(model.classes_)
            self.n_classes_ = model.n_classes_
        self.depth_features_ = dict(model.depth_features_)
        self.depth_improvements_ = dict(model.depth_improvements_)
        self.root = model.root
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RDT":
        X = np.asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64, order="C")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64, order="C")

        self.n_features_ = X.shape[1]

        # The native fit runs with the GIL released; only publish the model
        # once it is fully fitted.
        model = _rdt_native.RDTModel(
            self.task, self.criterion, self.max_depth,
            self.min_samples_split, self.min_samples_leaf, self.min_impurity_decrease,
        )
        model.fit(X, y, sample_weight)
        self._model = model

        if self.task == "classification":
            self.classes_ = np.array(self._model.classes_)
            self.n_classes_ = self._model.n_classes_

        self.depth_features_ = dict(self._model.depth_features_)
        self.depth_improvements_ = dict(self._model.depth_improvements_)
        self.root = self._model.root

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Tree not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64, order="C")
        return np.array(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        if self._model is None:
            raise ValueError("Tree not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64, order="C")
        return np.array(self._model.predict_proba(X))
