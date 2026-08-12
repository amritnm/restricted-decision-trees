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
    """

    def __init__(
        self,
        task: str = "classification",
        criterion: Optional[str] = None,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
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

        self.root = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.depth_features_ = {}

        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RDT":
        X = np.asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64, order="C")

        self.n_features_ = X.shape[1]

        self._model = _rdt_native.RDTModel(
            self.task, self.criterion, self.max_depth,
            self.min_samples_split, self.min_samples_leaf,
        )
        self._model.fit(X, y)

        if self.task == "classification":
            self.classes_ = np.array(self._model.classes_)
            self.n_classes_ = self._model.n_classes_

        self.depth_features_ = dict(self._model.depth_features_)
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
