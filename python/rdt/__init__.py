"""Native C++ implementation of Restricted Decision Trees (RDT).

Sklearn-style wrapper around the nanobind-compiled ``_rdt_native.RDTModel``,
matching the public surface of ``rdt_optimal_cython.OptimalRDTCython`` so it
can be used as a near drop-in replacement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _rdt_native
from ._validation import check_X, check_X_y, check_tree_method

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
        A depth's winning feature is only committed if its total weighted
        impurity decrease -- the sum over the depth's nodes of
        ``N_t / N * (impurity - N_t_L / N_t * left_impurity - N_t_R / N_t * right_impurity)``,
        with ``N`` the total (weighted) number of training samples -- meets
        this threshold (up to machine epsilon, as in scikit-learn); otherwise
        every node at that depth becomes a leaf. Same quantity and
        comparison as sklearn's parameter of the same name, which sklearn
        applies per node rather than per depth (XGBoost/LightGBM's closest
        equivalent is gamma/min_split_loss).
    presort : bool, default=True
        Split search implementation. True (default) sorts each feature once
        per fit and scores every node of a depth in one pass. False sorts
        every (node, feature) pair's samples at every depth. Both perform the
        same floating-point operations (tied feature values are visited in
        row order by both), so they produce identical trees; False only
        trades speed for the presort's memory (4 bytes per row and feature).
        Ignored with ``tree_method='hist'``.
    tree_method : {'exact', 'hist'}, default='exact'
        'exact' scores every boundary between distinct feature values.
        'hist' bins every feature once per fit (``max_bin`` bins at most,
        from the rows with positive weight: one bin per distinct value when
        there are at most ``max_bin`` of them, else approximately equal-count
        bins) and scores the boundaries between bins, from per-node
        histograms. A split's threshold lies between its two bins (the exact
        search's midpoint when bins hold one value each), so predicting on
        the raw ``X`` routes every row as its bins did.
        ``fit`` then accepts float32 or float64 ``X`` in any memory layout
        (C or Fortran order, strided views, ``np.memmap``) without copying it.
        Not implemented for ``criterion='mae'``.
    max_bin : int, default=256
        Maximum number of bins per feature with ``tree_method='hist'``
        (2 to 65536; bins are stored as uint8, or every feature's as uint16
        when any feature has more than 256 bins).
    n_jobs : int, optional
        Threads for the ``tree_method='hist'`` split search (features are
        scored in parallel; the tree does not depend on it). None = 1,
        -1 = all hardware threads.
    """

    def __init__(
        self,
        task: str = "classification",
        criterion: Optional[str] = None,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        presort: bool = True,
        tree_method: str = "exact",
        max_bin: int = 256,
        n_jobs: Optional[int] = None,
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
        self.presort = presort
        self.tree_method, self.max_bin = check_tree_method(tree_method, max_bin, self.criterion)
        self.n_jobs = n_jobs

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
                      min_impurity_decrease: float, n_features: int,
                      tree_method: str = "exact", max_bin: int = 256) -> "RDT":
        """Wrap an already-fitted native RDTModel (e.g. one sub-tree of an
        ensemble) for introspection -- .root, .depth_features_, .predict(),
        rdt.visualize compatibility -- without re-fitting."""
        self = cls(
            task=task, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease, tree_method=tree_method, max_bin=max_bin,
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
        hist = self.tree_method == "hist"
        X, y, sample_weight = check_X_y(X, y, sample_weight, any_float_layout=hist)

        self.n_features_ = X.shape[1]

        # The native fit runs with the GIL released; only publish the model
        # once it is fully fitted.
        model = _rdt_native.RDTModel(
            self.task, self.criterion, self.max_depth,
            self.min_samples_split, self.min_samples_leaf, self.min_impurity_decrease,
            presort=self.presort, tree_method=self.tree_method, max_bin=self.max_bin,
            n_jobs=1 if self.n_jobs is None else int(self.n_jobs),
        )
        if hist:
            model.fit_hist(X, y, sample_weight)
        else:
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
        X = check_X(X, self.n_features_)
        return np.array(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        if self._model is None:
            raise ValueError("Tree not fitted yet. Call fit() first.")
        X = check_X(X, self.n_features_)
        return np.array(self._model.predict_proba(X))
