"""Random Forest and Gradient Boosting ensembles of RDTs, native C++ core.

Parameter names follow scikit-learn where a concept maps to a scikit-learn
parameter, with the XGBoost/LightGBM name accepted as an alias where they
differ (setting both to conflicting non-default values raises). See the
project's ensembling plan for the full parameter mapping table; the short
version:

- ``max_features`` / ``colsample_bytree``: RDT's structural constraint
  (every node at a depth splits on the same feature) means this can't work
  scikit-learn's usual way -- a fresh random subset *per node*. Instead each
  tree gets ONE random column subset chosen before its own fit(), matching
  this project's earlier Python ``ensemble/random_forest_rdt.py``.
- ``min_samples_leaf`` / ``min_child_weight``: aliases. There's no hessian
  in this first-order boosting engine, so ``min_child_weight`` (normally
  hessian-based in XGBoost) is treated as a plain sample-count constraint.
- ``min_impurity_decrease`` / ``gamma``: real, aliases of the same pruning
  threshold.
- ``reg_alpha`` / ``reg_lambda`` (GradientBoostingRDT only): a real effect
  (post-hoc leaf-value soft-thresholding / ridge-style shrinkage) but not
  XGBoost's exact Newton/hessian-derived leaf formula -- this engine is
  first-order (Friedman-style), not second-order.
- ``n_iter_no_change`` / ``early_stopping_rounds``: aliases.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from . import RDT, _rdt_native

__all__ = ["RandomForestRDT", "GradientBoostingRDT"]


def _resolve_alias(primary_value, primary_default, alias_value, primary_name, alias_name):
    if alias_value is not None:
        if primary_value != primary_default and primary_value != alias_value:
            raise ValueError(
                f"Both '{primary_name}'={primary_value!r} and '{alias_name}'={alias_value!r} "
                "were set to conflicting values; they are aliases for the same parameter "
                "-- set only one."
            )
        return alias_value
    return primary_value


def _resolve_max_features(value):
    if value is None:
        return _rdt_native._max_features_all()
    if isinstance(value, str):
        v = value.lower()
        if v == "sqrt":
            return _rdt_native._max_features_sqrt()
        if v == "log2":
            return _rdt_native._max_features_log2()
        raise ValueError(f"Unknown max_features/colsample_bytree string: {value!r}")
    if isinstance(value, bool):
        raise ValueError(f"Invalid max_features/colsample_bytree: {value!r}")
    if isinstance(value, (int, np.integer)):
        return _rdt_native._max_features_int(int(value))
    if isinstance(value, (float, np.floating)):
        return _rdt_native._max_features_float(float(value))
    raise ValueError(f"Invalid max_features/colsample_bytree: {value!r}")


def _resolve_random_state(random_state) -> int:
    if random_state is None:
        return int(np.random.default_rng().integers(0, 2**63 - 1))
    return int(random_state)


def _resolve_n_jobs(n_jobs) -> int:
    return 1 if n_jobs is None else int(n_jobs)


def _class_weight_to_sample_weight(y: np.ndarray, classes: np.ndarray, class_weight,
                                    base_sample_weight: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if class_weight is None:
        return base_sample_weight
    weights = np.ones(len(y), dtype=np.float64) if base_sample_weight is None else np.array(base_sample_weight, dtype=np.float64)
    if class_weight == "balanced" or class_weight == "balanced_subsample":
        n = len(y)
        n_classes = len(classes)
        class_w = {c: n / (n_classes * np.sum(y == c)) for c in classes}
    elif isinstance(class_weight, dict):
        class_w = class_weight
    else:
        raise ValueError(f"Invalid class_weight: {class_weight!r}")
    for c, w in class_w.items():
        weights[y == c] *= w
    return weights


class RandomForestRDT:
    """Random Forest of Restricted Decision Trees, backed by a native C++
    core with real cross-tree parallelism.

    Parameters
    ----------
    n_estimators : int, default=100
    task : {'classification', 'regression'}, default='classification'
    criterion : str, optional
        Defaults to 'gini' (classification) or 'mse' (regression).
    max_depth : int, default=5
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    min_child_weight : int, optional
        XGBoost-style alias of ``min_samples_leaf`` (see module docstring).
    min_impurity_decrease : float, default=0.0
    gamma : float, optional
        XGBoost-style alias of ``min_impurity_decrease``.
    max_features : {'sqrt', 'log2', None} or int or float, default='sqrt'
        Chosen once per tree, not per node (see module docstring).
    colsample_bytree : {'sqrt', 'log2', None} or int or float, optional
        XGBoost/LightGBM-style alias of ``max_features``.
    bootstrap : bool, default=True
    max_samples : float, default=1.0
        Fraction of samples to draw (with replacement) per tree when
        ``bootstrap=True``.
    oob_score : bool, default=False
    class_weight : 'balanced' or dict, optional
        Converted to per-sample weights internally.
    random_state : int, optional
    n_jobs : int, optional
        Number of threads for parallel tree building. None = 1 (sequential),
        -1 = all available hardware threads.

    Attributes (after fitting)
    ---------------------------
    estimators_ : list of RDT
        Each sub-tree, individually introspectable (.root, .depth_features_,
        compatible with rdt.visualize) and independently usable for
        prediction on data pre-restricted to that tree's own feature subset.
    feature_importances_ : ndarray of shape (n_features,)
    oob_score_ : float
        Only set if oob_score=True.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        task: str = "classification",
        criterion: Optional[str] = None,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_child_weight: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        gamma: Optional[float] = None,
        max_features: Union[str, int, float, None] = "sqrt",
        colsample_bytree: Union[str, int, float, None] = None,
        bootstrap: bool = True,
        max_samples: float = 1.0,
        oob_score: bool = False,
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: Optional[int] = None,
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

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = _resolve_alias(
            min_samples_leaf, 1, min_child_weight, "min_samples_leaf", "min_child_weight")
        self.min_impurity_decrease = _resolve_alias(
            min_impurity_decrease, 0.0, gamma, "min_impurity_decrease", "gamma")
        self.max_features = _resolve_alias(
            max_features, "sqrt", colsample_bytree, "max_features", "colsample_bytree")
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimators_: List[RDT] = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None

        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RandomForestRDT":
        X = np.asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64, order="C")
        self.n_features_ = X.shape[1]

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            sample_weight = _class_weight_to_sample_weight(y, self.classes_, self.class_weight, sample_weight)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64, order="C")

        self._model = _rdt_native.RandomForestModel(
            self.task, self.criterion, self.max_depth, self.min_samples_split, self.min_samples_leaf,
            self.min_impurity_decrease, self.n_estimators, _resolve_max_features(self.max_features),
            self.bootstrap, self.max_samples, self.oob_score,
            _resolve_random_state(self.random_state), _resolve_n_jobs(self.n_jobs),
        )
        self._model.fit(X, y, sample_weight)

        self.feature_importances_ = np.array(self._model.feature_importances_)
        if self.oob_score:
            self.oob_score_ = self._model.oob_score_

        feature_subsets = self._model.feature_subsets_
        self.estimators_ = [
            RDT._wrap_native(
                tree, task=self.task, criterion=self.criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease, n_features=len(feature_subsets[i]),
            )
            for i, tree in enumerate(self._model.estimators_)
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Forest not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64, order="C")
        return np.array(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        if self._model is None:
            raise ValueError("Forest not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64, order="C")
        return np.array(self._model.predict_proba(X))


class GradientBoostingRDT:
    """Gradient Boosting of Restricted Decision Trees, backed by a native
    C++ core. First-order (Friedman-style) boosting -- see module docstring
    for how reg_alpha/reg_lambda differ from XGBoost's exact mechanics.

    Parameters
    ----------
    n_estimators : int, default=100
    task : {'classification', 'regression'}, default='classification'
    loss : str, optional
        Classification: 'log_loss' (binary + multiclass, default),
        'exponential' (binary only). Regression: 'squared_error' (default),
        'absolute_error', 'huber', 'quantile'.
    learning_rate : float, default=0.1
    max_depth : int, default=3
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    min_child_weight : int, optional
        XGBoost-style alias of ``min_samples_leaf``.
    min_impurity_decrease : float, default=0.0
    gamma : float, optional
        XGBoost-style alias of ``min_impurity_decrease``.
    subsample : float, default=1.0
        Row fraction per round (without replacement) -- stochastic GB.
    max_features : {'sqrt', 'log2', None} or int or float, default=None (all)
        Chosen once per tree, not per node (see module docstring).
    colsample_bytree : {'sqrt', 'log2', None} or int or float, optional
        XGBoost/LightGBM-style alias of ``max_features``.
    validation_fraction : float, default=0.1
    n_iter_no_change : int, optional
        Enables early stopping when set (> 0).
    early_stopping_rounds : int, optional
        XGBoost-style alias of ``n_iter_no_change``.
    tol : float, default=1e-4
    reg_alpha : float, default=0.0
    reg_lambda : float, default=0.0
    scale_pos_weight : float, optional
        XGBoost-style: multiplies the positive class's (``classes_[1]``)
        sample weight. Binary classification only.
    huber_delta : float, default=1.0
        Only used when loss='huber'. Unlike sklearn/XGBoost, this is a fixed
        value, not adaptively recomputed from a residual quantile each round.
    quantile_alpha : float, default=0.5
        Only used when loss='quantile'; the target quantile.
    random_state : int, optional
    n_jobs : int, optional
        Threads for a multiclass round's per-class trees (independent within
        a round); boosting rounds themselves are inherently sequential.
        None = 1, -1 = all available hardware threads.

    Attributes (after fitting)
    ---------------------------
    feature_importances_ : ndarray of shape (n_features,)
    train_score_ : ndarray
        Training loss at each round actually fitted.
    n_estimators_ : int
        Rounds actually fitted -- less than ``n_estimators`` if early
        stopping triggered.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        task: str = "classification",
        loss: Optional[str] = None,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_child_weight: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        gamma: Optional[float] = None,
        subsample: float = 1.0,
        max_features: Union[str, int, float, None] = None,
        colsample_bytree: Union[str, int, float, None] = None,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        tol: float = 1e-4,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        scale_pos_weight: Optional[float] = None,
        huber_delta: float = 1.0,
        quantile_alpha: float = 0.5,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.task = task.lower()
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")

        if loss is None:
            self.loss = "log_loss" if self.task == "classification" else "squared_error"
        else:
            self.loss = loss.lower()
        classification_losses = ("log_loss", "exponential")
        regression_losses = ("squared_error", "absolute_error", "huber", "quantile")
        if self.task == "classification" and self.loss not in classification_losses:
            raise ValueError(f"For classification, loss must be one of {classification_losses}")
        if self.task == "regression" and self.loss not in regression_losses:
            raise ValueError(f"For regression, loss must be one of {regression_losses}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = _resolve_alias(
            min_samples_leaf, 1, min_child_weight, "min_samples_leaf", "min_child_weight")
        self.min_impurity_decrease = _resolve_alias(
            min_impurity_decrease, 0.0, gamma, "min_impurity_decrease", "gamma")
        self.subsample = subsample
        self.max_features = _resolve_alias(
            max_features, None, colsample_bytree, "max_features", "colsample_bytree")
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = _resolve_alias(
            n_iter_no_change, None, early_stopping_rounds, "n_iter_no_change", "early_stopping_rounds")
        self.tol = tol
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.huber_delta = huber_delta
        self.quantile_alpha = quantile_alpha
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.feature_importances_ = None
        self.train_score_ = None
        self.n_estimators_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None

        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "GradientBoostingRDT":
        X = np.asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64, order="C")
        self.n_features_ = X.shape[1]

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            if self.loss == "exponential" and self.n_classes_ != 2:
                raise ValueError("loss='exponential' only supports binary classification")
            if self.scale_pos_weight is not None:
                if self.n_classes_ != 2:
                    raise ValueError("scale_pos_weight only supports binary classification")
                base = np.ones(len(y), dtype=np.float64) if sample_weight is None else np.array(sample_weight, dtype=np.float64)
                base[y == self.classes_[1]] *= self.scale_pos_weight
                sample_weight = base
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64, order="C")

        self._model = _rdt_native.GradientBoostingModel(
            self.task, self.loss, self.max_depth, self.min_samples_split, self.min_samples_leaf,
            self.min_impurity_decrease, self.n_estimators, self.learning_rate, self.subsample,
            _resolve_max_features(self.max_features), self.validation_fraction,
            self.n_iter_no_change if self.n_iter_no_change is not None else 0, self.tol,
            self.reg_alpha, self.reg_lambda, self.huber_delta, self.quantile_alpha,
            _resolve_random_state(self.random_state), _resolve_n_jobs(self.n_jobs),
        )
        self._model.fit(X, y, sample_weight)

        self.feature_importances_ = np.array(self._model.feature_importances_)
        self.train_score_ = np.array(self._model.train_score_)
        self.n_estimators_ = self._model.n_estimators_fitted_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64, order="C")
        return np.array(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        if self._model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64, order="C")
        return np.array(self._model.predict_proba(X))
