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
- ``min_samples_leaf`` / ``min_child_weight``: aliases (RandomForestRDT, and
  GradientBoostingRDT with the default ``split_criterion='friedman_mse'``).
  Those trees are grown on the gradient alone (hessians only enter the
  per-leaf Newton step), so ``min_child_weight`` (normally hessian-based in
  XGBoost) is treated as a plain sample-count constraint.
- ``min_impurity_decrease`` / ``gamma``: real, aliases of the same pruning
  threshold.
- ``reg_alpha`` / ``reg_lambda`` (GradientBoostingRDT only): with
  ``split_criterion='friedman_mse'`` a real effect (post-hoc leaf-value
  soft-thresholding / ridge-style shrinkage) but not XGBoost's exact
  regularised leaf formula -- that engine is Friedman's gradient boosting as
  in scikit-learn (squared-error trees on the gradient, then one Newton step
  per leaf for log_loss/exponential).
- ``GradientBoostingRDT(split_criterion='newton')`` grows the trees XGBoost's
  way instead: per-row gradients and hessians, split gain
  ``T(G_L)^2/(H_L+lambda) + T(G_R)^2/(H_R+lambda) - T(G)^2/(H+lambda)`` with
  ``T(G) = sign(G) max(|G| - alpha, 0)``, leaves ``-T(G)/(H+lambda)``, and
  ``reg_lambda``, ``reg_alpha``, ``gamma`` (alias ``min_impurity_decrease``),
  ``min_child_weight`` (a hessian sum, no longer an alias of
  ``min_samples_leaf``) and ``max_delta_step`` with XGBoost's meanings and
  defaults (1, 0, 0, 1, 0).
- ``n_iter_no_change`` / ``early_stopping_rounds``: aliases.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from . import RDT, _rdt_native
from ._validation import check_X, check_X_y, check_sample_weight, check_tree_method

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


def _check_colsample_bylevel(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not (0.0 < float(value) <= 1.0):
        raise ValueError(f"colsample_bylevel must be a number in (0, 1], got {value!r}")
    return float(value)


def _resolve_random_state(random_state) -> int:
    if random_state is None:
        return int(np.random.default_rng().integers(0, 2**63 - 1))
    return int(random_state)


def _resolve_n_jobs(n_jobs) -> int:
    return 1 if n_jobs is None else int(n_jobs)


def _resolve_max_samples(max_samples):
    """scikit-learn semantics -> (fraction, absolute count) for the native core.

    None draws n_samples rows, an int draws exactly that many rows, a float
    draws ``floor(max_samples * n_samples)`` rows (at least 1). A count of 0
    tells the core to use the fraction.
    """
    if max_samples is None:
        return 1.0, 0
    if isinstance(max_samples, (bool, np.bool_)):
        raise ValueError(f"Invalid max_samples: {max_samples!r}")
    if isinstance(max_samples, (int, np.integer)):
        if not 1 <= max_samples <= np.iinfo(np.uint32).max:
            raise ValueError(f"max_samples as an int must be >= 1, got {max_samples!r}")
        return 1.0, int(max_samples)
    if isinstance(max_samples, (float, np.floating)):
        if not (np.isfinite(max_samples) and max_samples > 0.0):
            raise ValueError(f"max_samples as a float must be finite and > 0, got {max_samples!r}")
        return float(max_samples), 0
    raise ValueError(f"Invalid max_samples: {max_samples!r}")


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
        A tree splits a depth only if the depth's total weighted impurity
        decrease -- the sum over its nodes of ``N_t / N * (node improvement)``,
        scikit-learn's scale (see ``rdt.RDT``) -- reaches this value.
    gamma : float, optional
        XGBoost-style alias of ``min_impurity_decrease`` (same scale, not
        XGBoost's gain scale).
    max_features : {'sqrt', 'log2', None} or int or float, default='sqrt'
        Chosen once per tree, not per node (see module docstring).
    colsample_bytree : {'sqrt', 'log2', None} or int or float, optional
        XGBoost/LightGBM-style alias of ``max_features``.
    bootstrap : bool, default=True
    max_samples : int, float or None, default=1.0
        Number of samples to draw (with replacement) per tree when
        ``bootstrap=True``, as in scikit-learn: an int draws exactly
        ``max_samples`` rows, a float draws
        ``max(1, floor(max_samples * n_samples))`` rows, None draws
        ``n_samples`` rows. (Up to 0.2.0 an int was treated as a fraction,
        so e.g. ``max_samples=50`` drew ``50 * n_samples`` rows per tree and
        ``max_samples=1`` drew all rows; it now draws one.)
    oob_score : bool, default=False
    class_weight : 'balanced' or dict, optional
        Converted to per-sample weights internally.
    random_state : int, optional
    n_jobs : int, optional
        Number of threads for parallel tree building. None = 1 (sequential),
        -1 = all available hardware threads.
    presort : bool, default=True
        Split search implementation. True (default) sorts each feature once
        per fit and scores every node of a depth in one pass. False sorts
        every (node, feature) pair's samples at every depth. Both perform the
        same floating-point operations (tied feature values are visited in
        row order by both), so they produce identical models; False only
        trades speed for the presort's memory (4 bytes per row and feature).
    colsample_bylevel : float, default=1.0
        Fraction in (0, 1] of the tree's columns (after ``max_features`` /
        ``colsample_bytree``) scored at each depth: every searched depth draws
        a fresh subset of ``max(1, floor(colsample_bylevel * m))`` of the
        tree's ``m`` columns, without replacement, and the depth's feature is
        chosen among them (the restricted-tree analogue of XGBoost's
        ``colsample_bylevel``). The draws come from a generator seeded from the
        tree's own seed, so they do not depend on ``n_jobs`` and leave the
        other random draws unchanged; 1.0 draws nothing.
    tree_method : {'exact', 'hist'}, default='exact'
        'hist': every feature is binned once per fit (at most ``max_bin``
        bins, from the rows with positive weight) and the trees are grown
        from per-node histograms on each tree's bootstrap rows / column
        subset of the bins (``n_jobs`` builds the trees in parallel); ``fit``
        then accepts float32 or float64 ``X`` in any memory layout (C /
        Fortran order, strided views, ``np.memmap``) without copying it, and
        reads it only to bin it (the OOB score uses the bins too). See
        ``rdt.RDT``. Not implemented for ``criterion='mae'``.
    max_bin : int, default=256
        Maximum bins per feature with ``tree_method='hist'`` (2 to 65536).

    Attributes (after fitting)
    ---------------------------
    estimators_ : list of RDT
        Each sub-tree, individually introspectable (.root, .depth_features_,
        compatible with rdt.visualize) and independently usable for
        prediction on data pre-restricted to that tree's own feature subset.
    feature_importances_ : ndarray of shape (n_features,)
    depth_features_ : list of dict
        One dict per tree, ``{depth: feature}``, the feature chosen at each
        depth as a column index of the ``X`` passed to ``fit`` (each tree's
        own column indices, as in ``estimators_[i].depth_features_``, mapped
        through its column subset).
    level_features_ : list of dict
        One dict per tree (in the order of ``depth_features_``),
        ``{depth: [columns]}``: the candidate columns of each depth the tree
        searched, as columns of ``X`` (all of the tree's columns when
        ``colsample_bylevel=1``). Replayed from the trees' seeds on each
        access (``fit`` does not store them: they hold ``n_trees x depths x
        columns`` entries); each ``depth_features_[t][d]`` is one of
        ``level_features_[t][d]``.
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
        max_samples: Union[int, float, None] = 1.0,
        oob_score: bool = False,
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        presort: bool = True,
        tree_method: str = "exact",
        max_bin: int = 256,
        colsample_bylevel: float = 1.0,
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
        self.presort = presort
        self.tree_method, self.max_bin = check_tree_method(tree_method, max_bin, self.criterion)
        self.colsample_bylevel = _check_colsample_bylevel(colsample_bylevel)

        self.estimators_: List[RDT] = []
        self.feature_importances_ = None
        self.depth_features_ = None
        self.oob_score_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None

        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RandomForestRDT":
        hist = self.tree_method == "hist"
        X, y, sample_weight = check_X_y(X, y, sample_weight, any_float_layout=hist)
        self.n_features_ = X.shape[1]

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            sample_weight = _class_weight_to_sample_weight(y, self.classes_, self.class_weight, sample_weight)
        if sample_weight is not None:
            sample_weight = check_sample_weight(sample_weight, len(y), "sample_weight combined with class_weight")

        max_samples_fraction, max_samples_count = _resolve_max_samples(self.max_samples)
        # The native fit runs with the GIL released; only publish the model
        # once it is fully fitted.
        model = _rdt_native.RandomForestModel(
            self.task, self.criterion, self.max_depth, self.min_samples_split, self.min_samples_leaf,
            self.min_impurity_decrease, self.n_estimators, _resolve_max_features(self.max_features),
            self.bootstrap, max_samples_fraction, self.oob_score,
            _resolve_random_state(self.random_state), _resolve_n_jobs(self.n_jobs),
            max_samples_count, presort=self.presort, tree_method=self.tree_method, max_bin=self.max_bin,
            colsample_bylevel=self.colsample_bylevel,
        )
        if hist:
            model.fit_hist(X, y, sample_weight)
        else:
            model.fit(X, y, sample_weight)
        self._model = model

        self.feature_importances_ = np.array(self._model.feature_importances_)
        self.depth_features_ = [dict(d) for d in self._model.depth_features_]
        if self.oob_score:
            self.oob_score_ = self._model.oob_score_

        feature_subsets = self._model.feature_subsets_
        self.estimators_ = [
            RDT._wrap_native(
                tree, task=self.task, criterion=self.criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease, n_features=len(feature_subsets[i]),
                tree_method=self.tree_method, max_bin=self.max_bin,
            )
            for i, tree in enumerate(self._model.estimators_)
        ]
        return self

    @property
    def level_features_(self) -> Optional[List[Dict[int, List[int]]]]:
        """Per tree, ``{depth: [candidate columns of X]}`` (see the class
        docstring), replayed from the trees' seeds on each access; None
        before ``fit``."""
        if self._model is None:
            return None
        return [dict(d) for d in self._model.level_features_]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Forest not fitted yet. Call fit() first.")
        X = check_X(X, self.n_features_)
        return np.array(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        if self._model is None:
            raise ValueError("Forest not fitted yet. Call fit() first.")
        X = check_X(X, self.n_features_)
        return np.array(self._model.predict_proba(X))


class GradientBoostingRDT:
    """Gradient Boosting of Restricted Decision Trees, backed by a native
    C++ core. Friedman's gradient boosting, as in scikit-learn's
    GradientBoostingClassifier/Regressor: each round fits an RDT to the
    negative gradient, then replaces each leaf value by the loss's one-step
    Newton estimate sum(w*g)/sum(w*h) (log_loss, exponential; squared_error's
    leaf mean is already exact; absolute_error/huber/quantile keep the leaf
    mean of the gradient). The initial score is the weighted constant
    minimiser of the loss (prior log-odds, log class priors, mean, median,
    quantile). ``sample_weight``/``scale_pos_weight`` weight the trees, the
    init, the leaf values, ``train_score_`` and the early-stopping
    validation loss. See the module docstring for how reg_alpha/reg_lambda
    differ from XGBoost's exact mechanics.

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
    min_child_weight : float, optional
        ``split_criterion='friedman_mse'``: XGBoost-style alias of
        ``min_samples_leaf``. ``split_criterion='newton'``: XGBoost's meaning,
        the minimum hessian sum in each child of a split (default 1.0);
        ``min_samples_leaf`` then stays a separate sample-count constraint.
    min_impurity_decrease : float, default=0.0
        ``split_criterion='friedman_mse'``: each round's tree splits a depth
        only if the depth's total weighted decrease in squared error of the
        gradient -- the sum over its nodes of ``N_t / N * (node improvement)``,
        scikit-learn's scale (see ``rdt.RDT``) -- reaches this value.
        ``split_criterion='newton'``: XGBoost's ``gamma`` (min_split_loss) on
        XGBoost's gain scale -- each node splits only if its gain reaches it.
    gamma : float, optional
        XGBoost-style alias of ``min_impurity_decrease`` (with
        ``split_criterion='friedman_mse'`` on the same scale, not XGBoost's
        gain scale).
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
    reg_lambda : float, optional
        Default 0.0 with ``split_criterion='friedman_mse'`` (post-hoc shrink,
        see the module docstring), 1.0 with ``'newton'`` (XGBoost's lambda).
    split_criterion : {'friedman_mse', 'newton'}, default='friedman_mse'
        'friedman_mse': trees grown on the negative gradient with a
        squared-error criterion (scikit-learn's gradient boosting).
        'newton': XGBoost's second-order tree growth -- per-row gradients g
        and hessians h (multiplied by ``sample_weight`` /
        ``scale_pos_weight``; log_loss: h = max(p(1-p), 1e-16), multiclass
        h = max(2p(1-p) w, 1e-16) per class as XGBoost's softmax;
        exponential: h = exp(-y F); squared_error: h = 1), each node's best
        split maximising the gain
        ``score(G_L, H_L) + score(G_R, H_R) - score(G, H)`` with
        ``score(G, H) = T(G)^2 / (H + reg_lambda)``,
        ``T(G) = sign(G) max(|G| - reg_alpha, 0)`` (XGBoost's scale, no 1/2;
        evaluated at the clipped weight when ``max_delta_step > 0``). A node
        splits only if its gain is >= ``gamma`` and > 1e-6 and each child's
        hessian sum is >= ``min_child_weight``; other nodes become leaves.
        Each depth's feature maximises the sum of (gain - gamma) over the
        nodes it splits. Leaves are ``-T(G) / (H + reg_lambda)`` (clipped to
        +-``max_delta_step`` if > 0) times ``learning_rate``;
        ``feature_importances_`` are normalised total gains. Needs a loss
        with a hessian: 'log_loss', 'exponential' or 'squared_error'.
    max_delta_step : float, default=0.0
        ``split_criterion='newton'`` only: if > 0, leaf weights (before
        ``learning_rate``) are clipped to +-max_delta_step, as in XGBoost.
    colsample_bylevel : float, default=1.0
        Fraction in (0, 1] of the tree's columns (after ``max_features`` /
        ``colsample_bytree``) scored at each depth: every searched depth draws
        a fresh subset of ``max(1, floor(colsample_bylevel * m))`` of the
        tree's ``m`` columns, without replacement, and the depth's feature is
        chosen among them (the restricted-tree analogue of XGBoost's
        ``colsample_bylevel``). The draws come from a generator seeded from the
        tree's own seed, so they do not depend on ``n_jobs`` and leave the
        other random draws unchanged; 1.0 draws nothing.
    tree_method : {'exact', 'hist'}, default='exact'
        'hist': every feature is binned once per fit (at most ``max_bin``
        bins, from the training rows -- after the early-stopping split --
        with positive weight) and the trees are grown from per-node
        histograms; ``fit`` then accepts float32 or float64 ``X`` in any
        memory layout (C / Fortran order, strided views, ``np.memmap``)
        without copying it and reads it only to bin it: the training scores
        (validation rows included) are updated through the bins. A round
        that fits one tree (binary, regression) scores its features in
        parallel with ``n_jobs`` threads; a multiclass round fits its
        per-class trees in parallel instead. See ``rdt.RDT``.
    max_bin : int, default=256
        Maximum bins per feature with ``tree_method='hist'`` (2 to 65536).
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
    presort : bool, default=True
        Split search implementation. True (default) sorts each feature once
        per fit and scores every node of a depth in one pass. False sorts
        every (node, feature) pair's samples at every depth. Both perform the
        same floating-point operations (tied feature values are visited in
        row order by both), so they produce identical models; False only
        trades speed for the presort's memory (4 bytes per row and feature).

    Attributes (after fitting)
    ---------------------------
    feature_importances_ : ndarray of shape (n_features,)
    train_score_ : ndarray
        Training loss (sample-weighted if weights were given) at each round
        actually fitted.
    n_estimators_ : int
        Rounds actually fitted -- less than ``n_estimators`` if early
        stopping triggered.
    depth_features_ : list of dict
        One dict per tree, ``{depth: feature}``, in the order the trees were
        fitted (round by round; a multiclass round fits one tree per class, in
        ``classes_`` order). Features are column indices of the ``X`` passed
        to ``fit`` (mapped through each tree's column subset).

    level_features_ : list of dict
        One dict per tree (in the order of ``depth_features_``),
        ``{depth: [columns]}``: the candidate columns of each depth the tree
        searched, as columns of ``X`` (all of the tree's columns when
        ``colsample_bylevel=1``). Replayed from the trees' seeds on each
        access (``fit`` does not store them: they hold ``n_trees x depths x
        columns`` entries); each ``depth_features_[t][d]`` is one of
        ``level_features_[t][d]``.

    Staged predictions (``staged_decision_function``,
    ``staged_predict_proba``, ``staged_predict``) yield one array per fitted
    round (``n_estimators_`` of them), each a new array; the last equals
    ``decision_function`` / ``predict_proba`` / ``predict`` exactly.
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
        min_child_weight: Optional[float] = None,
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
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        huber_delta: float = 1.0,
        quantile_alpha: float = 0.5,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        presort: bool = True,
        split_criterion: str = "friedman_mse",
        max_delta_step: float = 0.0,
        tree_method: str = "exact",
        max_bin: int = 256,
        colsample_bylevel: float = 1.0,
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

        self.split_criterion = str(split_criterion).lower()
        if self.split_criterion not in ("friedman_mse", "newton"):
            raise ValueError("split_criterion must be 'friedman_mse' or 'newton'")
        newton = self.split_criterion == "newton"
        if newton and self.loss not in ("log_loss", "exponential", "squared_error"):
            raise ValueError(
                f"split_criterion='newton' needs a loss with a hessian ('log_loss', 'exponential', "
                f"'squared_error'); loss={self.loss!r} has none")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        if newton:
            # XGBoost's min_child_weight: a hessian sum, separate from the
            # min_samples_leaf count.
            self.min_samples_leaf = min_samples_leaf
            self.min_child_weight = 1.0 if min_child_weight is None else min_child_weight
        else:
            self.min_samples_leaf = _resolve_alias(
                min_samples_leaf, 1, min_child_weight, "min_samples_leaf", "min_child_weight")
            self.min_child_weight = min_child_weight
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
        self.reg_lambda = (1.0 if newton else 0.0) if reg_lambda is None else reg_lambda
        self.max_delta_step = max_delta_step
        if not (np.isfinite(self.max_delta_step) and self.max_delta_step >= 0):
            raise ValueError(f"max_delta_step must be a finite number >= 0, got {max_delta_step!r}")
        if newton:
            for name in ("reg_lambda", "reg_alpha", "min_impurity_decrease", "min_child_weight"):
                value = getattr(self, name)
                if not (np.isfinite(value) and value >= 0):
                    shown = "gamma/min_impurity_decrease" if name == "min_impurity_decrease" else name
                    raise ValueError(f"{shown} must be a finite number >= 0, got {value!r}")
        elif self.max_delta_step != 0:
            raise ValueError("max_delta_step only applies to split_criterion='newton'")
        self.scale_pos_weight = scale_pos_weight
        self.huber_delta = huber_delta
        self.quantile_alpha = quantile_alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.presort = presort
        self.tree_method, self.max_bin = check_tree_method(tree_method, max_bin)
        self.colsample_bylevel = _check_colsample_bylevel(colsample_bylevel)

        self.feature_importances_ = None
        self.train_score_ = None
        self.n_estimators_ = None
        self.depth_features_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None

        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "GradientBoostingRDT":
        hist = self.tree_method == "hist"
        X, y, sample_weight = check_X_y(X, y, sample_weight, any_float_layout=hist)
        self.n_features_ = X.shape[1]

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ < 2:
                raise ValueError(
                    f"GradientBoostingRDT classification needs at least 2 classes in y; "
                    f"got {self.n_classes_} ({self.classes_.tolist()})."
                )
            if self.loss == "exponential" and self.n_classes_ != 2:
                raise ValueError("loss='exponential' only supports binary classification")
            if self.scale_pos_weight is not None:
                if not (np.isfinite(self.scale_pos_weight) and self.scale_pos_weight > 0):
                    raise ValueError(f"scale_pos_weight must be a finite number > 0, got {self.scale_pos_weight!r}")
                if self.n_classes_ != 2:
                    raise ValueError("scale_pos_weight only supports binary classification")
                base = np.ones(len(y), dtype=np.float64) if sample_weight is None else np.array(sample_weight, dtype=np.float64)
                base[y == self.classes_[1]] *= self.scale_pos_weight
                sample_weight = base
        if sample_weight is not None:
            sample_weight = check_sample_weight(sample_weight, len(y), "sample_weight combined with scale_pos_weight")

        # The native fit runs with the GIL released; only publish the model
        # once it is fully fitted.
        model = _rdt_native.GradientBoostingModel(
            self.task, self.loss, self.max_depth, self.min_samples_split, self.min_samples_leaf,
            self.min_impurity_decrease, self.n_estimators, self.learning_rate, self.subsample,
            _resolve_max_features(self.max_features), self.validation_fraction,
            self.n_iter_no_change if self.n_iter_no_change is not None else 0, self.tol,
            self.reg_alpha, self.reg_lambda, self.huber_delta, self.quantile_alpha,
            _resolve_random_state(self.random_state), _resolve_n_jobs(self.n_jobs), presort=self.presort,
            split_criterion=self.split_criterion,
            min_child_weight=float(self.min_child_weight) if self.split_criterion == "newton" else 1.0,
            max_delta_step=float(self.max_delta_step), tree_method=self.tree_method, max_bin=self.max_bin,
            colsample_bylevel=self.colsample_bylevel,
        )
        if hist:
            model.fit_hist(X, y, sample_weight)
        else:
            model.fit(X, y, sample_weight)
        self._model = model

        self.feature_importances_ = np.array(self._model.feature_importances_)
        self.train_score_ = np.array(self._model.train_score_)
        self.n_estimators_ = self._model.n_estimators_fitted_
        self.depth_features_ = [dict(d) for d in self._model.depth_features_]
        return self

    @property
    def level_features_(self) -> Optional[List[Dict[int, List[int]]]]:
        """Per tree, ``{depth: [candidate columns of X]}`` (see the class
        docstring), replayed from the trees' seeds on each access; None
        before ``fit``."""
        if self._model is None:
            return None
        return [dict(d) for d in self._model.level_features_]

    def _check_predict_X(self, X, classification_only: bool = False, what: str = "") -> np.ndarray:
        if classification_only and self.task != "classification":
            raise ValueError(f"{what} only available for classification")
        if self._model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return check_X(X, self.n_features_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_predict_X(X)
        return np.array(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._check_predict_X(X, True, "predict_proba")
        return np.array(self._model.predict_proba(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw scores (the initial score plus every round's trees, before the
        link function), as scikit-learn's ``decision_function``: shape
        ``(n_samples,)`` for binary classification (the positive class's
        log-odds; half of it for ``loss='exponential'``),
        ``(n_samples, n_classes)`` for multiclass. Classification only."""
        X = self._check_predict_X(X, True, "decision_function")
        return np.asarray(self._model.decision_function(X))

    def staged_decision_function(self, X: np.ndarray):
        """Yield ``decision_function(X)`` after each fitted round (one array
        per round, ``n_estimators_`` in total, each a new array; the last
        equals ``decision_function(X)``). Classification only. ``X`` is
        validated before the first stage is computed."""
        X = self._check_predict_X(X, True, "staged_decision_function")
        return self._staged(X, "decision")

    def staged_predict_proba(self, X: np.ndarray):
        """Yield ``predict_proba(X)`` after each fitted round (see
        ``staged_decision_function``)."""
        X = self._check_predict_X(X, True, "staged_predict_proba")
        return self._staged(X, "proba")

    def staged_predict(self, X: np.ndarray):
        """Yield ``predict(X)`` after each fitted round (see
        ``staged_decision_function``); also available for regression."""
        X = self._check_predict_X(X)
        return self._staged(X, "predict")

    def _staged(self, X: np.ndarray, kind: str):
        # The native iterator adds one round per step (GIL released) and
        # returns a fresh array each time; it holds the native model it was
        # created from, so a later fit() of this estimator does not affect it.
        yield from self._model._staged(X, kind)
