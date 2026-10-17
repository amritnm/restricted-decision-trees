"""Input validation shared by RDT, RandomForestRDT and GradientBoostingRDT.

The native core trusts its inputs completely: it indexes ``y`` and
``sample_weight`` by row number, indexes ``X`` by the column numbers it saw at
fit time, and sorts feature values with ``<`` (not a strict weak ordering once
NaN is present). So every public ``fit``/``predict``/``predict_proba`` call
goes through these checks first, mirroring scikit-learn's ``check_X_y`` /
``check_array`` / ``_check_sample_weight`` behaviour (``ValueError`` on bad
shapes and on NaN/infinity).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _assert_all_finite(a: np.ndarray, name: str) -> None:
    # Fast path: the sum of an all-finite array is finite (unless it
    # overflows), so the O(n) elementwise mask is only built when needed.
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.sum(a)
    if not np.isfinite(total) and not np.isfinite(a).all():
        raise ValueError(
            f"Input {name} contains NaN or infinity. Missing values are not "
            "supported; impute or drop them before calling fit/predict."
        )


def check_X(X, n_features: Optional[int] = None, any_float_layout: bool = False, name: str = "X") -> np.ndarray:
    """Return ``X`` as a C-contiguous float64 2D array, or raise ValueError.

    ``n_features`` (when given) is the number of columns seen during fit.
    ``name``: how errors refer to the array (e.g. an eval set's features).
    ``any_float_layout`` (tree_method='hist' fits): a float32 / float64
    array in native byte order is returned as is -- any memory order or
    strides, a memory-mapped file included -- instead of being copied; the
    native binning reads it in place (only unaligned data, or strides that
    are not a whole number of elements, are copied). Other inputs are
    converted to float64.
    """
    if any_float_layout and isinstance(X, np.ndarray) and X.dtype in (np.float32, np.float64) \
            and X.dtype.isnative:
        X = X.view(np.ndarray) if type(X) is not np.ndarray else X  # e.g. np.memmap: a view, no copy
    elif any_float_layout:
        X = np.asarray(X)
        if not (X.dtype in (np.float32, np.float64) and X.dtype.isnative):
            X = np.asarray(X, dtype=np.float64)
    if any_float_layout and (not X.flags.aligned or any(s % X.itemsize for s in X.strides)):
        # e.g. a view of a packed structured array: the native code reads typed elements, so
        # unaligned data or strides that are not a whole number of elements are copied first.
        X = np.ascontiguousarray(X)
    if not any_float_layout:
        X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array for {name}, got an array of shape {X.shape}.")
    if X.shape[0] < 1:
        raise ValueError(f"{name} has 0 samples (shape={X.shape}); at least 1 is required.")
    if X.shape[1] < 1:
        raise ValueError(f"{name} has 0 features (shape={X.shape}); at least 1 is required.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"{name} has {X.shape[1]} features, but this model was fitted with {n_features} features."
        )
    _assert_all_finite(X, name)
    return X


def check_tree_method(tree_method, max_bin, criterion: Optional[str] = None) -> Tuple[str, int]:
    """Validate ``tree_method`` / ``max_bin``; MAE has no histogram search."""
    tree_method = str(tree_method).lower()
    if tree_method not in ("exact", "hist"):
        raise ValueError(f"tree_method must be 'exact' or 'hist', got {tree_method!r}")
    if isinstance(max_bin, (bool, np.bool_)) or not isinstance(max_bin, (int, np.integer)) \
            or not 2 <= max_bin <= 65536:
        raise ValueError(f"max_bin must be an int in [2, 65536], got {max_bin!r}")
    if tree_method == "hist" and criterion == "mae":
        raise NotImplementedError("tree_method='hist' is not implemented for criterion='mae'; use 'exact'")
    return tree_method, int(max_bin)


def check_hist_subtraction(hist_subtraction, max_hist_cache_mb) -> Tuple[bool, float]:
    """Validate ``hist_subtraction`` / ``max_hist_cache_mb``."""
    if not isinstance(hist_subtraction, (bool, np.bool_)):
        raise ValueError(f"hist_subtraction must be a bool, got {hist_subtraction!r}")
    if isinstance(max_hist_cache_mb, (bool, np.bool_)) or not isinstance(max_hist_cache_mb, (int, float, np.integer,
                                                                                             np.floating)) \
            or not np.isfinite(max_hist_cache_mb) or max_hist_cache_mb < 0:
        raise ValueError(f"max_hist_cache_mb must be a finite number >= 0, got {max_hist_cache_mb!r}")
    return bool(hist_subtraction), float(max_hist_cache_mb)


def check_sample_weight(sample_weight, n_samples: int, name: str = "sample_weight", x_name: str = "X") -> np.ndarray:
    """Return the weights as a C-contiguous float64 1D array of length
    ``n_samples``, or raise ValueError unless they are finite, non-negative
    and not all zero.

    Applied to ``sample_weight`` itself and again to the *effective* weights
    a model fits with (after ``class_weight`` / ``scale_pos_weight``).
    """
    sample_weight = np.asarray(sample_weight, dtype=np.float64, order="C")
    if sample_weight.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got an array of shape {sample_weight.shape}.")
    if sample_weight.shape[0] != n_samples:
        raise ValueError(f"{x_name} has {n_samples} samples, but {name} has {sample_weight.shape[0]}.")
    _assert_all_finite(sample_weight, name)
    if np.any(sample_weight < 0):
        raise ValueError(f"{name} must be non-negative.")
    if not np.any(sample_weight):
        raise ValueError(f"{name} must contain at least one non-zero value.")
    return sample_weight


def check_X_y(X, y, sample_weight=None, any_float_layout: bool = False
              ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate fit inputs: shapes agree, everything finite, weights
    non-negative and not all zero. ``any_float_layout``: see ``check_X``."""
    X = check_X(X, any_float_layout=any_float_layout)
    n_samples = X.shape[0]

    y = np.asarray(y, dtype=np.float64, order="C")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got an array of shape {y.shape}.")
    if y.shape[0] != n_samples:
        raise ValueError(f"X has {n_samples} samples, but y has {y.shape[0]}.")
    _assert_all_finite(y, "y")

    if sample_weight is not None:
        sample_weight = check_sample_weight(sample_weight, n_samples)

    return X, y, sample_weight


def check_eval_set(eval_set, eval_sample_weight, n_features: int, any_float_layout: bool = False,
                   classes: Optional[np.ndarray] = None
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Validate ``fit``'s ``eval_set`` / ``eval_sample_weight`` (XGBoost's
    sklearn API, a single validation set): ``eval_set`` is a ``(X_val, y_val)``
    tuple or a list holding exactly one such pair; ``eval_sample_weight`` a 1D
    array (or a list holding exactly one). ``X_val`` follows ``X``'s rules
    (``any_float_layout``: see ``check_X``) and must have ``n_features``
    columns; ``y_val``'s labels must be among ``classes`` (classification).
    Returns ``(X_val, y_val, w_val)``, or None without an eval set.
    """
    if eval_set is None:
        if eval_sample_weight is not None:
            raise ValueError("eval_sample_weight was given without eval_set.")
        return None
    if isinstance(eval_set, tuple) and len(eval_set) == 2:
        pair = eval_set
    elif isinstance(eval_set, list):
        if len(eval_set) != 1:
            raise ValueError(
                f"eval_set takes a single validation set, (X_val, y_val) or [(X_val, y_val)]; got a list of "
                f"{len(eval_set)} entries. Early stopping and evals_result_ use exactly one eval set.")
        pair = eval_set[0]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("eval_set's entry must be an (X_val, y_val) pair.")
    else:
        raise ValueError("eval_set must be an (X_val, y_val) tuple or a list holding one such tuple, "
                         f"got {type(eval_set).__name__}.")
    X_val = check_X(pair[0], n_features, any_float_layout=any_float_layout, name="eval_set X_val")
    m = X_val.shape[0]
    y_val = np.asarray(pair[1], dtype=np.float64, order="C")
    if y_val.ndim != 1:
        raise ValueError(f"eval_set y_val must be a 1D array, got an array of shape {y_val.shape}.")
    if y_val.shape[0] != m:
        raise ValueError(f"eval_set X_val has {m} samples, but y_val has {y_val.shape[0]}.")
    _assert_all_finite(y_val, "eval_set y_val")
    if classes is not None:
        unseen = np.setdiff1d(np.unique(y_val), classes)
        if unseen.size:
            raise ValueError(f"eval_set y_val has labels not seen in the training y: {unseen[:10].tolist()}.")
    w_val = None
    if eval_sample_weight is not None:
        w = eval_sample_weight
        if isinstance(w, (list, tuple)) and len(w) == 1 and np.ndim(w[0]) == 1:
            w = w[0]  # [w_val], like XGBoost's sample_weight_eval_set
        w_val = check_sample_weight(w, m, "eval_sample_weight", x_name="eval_set X_val")
    return X_val, y_val, w_val
