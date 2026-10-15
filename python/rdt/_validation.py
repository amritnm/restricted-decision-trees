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


def check_X(X, n_features: Optional[int] = None) -> np.ndarray:
    """Return ``X`` as a C-contiguous float64 2D array, or raise ValueError.

    ``n_features`` (when given) is the number of columns seen during fit.
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array for X, got an array of shape {X.shape}.")
    if X.shape[0] < 1:
        raise ValueError(f"X has 0 samples (shape={X.shape}); at least 1 is required.")
    if X.shape[1] < 1:
        raise ValueError(f"X has 0 features (shape={X.shape}); at least 1 is required.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but this model was fitted with {n_features} features."
        )
    _assert_all_finite(X, "X")
    return X


def check_sample_weight(sample_weight, n_samples: int, name: str = "sample_weight") -> np.ndarray:
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
        raise ValueError(f"X has {n_samples} samples, but {name} has {sample_weight.shape[0]}.")
    _assert_all_finite(sample_weight, name)
    if np.any(sample_weight < 0):
        raise ValueError(f"{name} must be non-negative.")
    if not np.any(sample_weight):
        raise ValueError(f"{name} must contain at least one non-zero value.")
    return sample_weight


def check_X_y(X, y, sample_weight=None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate fit inputs: shapes agree, everything finite, weights
    non-negative and not all zero."""
    X = check_X(X)
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
