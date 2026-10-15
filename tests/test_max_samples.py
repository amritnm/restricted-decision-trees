"""RandomForestRDT(max_samples=...) follows scikit-learn: an int is a row
count, a float a fraction of n_samples, None means n_samples.

Up to 0.2.0 an int was multiplied by n_samples like a fraction
(max_samples=50 on 400 rows drew 20,000 rows per tree).
"""

from __future__ import annotations

import numpy as np
import pytest

from rdt.ensemble import RandomForestRDT


def _rows_per_tree(max_samples, n=400, bootstrap=True):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(float)
    rf = RandomForestRDT(n_estimators=3, max_depth=2, max_features=None, max_samples=max_samples,
                         bootstrap=bootstrap, random_state=0).fit(X, y)
    return [t.root.n_samples for t in rf.estimators_]


def test_int_max_samples_is_a_row_count():
    # max_samples=1 now draws a single row per tree (0.2.0: all rows)
    for max_samples, expected in ((50, 50), (1, 1), (1000, 1000), (np.int64(37), 37)):
        assert _rows_per_tree(max_samples) == [expected] * 3


def test_float_and_none_max_samples():
    assert _rows_per_tree(0.125) == [50] * 3      # fraction, unchanged
    assert _rows_per_tree(2.0) == [800] * 3       # > 1 still allowed, as in scikit-learn
    assert _rows_per_tree(None) == [400] * 3      # newly accepted
    assert _rows_per_tree(5, bootstrap=False) == [400] * 3  # ignored without bootstrap


def test_invalid_max_samples_raise():
    # (float("inf") is rejected too; it is left out because 0.2.0 cast it to a
    # multi-gigabyte bootstrap size, so running this file against 0.2.0 to see
    # the failures would exhaust memory instead.)
    for bad in (0, -3, 0.0, -0.5, float("nan"), True, "half"):
        with pytest.raises(ValueError, match="max_samples"):
            _rows_per_tree(bad)
