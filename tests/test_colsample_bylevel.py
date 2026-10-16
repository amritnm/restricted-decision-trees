"""colsample_bylevel (GradientBoostingRDT, RandomForestRDT).

- Each searched depth's feature is chosen among a fresh subset of
  max(1, floor(f * m)) of the tree's m columns, exposed (replayed from the
  tree's seed) as level_features_.
- Every search path honours the subset: presort on / off agree bit for bit,
  newton included, and hist agrees with exact on few-valued data.
- The draws have their own generator: results do not depend on n_jobs, and
  the other draws (bootstrap rows, column subsets) are unchanged.
- colsample_bylevel=1.0 (the default) changes nothing; the bitwise comparison
  of defaults against the previous build is done outside the test suite.
"""

import math

import numpy as np
import pytest

from rdt.ensemble import GradientBoostingRDT, RandomForestRDT


def _data(kind="binary", n=300, p=10, seed=0, grid=False):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 24, (n, p)) / 4.0 if grid else rng.standard_normal((n, p))
    s = X[:, 0] - X[:, 3] + 0.5 * X[:, 5] * X[:, 6] + rng.normal(0, 0.5, n)
    if kind == "regression":
        return X, s
    if kind == "multi":
        return X, np.digitize(s, np.quantile(s, [0.3, 0.7])).astype(float)
    return X, (s > np.median(s)).astype(float)


def _models(f, **kw):
    return [
        ("gb", GradientBoostingRDT(n_estimators=6, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                   colsample_bylevel=f, random_state=1, **kw)),
        ("gb-multi", GradientBoostingRDT(n_estimators=3, max_depth=3, colsample_bylevel=f, random_state=2, **kw)),
        ("gb-newton", GradientBoostingRDT(n_estimators=6, max_depth=4, split_criterion="newton",
                                          colsample_bylevel=f, random_state=3, **kw)),
        ("rf", RandomForestRDT(n_estimators=8, max_depth=4, max_features=7, colsample_bylevel=f,
                               oob_score=True, random_state=4, **kw)),
    ]


def _y_for(name, seed=0, grid=False):
    return _data("multi" if name == "gb-multi" else "binary", seed=seed, grid=grid)


def _assert_same(a, b, X, exact_importances=True):
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    if exact_importances:
        assert np.array_equal(a.feature_importances_, b.feature_importances_)
    else:  # hist vs exact: the same splits, gains summed in a different order
        np.testing.assert_allclose(a.feature_importances_, b.feature_importances_, rtol=1e-12)
    assert a.depth_features_ == b.depth_features_
    assert a.level_features_ == b.level_features_


@pytest.mark.parametrize("f", [0.15, 0.5])
def test_presort_on_off_bitwise(f):
    for (name, a), (_, b) in zip(_models(f, presort=True), _models(f, presort=False)):
        X, y = _y_for(name)
        _assert_same(a.fit(X, y), b.fit(X, y), X)


def test_hist_honours_the_level_subsets():
    # few-valued features: hist equals exact, so both must use the same candidates
    for (name, a), (_, b) in zip(_models(0.4), _models(0.4, tree_method="hist")):
        X, y = _y_for(name, grid=True)
        _assert_same(a.fit(X, y), b.fit(X, y), X, exact_importances=False)


@pytest.mark.parametrize("f", [0.1, 0.35, 0.8])
def test_chosen_feature_lies_in_its_depth_subset(f):
    for name, m in _models(f):
        X, y = _y_for(name, seed=5)
        m.fit(X, y)
        if name == "rf":
            subsets = m._model.feature_subsets_
        else:
            # boosting's column subsets: the level candidates' union is within them; colsample_bytree=0.8
            # of 10 columns leaves 8 (gb), all 10 otherwise
            subsets = None
        assert len(m.level_features_) == len(m.depth_features_)
        for t, (levels, chosen) in enumerate(zip(m.level_features_, m.depth_features_)):
            m_cols = len(subsets[t]) if subsets is not None else (8 if name == "gb" else 10)
            k = max(1, math.floor(f * m_cols))
            assert sorted(levels) == list(range(len(levels)))
            assert len(chosen) <= len(levels) <= len(chosen) + 1  # a last searched depth may find no split
            for d, cols in levels.items():
                assert len(cols) == k and cols == sorted(set(cols))
                if subsets is not None:
                    assert set(cols) <= set(subsets[t])
            for d, feat in chosen.items():
                assert feat in levels[d]


def test_single_candidate_per_level():
    X, y = _data(n=400, p=12)
    m = RandomForestRDT(n_estimators=5, max_depth=3, max_features=None, colsample_bylevel=0.05,
                        random_state=0).fit(X, y)
    for levels, chosen in zip(m.level_features_, m.depth_features_):
        for d, feat in chosen.items():
            assert levels[d] == [feat]


def test_default_draws_nothing_and_keeps_other_draws():
    X, y = _data()
    full = RandomForestRDT(n_estimators=6, max_depth=3, max_features=5, random_state=0, oob_score=True).fit(X, y)
    one = RandomForestRDT(n_estimators=6, max_depth=3, max_features=5, random_state=0, oob_score=True,
                          colsample_bylevel=1.0).fit(X, y)
    _assert_same(full, one, X)
    for levels, subset in zip(full.level_features_, full._model.feature_subsets_):
        assert all(cols == list(subset) for cols in levels.values())
    half = RandomForestRDT(n_estimators=6, max_depth=3, max_features=5, random_state=0, oob_score=True,
                           colsample_bylevel=0.5).fit(X, y)
    assert half._model.feature_subsets_ == full._model.feature_subsets_  # column draws undisturbed
    # depth 0 of a tree whose level-0 subset contains the default choice keeps it
    for t, levels in enumerate(half.level_features_):
        if full.depth_features_[t].get(0) in levels.get(0, []):
            assert half.depth_features_[t][0] == full.depth_features_[t][0]


@pytest.mark.parametrize("tree_method", ["exact", "hist"])
def test_deterministic_and_n_jobs_invariant(tree_method):
    for name, _ in _models(0.4):
        X, y = _y_for(name, seed=7)
        fits = [dict(_models(0.4, n_jobs=j, tree_method=tree_method))[name].fit(X, y) for j in (1, 2, 4)]
        again = dict(_models(0.4, n_jobs=1, tree_method=tree_method))[name].fit(X, y)
        for m in fits[1:] + [again]:
            _assert_same(fits[0], m, X)


def test_validation():
    for bad in (0, 0.0, -0.5, 1.5, float("nan"), True, "0.5", None):
        with pytest.raises(ValueError, match="colsample_bylevel"):
            GradientBoostingRDT(colsample_bylevel=bad)
        with pytest.raises(ValueError, match="colsample_bylevel"):
            RandomForestRDT(colsample_bylevel=bad)
    assert GradientBoostingRDT(colsample_bylevel=1).colsample_bylevel == 1.0
