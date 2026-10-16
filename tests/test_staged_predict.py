"""Staged predictions of GradientBoostingRDT and the ensembles' depth_features_.

- Stage r of staged_decision_function / staged_predict_proba / staged_predict
  is exactly the prediction of a model fitted with n_estimators=r (each
  round's subsets are seeded by its round index, so the first r rounds do not
  depend on n_estimators).
- The last stage is bitwise equal to decision_function / predict_proba /
  predict, including when early stopping fitted fewer rounds than requested.
- Every stage is a new array: nothing yielded earlier changes later.
- depth_features_ maps each tree's own feature indices back to columns of X.
"""

import numpy as np
import pytest

from rdt.ensemble import GradientBoostingRDT, RandomForestRDT


def _data(task, n_classes=2, n=160, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = np.round(rng.standard_normal((n, p)) * 3) / 3  # ties
    if task == "regression":
        y = 2 * X[:, 0] + np.sin(3 * X[:, 1]) + 0.2 * rng.standard_normal(n)
    else:
        score = X[:, 0] + 0.7 * X[:, 1] + 0.4 * rng.standard_normal(n)
        y = np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])).astype(float)
    w = rng.uniform(0.2, 2.0, n)
    return X, y, w


CONFIGS = [
    dict(task="classification", loss="log_loss", n_classes=2),
    dict(task="classification", loss="log_loss", n_classes=3),
    dict(task="classification", loss="exponential", n_classes=2),
    dict(task="regression", loss="squared_error"),
    dict(task="regression", loss="huber"),
]


def _gb(cfg, **kw):
    params = dict(task=cfg["task"], loss=cfg["loss"], max_depth=3, learning_rate=0.3, subsample=0.8,
                  colsample_bytree=0.6, reg_lambda=0.5, random_state=3)
    params.update(kw)
    return GradientBoostingRDT(**params)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c['task']}-{c['loss']}-{c.get('n_classes', '')}")
def test_stage_r_equals_model_with_r_rounds(cfg):
    X, y, w = _data(cfg["task"], cfg.get("n_classes", 2))
    R = 6
    full = _gb(cfg, n_estimators=R).fit(X, y, sample_weight=w)
    stages_pred = list(full.staged_predict(X))
    assert len(stages_pred) == R
    if cfg["task"] == "classification":
        stages_dec = list(full.staged_decision_function(X))
        stages_proba = list(full.staged_predict_proba(X))
    for r in range(1, R + 1):
        m = _gb(cfg, n_estimators=r).fit(X, y, sample_weight=w)
        assert np.array_equal(stages_pred[r - 1], m.predict(X))
        if cfg["task"] == "classification":
            assert np.array_equal(stages_dec[r - 1], m.decision_function(X))
            assert np.array_equal(stages_proba[r - 1], m.predict_proba(X))


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c['task']}-{c['loss']}-{c.get('n_classes', '')}")
def test_last_stage_bitwise_equals_predict_with_early_stopping(cfg):
    X, y, w = _data(cfg["task"], cfg.get("n_classes", 2), n=300, seed=1)
    m = _gb(cfg, n_estimators=300, learning_rate=0.5, n_iter_no_change=2, validation_fraction=0.3)
    m.fit(X, y, sample_weight=w)
    assert m.n_estimators_ < 300
    Xt = np.random.default_rng(5).standard_normal((70, X.shape[1]))
    stages = list(m.staged_predict(Xt))
    assert len(stages) == m.n_estimators_
    assert np.array_equal(stages[-1], m.predict(Xt))
    if cfg["task"] == "classification":
        dec = list(m.staged_decision_function(Xt))
        proba = list(m.staged_predict_proba(Xt))
        assert len(dec) == len(proba) == m.n_estimators_
        assert np.array_equal(dec[-1], m.decision_function(Xt))
        assert np.array_equal(proba[-1], m.predict_proba(Xt))


def test_decision_function_shapes_and_link():
    X, y, _ = _data("classification", 2)
    for loss, scale in (("log_loss", 1.0), ("exponential", 2.0)):
        m = GradientBoostingRDT(n_estimators=5, loss=loss, random_state=0).fit(X, y)
        d = m.decision_function(X)
        assert d.shape == (len(X),)
        np.testing.assert_allclose(m.predict_proba(X)[:, 1], 1 / (1 + np.exp(-scale * d)), rtol=1e-12)
    X, y, _ = _data("classification", 4)
    m = GradientBoostingRDT(n_estimators=5, random_state=0).fit(X, y)
    d = m.decision_function(X)
    assert d.shape == (len(X), 4)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    np.testing.assert_allclose(m.predict_proba(X), e / e.sum(axis=1, keepdims=True), rtol=1e-12)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_staged_yields_are_independent_arrays(n_classes):
    X, y, _ = _data("classification", n_classes)
    m = GradientBoostingRDT(n_estimators=8, learning_rate=0.5, random_state=0).fit(X, y)
    for name in ("staged_decision_function", "staged_predict_proba", "staged_predict"):
        gen = getattr(m, name)
        kept, snapshots = [], []
        for stage in gen(X[:1]):  # a single row: the case a transposed view used to alias
            kept.append(stage)
            snapshots.append(stage.copy())
        for a, b in zip(kept, snapshots):
            assert np.array_equal(a, b)  # not overwritten by later stages
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                assert not np.shares_memory(kept[i], kept[j])
        if name != "staged_predict":
            assert not np.array_equal(snapshots[0], snapshots[-1])  # the stages do move


def test_staged_validates_X_eagerly_and_classification_only():
    X, y, _ = _data("classification", 2)
    m = GradientBoostingRDT(n_estimators=3, random_state=0)
    with pytest.raises(ValueError, match="not fitted"):
        m.staged_predict(X)
    m.fit(X, y)
    with pytest.raises(ValueError, match="features"):
        m.staged_predict_proba(X[:, :2])  # raises at the call, before iterating
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m.staged_decision_function(bad)
    Xr, yr, _ = _data("regression")
    r = GradientBoostingRDT(task="regression", n_estimators=3, random_state=0).fit(Xr, yr)
    for name in ("decision_function", "staged_decision_function", "staged_predict_proba"):
        with pytest.raises(ValueError, match="classification"):
            getattr(r, name)(Xr)
    assert len(list(r.staged_predict(Xr))) == 3


def test_staged_iterator_and_refit():
    X, y, _ = _data("classification", 2)
    m = GradientBoostingRDT(n_estimators=4, random_state=0).fit(X, y)
    expected = [s.copy() for s in m.staged_predict_proba(X)]
    gen = m.staged_predict_proba(X)
    first = next(gen)
    m.fit(X, 1 - y)  # the estimator publishes a new native model; the generator keeps the old one
    rest = list(gen)
    assert all(np.array_equal(a, b) for a, b in zip([first] + rest, expected))

    # The generator is bound to the model at the staged_* call, not at the
    # first next(): a refit in between does not change what it yields.
    m.fit(X, y)
    gen = m.staged_predict_proba(X)
    m.fit(X, 1 - y)
    assert all(np.array_equal(a, b) for a, b in zip(list(gen), expected))
    assert len(expected) == 4

    # Refitting the very native object a native iterator reads is detected.
    it = m._model._staged(np.ascontiguousarray(X), "proba")
    next(it)
    m._model.fit(np.ascontiguousarray(X), np.ascontiguousarray(y.astype(float)), None)
    with pytest.raises(RuntimeError, match="refitted"):
        next(it)


def test_depth_features_in_original_column_space():
    rng = np.random.default_rng(0)
    n, p, informative = 400, 10, 7
    X = rng.standard_normal((n, p))
    y = (X[:, informative] > 0.1).astype(float)

    rf = RandomForestRDT(n_estimators=30, max_depth=2, max_features=3, random_state=0).fit(X, y)
    subsets = rf._model.feature_subsets_
    assert len(rf.depth_features_) == 30
    for t, (df, subset) in enumerate(zip(rf.depth_features_, subsets)):
        local = rf.estimators_[t].depth_features_
        assert df == {d: subset[f] for d, f in local.items()}
        if informative in subset:
            assert df[0] == informative
    assert any(df.get(0) == informative for df in rf.depth_features_)

    gb = GradientBoostingRDT(n_estimators=20, max_depth=1, colsample_bytree=0.3, random_state=0).fit(X, y)
    roots = [df[0] for df in gb.depth_features_ if 0 in df]
    # The subsets have 3 columns, so a tree's own indices are 0..2: seeing 7 means the
    # indices were mapped to X's columns.
    assert informative in roots
    assert all(0 <= f < p for f in roots)

    gb3 = GradientBoostingRDT(n_estimators=4, max_depth=2, random_state=0).fit(
        X, np.digitize(X[:, informative], [-0.5, 0.5]).astype(float))
    assert len(gb3.depth_features_) == 4 * 3  # one tree per class per round
    assert all(df[0] == informative for df in gb3.depth_features_)


_CONCURRENT_NEXT = r"""
import threading
import numpy as np
from rdt.ensemble import GradientBoostingRDT

rng = np.random.default_rng(0)
X = rng.standard_normal((2000, 4))
y = (X[:, 0] + 0.5 * rng.standard_normal(2000) > 0).astype(float)
m = GradientBoostingRDT(n_estimators=3, max_depth=6, random_state=0).fit(X, y)
Xbig = np.ascontiguousarray(np.tile(X, (150, 1)))  # each round takes a while
it = m._model._staged(Xbig, "proba")
outcomes = []

def advance():
    try:
        next(it)
        outcomes.append("ok")
    except RuntimeError as e:
        outcomes.append("busy" if "already" in str(e) else repr(e))

threads = [threading.Thread(target=advance) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert outcomes.count("ok") >= 1 and set(outcomes) <= {"ok", "busy"}, outcomes
print("done", sorted(outcomes))
"""


def test_native_staged_iterator_concurrent_next_does_not_deadlock():
    # The native iterator advances with the GIL released. A second thread
    # calling __next__ on the same native iterator meanwhile must not block
    # on it while holding the GIL (the first thread then waits for the GIL
    # forever); it raises RuntimeError instead. (The public staged_* methods
    # wrap the iterator in a generator, which refuses re-entry itself.)
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-c", _CONCURRENT_NEXT], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "done" in proc.stdout
