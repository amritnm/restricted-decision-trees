"""Resource-usage contracts of the native core: predict does not copy X, the
GIL is released while fitting/predicting, and ensembles do not retain
per-node training indices."""

import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from rdt import RDT
from rdt.ensemble import GradientBoostingRDT, RandomForestRDT
from rdt.visualize import export_tree_dict, print_tree


# --------------------------------------------------------------------------- no copies of X
_PREDICT_MEMORY_SCRIPT = textwrap.dedent("""
    import resource, sys
    import numpy as np
    from rdt.ensemble import GradientBoostingRDT, RandomForestRDT

    def peak():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024

    kind = sys.argv[1]
    rng = np.random.default_rng(0)
    n, p = 20000, 1000                      # X: 160 MB, filled in blocks (no big temporaries)
    X = np.empty((n, p))
    for i in range(0, n, 500):
        X[i:i + 500] = rng.standard_normal((500, p))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    Xfit, yfit = np.ascontiguousarray(X[:1000]), y[:1000]
    if kind == "gb":
        m = GradientBoostingRDT(n_estimators=4, max_depth=3, random_state=0).fit(Xfit, yfit)
    else:
        m = RandomForestRDT(n_estimators=4, max_depth=3, max_features=None, random_state=0).fit(Xfit, yfit)
    m.predict_proba(X[:10])
    before = peak()
    m.predict_proba(X)
    m.predict(X)
    print(peak() - before, X.nbytes)
""")


@pytest.mark.parametrize("kind", ["gb", "rf"])
def test_predict_does_not_copy_X(kind):
    pytest.importorskip("resource")
    out = subprocess.run([sys.executable, "-c", _PREDICT_MEMORY_SCRIPT, kind],
                         capture_output=True, text=True, check=True).stdout.split()
    growth, x_bytes = int(out[0]), int(out[1])
    # v0.2.0 materialized X[:, tree_columns] for every tree (here: all of X).
    assert growth < 0.2 * x_bytes, f"predict grew peak RSS by {growth / 2**20:.0f} MiB (X is {x_bytes / 2**20:.0f} MiB)"


# --------------------------------------------------------------------------- GIL
def _ticks_during(fn):
    """Run fn() while a Python thread counts; return (ticks while fn ran, seconds)."""
    ticks = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ticks[0] += 1
            time.sleep(0.0005)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    time.sleep(0.05)
    start_ticks = ticks[0]
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    during = ticks[0] - start_ticks
    stop.set()
    t.join()
    return during, elapsed


@pytest.mark.parametrize("model", ["rdt", "rf", "gb"])
def test_fit_releases_the_gil(model):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 40))
    y = (X[:, 0] + rng.normal(size=20000) > 0).astype(float)
    rounds = 10
    while True:
        if model == "rdt":
            est = RDT(max_depth=4 + rounds // 10)
        elif model == "rf":
            est = RandomForestRDT(n_estimators=rounds, max_depth=6, max_features=None, random_state=0)
        else:
            est = GradientBoostingRDT(n_estimators=rounds, max_depth=6, random_state=0)
        ticks, elapsed = _ticks_during(lambda: est.fit(X, y))
        if elapsed >= 0.25 or rounds >= 640:
            break
        rounds *= 2
    # With the GIL held for the whole native call the counter cannot advance at all.
    assert ticks >= 10, f"background thread ran {ticks} times during a {elapsed:.2f}s fit"


def test_predict_releases_the_gil():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5000, 20))
    y = (X[:, 0] > 0).astype(float)
    gb = GradientBoostingRDT(n_estimators=200, max_depth=6, random_state=0).fit(X, y)
    Xbig = np.repeat(X, 40, axis=0)
    ticks, elapsed = _ticks_during(lambda: gb.predict_proba(Xbig))
    assert ticks >= 5 or elapsed < 0.05, f"background thread ran {ticks} times during a {elapsed:.2f}s predict"


# --------------------------------------------------------------------------- node indices
def _nodes(node):
    yield node
    if not node.is_leaf:
        yield from _nodes(node.left)
        yield from _nodes(node.right)


def test_standalone_tree_keeps_node_indices():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(float)
    tree = RDT(max_depth=3).fit(X, y)
    assert list(tree.root.indices) == list(range(300))
    for node in _nodes(tree.root):
        assert len(node.indices) == node.n_samples
        if not node.is_leaf:
            assert node.left.n_samples + node.right.n_samples == node.n_samples
    export_tree_dict(tree, y)  # still supported


def test_bootstrap_trees_do_not_retain_node_indices(capsys):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    rf = RandomForestRDT(n_estimators=3, max_depth=3, max_samples=0.5, random_state=0).fit(X, y)
    for tree in rf.estimators_:
        assert tree.root.n_samples == 200  # bootstrap size
        for node in _nodes(tree.root):
            assert not hasattr(node, "indices")
            with pytest.raises(AttributeError, match="not retained"):
                node.indices
            assert node.n_samples > 0
            if not node.is_leaf:
                assert node.left.n_samples + node.right.n_samples == node.n_samples
    print_tree(rf.estimators_[0])
    assert "samples=200" in capsys.readouterr().out
    with pytest.raises(ValueError, match="indices"):
        export_tree_dict(rf.estimators_[0], y)


def test_forest_without_bootstrap_keeps_node_indices():
    # Its trees are fitted on the training rows themselves, so the indices are meaningful.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(float)
    rf = RandomForestRDT(n_estimators=2, max_depth=3, bootstrap=False, max_features=None, random_state=0).fit(X, y)
    for tree in rf.estimators_:
        assert list(tree.root.indices) == list(range(300))
        for node in _nodes(tree.root):
            assert len(node.indices) == node.n_samples
    d = export_tree_dict(rf.estimators_[0], y)
    assert d["samples"] == 300


# --------------------------------------------------------------------------- concurrent use of one native object
def _native_gb(n_estimators):
    from rdt import _rdt_native
    return _rdt_native.GradientBoostingModel(
        "classification", "log_loss", 6, 2, 1, 0.0, n_estimators, 0.1, 1.0, _rdt_native._max_features_all(),
        0.1, 0, 1e-4, 0.0, 0.0, 1.0, 0.5, 0, 1)


def _slow_fit_model(X, y):
    """A fitted native GB model whose fit() takes >= ~0.3 s here."""
    rounds = 20
    while True:
        m = _native_gb(rounds)
        t0 = time.perf_counter()
        m.fit(X, y, None)
        if time.perf_counter() - t0 >= 0.3 or rounds >= 2560:
            return m
        rounds *= 2


def _calls_while(running, fn):
    """Call fn() repeatedly while `running` (a thread) is alive; return the exceptions raised."""
    raised = []
    while running.is_alive():
        try:
            fn()
        except RuntimeError as e:
            raised.append(e)
        time.sleep(0.002)
    return raised


def test_concurrent_fit_or_predict_during_fit_raises():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20000, 20))
    y = (X[:, 0] + rng.normal(size=20000) > 0).astype(float)
    m = _slow_fit_model(X, y)
    expected = m.predict_proba(X[:50])

    for probe in (lambda: m.predict_proba(X[:50]), lambda: m.fit(X, y, None), lambda: m.train_score_):
        t = threading.Thread(target=m.fit, args=(X, y, None))
        t.start()
        raised = _calls_while(t, probe)
        t.join()
        assert raised and all("another thread" in str(e) for e in raised)
    # the object is intact and usable afterwards
    np.testing.assert_array_equal(m.predict_proba(X[:50]), expected)


def _predict_setup():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(5000, 20))
    y = (X[:, 0] > 0).astype(float)
    m = _native_gb(300)
    m.fit(X, y, None)
    Xbig = np.repeat(X, 40, axis=0)
    return m, X, y, Xbig, m.predict_proba(Xbig)


def test_predicts_on_one_native_object_may_overlap():
    m, _, _, Xbig, expected = _predict_setup()
    spans, results = [], []

    def reader():
        t0 = time.perf_counter()
        results.append(m.predict_proba(Xbig))
        spans.append((t0, time.perf_counter()))

    readers = [threading.Thread(target=reader) for _ in range(2)]
    [t.start() for t in readers]
    [t.join() for t in readers]
    assert len(results) == 2
    for r in results:
        np.testing.assert_array_equal(r, expected)
    assert max(a for a, _ in spans) < min(b for _, b in spans)  # they ran at the same time


def test_fit_during_predict_raises():
    m, X, y, Xbig, expected = _predict_setup()
    results = []

    def reader():
        while True:  # a probing fit() may get in first; refitting on the same data is harmless
            try:
                results.append(m.predict_proba(Xbig))
                return
            except RuntimeError:
                time.sleep(0.01)

    t = threading.Thread(target=reader)
    t.start()
    raised = _calls_while(t, lambda: m.fit(X, y, None))
    t.join()
    assert raised and all("in use by another thread" in str(e) for e in raised)
    np.testing.assert_array_equal(results[0], expected)
