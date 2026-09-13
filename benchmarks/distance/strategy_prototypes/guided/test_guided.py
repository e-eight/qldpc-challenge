"""Toy searches only; no corpus witnesses are generated here."""
import numpy as np
import pytest

from benchmarks.distance.strategy_prototypes.guided import adapter


def rank(a):
    a = a.copy()
    r = 0
    for c in range(a.shape[1]):
        pivots = np.flatnonzero(a[r:, c])
        if not len(pivots):
            continue
        j = r + pivots[0]
        a[[r, j]] = a[[j, r]]
        for i in range(len(a)):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        r += 1
        if r == len(a):
            break
    return r


def collect(session, batches):
    result = []
    for count in batches:
        batch = session.advance(count)
        result.extend((e.trial, e.weight, tuple(e.support)) for e in batch.improvements)
    return result, batch


@pytest.mark.parametrize("n", [7, 65, 129])
def test_validity_and_batch_determinism(n):
    rng = np.random.default_rng(12 + n)
    own = rng.integers(0, 2, (2, n), dtype=np.uint8)
    kernel = adapter.native.Prepared(np.zeros((0, n), dtype=np.uint8), own).kernel
    opposite = (rng.integers(0, 2, (2, len(kernel)), dtype=np.uint8) @ kernel) % 2
    prepared = adapter.prepare(own, opposite)
    a, batch_a = collect(prepared.session(23), [513])
    b, batch_b = collect(prepared.session(23), [1, 16, 111, 200, 185])
    assert a == b
    assert a
    assert all(a[i][1] > a[i + 1][1] for i in range(len(a) - 1))
    for _, weight, support in a:
        vector = np.zeros(n, dtype=np.uint8)
        vector[list(support)] = 1
        assert weight == int(vector.sum())
        assert not ((opposite @ vector) % 2).any()
        assert rank(np.vstack([own, vector])) > rank(own)
    assert batch_a.exchanges == batch_b.exchanges
    assert batch_a.accepted_children == batch_b.accepted_children
    assert batch_a.immigrants == 12  # Four initial + 64,128,...,512.


def test_steane_exact_minimum_and_run_emits():
    h = np.array([[1,1,1,1,0,0,0], [1,1,0,0,1,1,0],
                  [1,0,1,0,1,0,1]], dtype=np.uint8)
    prepared = adapter.prepare(h, h)
    events, batch = collect(prepared.session(15), [256])
    assert batch.best_weight == 3
    assert min(e[1] for e in events) == 3
    captured = []
    result = prepared.run(.01, 15, lambda weight, support: captured.append((weight, support)))
    assert captured and min(w for w, _ in captured) == 3
    assert result["trials"] > 0


def test_no_logical_and_zero_budget():
    h = np.eye(4, dtype=np.uint8)
    prepared = adapter.prepare(h, np.zeros((0, 4), dtype=np.uint8))
    def fail(*args):
        raise AssertionError("unexpected witness")
    assert prepared.run(.01, 0, fail)["trials"] == 0
    assert prepared.run(0, 0, fail)["trials"] == 0
