import itertools

import numpy as np
import pytest
from run import gf2
from strategy_prototypes.component_search.adapter import Search, components, compress_duals


def test_components_match_independent_reachability_and_relabel():
    h = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]], dtype=np.int8)
    assert components(h, list(range(6))) == [[0, 1, 2], [3, 4], [5]]
    assert components(h, [0, 2, 3, 5]) == [[0], [2], [3], [5]]
    p = np.array([4, 2, 5, 0, 3, 1])
    actual = {frozenset(p[g]) for g in components(h[::-1][:, p], range(6))}
    assert actual == {frozenset(g) for g in components(h, range(6))}


@pytest.mark.parametrize("coords", [[0, 0], [-1], [6]])
def test_bad_coordinates(coords):
    with pytest.raises(ValueError):
        components(np.zeros((0, 6), dtype=np.int8), coords)


def test_high_k_logical_compression_preserves_entire_pairing_kernel():
    rng = np.random.default_rng(19)
    basis = rng.integers(0, 2, (8, 90), dtype=np.int8)
    duals = rng.integers(0, 2, (75, 90), dtype=np.int8)
    reduced, columns = compress_duals(basis, duals)
    assert len(reduced) <= 8 and np.array_equal(reduced, duals[columns])
    for bits in itertools.product(range(2), repeat=8):
        word = np.array(bits, dtype=np.int8) @ basis % 2
        assert bool(np.any(duals @ word % 2)) == bool(np.any(reduced @ word % 2))


@pytest.mark.parametrize("method", ["metadata-components", "auto-components", "exact-components"])
def test_search_exact_supports_and_exhaustion(method):
    own = np.zeros((0, 6), dtype=np.int8)
    opposite = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1]], dtype=np.int8)
    events = []
    result = Search(own, opposite, method, [list(range(3)), list(range(3, 6))]).run(
        0.15, 17, lambda *e: events.append(e)
    )
    assert events and min(e[0] for e in events) == 3
    spaces = [s for s in result["spaces"] if "stats" in s]
    assert len(spaces) == 2
    for s in spaces:
        assert s["dimension"] == 1 and s["logical_rank"] == 1
        assert s["status"] == "exhausted" and s["stats"]["enumerated"] == 1
        assert s["stats"]["best"] == 3
    for weight, support, stage in events:
        v = np.zeros(6, dtype=np.int8)
        v[support] = 1
        assert gf2.commutes(v, opposite) and not gf2.in_rowspace(v, own)
    if method == "exact-components":
        assert "fallback" not in result


def test_callback_failure_stops_search():
    def fail(*args):
        raise OSError("persist failed")

    with pytest.raises(OSError, match="persist failed"):
        Search(np.zeros((0, 4), dtype=np.int8), np.zeros((0, 4), dtype=np.int8), "exact-components").run(0.1, 0, fail)
