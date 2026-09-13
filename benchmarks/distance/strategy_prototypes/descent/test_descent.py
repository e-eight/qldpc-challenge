"""Toy-only invariant checks; never search or discard corpus witnesses."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _stabilizer_descent import Descent  # noqa: E402


def span(rows):
    result = {bytes(rows.shape[1])}
    for row in rows:
        result |= {bytes(np.frombuffer(value, dtype=np.uint8) ^ row) for value in result}
    return result


@pytest.mark.parametrize("n", [9, 63, 64, 65, 129])
def test_shortening_preserves_coset_and_syndrome(n):
    own = np.zeros((4, n), dtype=np.uint8)
    own[0, [0, 1, 2]] = 1
    own[1, [1, 3, 4]] = 1
    own[2, [4, 5, 6]] = 1
    own[3] = own[0] ^ own[1]  # Include a dependent stabilizer.
    opposite = np.zeros((1, n), dtype=np.uint8)
    opposite[0, [0, 1, 3]] = 1  # Commutes with each own row.
    seed = own[0].copy()
    seed[-1] = 1
    stabilizers = span(own)
    assert not np.any((own @ opposite.T) % 2)
    assert bytes(seed) not in stabilizers
    result = Descent(own).run(np.flatnonzero(seed).tolist(), 1.0, 30, 300, True)
    assert result["best_weight"] == 1
    assert result["steps"] == 300
    previous = int(seed.sum())
    for weight, support in result["improvements"]:
        word = np.zeros(n, dtype=np.uint8)
        word[support] = 1
        assert weight == len(support) < previous
        assert bytes(seed ^ word) in stabilizers
        assert bytes(word) not in stabilizers
        assert not np.any((opposite @ word) % 2)
        previous = weight


def test_empty_stabilizers_and_zero_time():
    solver = Descent(np.zeros((0, 7), dtype=np.uint8))
    assert solver.run([6], 0.01, 0)["steps"] == 0
    solver = Descent(np.eye(7, dtype=np.uint8))
    result = solver.run([6], 0.0, 0)
    assert result["steps"] == 0
    assert result["best_weight"] == 1


def test_reproducibility_with_fixed_step_limit():
    own = np.random.default_rng(33).integers(0, 2, (6, 15), dtype=np.uint8)
    solver = Descent(own)
    a = solver.run([0, 3, 7, 8, 10, 13], 1.0, 7, 500, True)
    b = solver.run([0, 3, 7, 8, 10, 13], 1.0, 7, 500, True)
    for key in ("improvements", "best_weight", "steps", "restarts", "uphill_moves", "neutral_moves"):
        assert a[key] == b[key]


def test_invalid_support():
    solver = Descent(np.eye(3, dtype=np.uint8))
    for support in ([0, 0], [-1], [3]):
        with pytest.raises(ValueError):
            solver.run(support, 0.01, 0)


def test_cold_adapter_emits_valid_toy_logicals():
    spec = importlib.util.spec_from_file_location("descent_adapter_test", HERE / "adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    own = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.uint8)
    opposite = np.array([[1, 1, 1, 1, 0]], dtype=np.uint8)
    prepared = adapter.prepare(own, opposite)
    events = []
    result = prepared.run(0.03, 16, lambda weight, support: events.append((weight, support)))
    stabilizers = span(own)
    assert events
    assert result["seed_bases"] > 0
    assert result["elapsed_seconds"] < 0.2
    for weight, support in events:
        word = np.zeros(5, dtype=np.uint8)
        word[support] = 1
        assert weight == len(support)
        assert not np.any((opposite @ word) % 2)
        assert bytes(word) not in stabilizers
