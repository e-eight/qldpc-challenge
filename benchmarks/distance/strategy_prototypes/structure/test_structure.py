"""Toy witnesses are validated and retained through the repository submission kit."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify"))
sys.path.insert(0, str(ROOT / "research" / "kit"))
import gf2
from submit import make_submission, save_submission

spec = importlib.util.spec_from_file_location("structure_adapter", HERE / "adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def collector(hx, hz, fallback_z, name):
    events = []
    directory = ROOT / "research" / "candidates" / "strategy-structure-tests"
    directory.mkdir(parents=True, exist_ok=True)

    def emit(weight, support):
        # Persist before assertions: failed assertions must not discard a witness.
        doc = make_submission(hx, hz, name=name, construction="Unit-test CSS fixture",
                              authors=["benchmark"], witnesses={"X": support, "Z": fallback_z})
        errors = save_submission(doc, directory / f"{name}-{len(events)}.json")
        assert not errors
        vector = np.zeros(hx.shape[1], dtype=np.int8)
        vector[support] = 1
        assert weight == len(support) == int(vector.sum())
        assert gf2.commutes(vector, hz)
        assert not gf2.in_rowspace(vector, hx)
        if events:
            assert weight < events[-1][0]
        events.append((weight, support))

    return events, emit


@pytest.mark.parametrize("n", [1, 63, 64, 65, 127, 128, 129])
def test_repetition_word_boundaries(n):
    hx = np.zeros((0, n), dtype=np.uint8)
    hz = np.zeros((max(0, n - 1), n), dtype=np.uint8)
    for r in range(n - 1):
        hz[r, r:r + 2] = 1
    events, emit = collector(hx, hz, [0], f"repetition-{n}")
    result = adapter.prepare(hx, hz).run(0.05, 7, emit)
    assert result["best_weight"] == n
    assert events[-1] == (n, list(range(n)))
    assert result["elapsed_seconds"] < 2


def test_stabilizers_rejected_and_disconnected_graph_terminates():
    hx = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    hz = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    events, emit = collector(hx, hz, [2], "degenerate-disconnected")
    result = adapter.prepare(hx, hz).run(0.05, 9, emit)
    assert events == [(2, [2, 3])]
    assert result["dependencies"] > result["improvements"]
    assert result["elapsed_seconds"] < 2


def test_empty_budget_and_no_logicals():
    hx = np.eye(65, dtype=np.uint8)
    hz = np.zeros((0, 65), dtype=np.uint8)
    search = adapter.prepare(hx, hz)
    for seconds in [0, 0.1]:
        result = search.run(seconds, 0, lambda *_: pytest.fail("unexpected logical"))
        assert result["columns"] == 0
        assert result["best_weight"] is None
    for seconds in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            search.run(seconds, 0, lambda *_: None)


def test_input_validation():
    with pytest.raises(ValueError, match="commute"):
        adapter.prepare(np.ones((1, 3), dtype=np.uint8), np.ones((1, 3), dtype=np.uint8))
