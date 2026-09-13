"""Toy tests only: no benchmark-corpus witness is discarded here."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("decoder_search_v2", Path(__file__).with_name("adapter.py"))
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def repetition():
    return (
        np.empty((0, 3), dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        np.array([[1, 0, 0]], dtype=np.uint8),
    )


def test_actual_decoder_repetition_and_json_counters():
    events = []
    counters = ADAPTER.search(*repetition(), 0.04, 42, lambda *event: events.append(event))
    assert counters["trials"] > 0
    assert counters["failed_decodes"] == 0
    assert events == [(3, [0, 1, 2], "decoder")]
    assert counters["distinct_exports"] == 1
    assert counters["successful_decodes"] == 1 + counters["duplicate_outputs"]
    json.dumps(counters)


def test_actual_decoder_steane_and_column_relabeling():
    checks = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    duals = np.ones((1, 7), dtype=np.uint8)
    for columns in [np.arange(7), np.random.default_rng(7).permutation(7)]:
        opposite = checks[:, columns]
        augmented = np.vstack([opposite, np.zeros(7, dtype=np.uint8)])
        syndrome = np.array([0, 0, 0, 1], dtype=np.uint8)
        for seed in range(4):
            word, result = ADAPTER.decode_trial(
                opposite, duals[:, columns], augmented, syndrome, np.random.default_rng(seed)
            )
            assert result["valid"]
            assert not ((opposite @ word) & 1).any()
            # Approximate decoding need not find the distance-three representative.
            assert word.sum() in (3, 7)


def test_inverse_permutation_and_odd_augmented_syndrome(monkeypatch):
    matrices = []
    own = np.empty((0, 5), dtype=np.uint8)
    opposite = np.array([[1, 1, 0, 0, 0]], dtype=np.uint8)
    duals = np.array([[0, 0, 1, 0, 0]], dtype=np.uint8)

    class Decoder:
        converge = True
        iter = 1

        def __init__(self, pcm, **kwargs):
            matrices.append(pcm.copy())
            assert kwargs["omp_thread_count"] == 1
            self.pcm = pcm

        def decode(self, syndrome):
            assert syndrome.tolist() == [0, 1]
            return self.pcm[-1].copy()

    monkeypatch.setattr(ADAPTER, "BpOsdDecoder", Decoder)
    events = []
    ADAPTER.search(own, opposite, duals, 0.01, 123, lambda *event: events.append(event))
    assert events == [(1, [2], "decoder")]
    assert any(matrix[-1].tolist() != duals[0].tolist() for matrix in matrices)


def test_failed_decode_is_counted_and_not_exported(monkeypatch):
    def failed(opposite, duals, augmented, syndrome, rng):
        return np.zeros(3, dtype=np.uint8), {
            "valid": False,
            "bp_converged": False,
            "bp_iterations": 100,
            "detector_weight": 1,
        }

    monkeypatch.setattr(ADAPTER, "decode_trial", failed)
    events = []
    result = ADAPTER.search(*repetition(), 0.01, 1, lambda *event: events.append(event))
    assert not events
    assert result["failed_decodes"] == result["trials"] > 0


def test_late_result_is_exported_and_persistence_failure_propagates(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ADAPTER.time, "perf_counter", lambda: now[0])

    def late(*args):
        now[0] = 2.0
        return np.ones(3, dtype=np.uint8), {
            "valid": True,
            "bp_converged": False,
            "bp_iterations": 100,
            "detector_weight": 1,
        }

    monkeypatch.setattr(ADAPTER, "decode_trial", late)
    events = []
    result = ADAPTER.search(*repetition(), 1.0, 1, lambda *event: events.append(event))
    assert events == [(3, [0, 1, 2], "decoder")]
    assert result["late_exports"] == 1
    assert result["trials"] == 1
    now[0] = 0.0

    def fail(*event):
        raise OSError("witness persistence failed")

    with pytest.raises(OSError, match="witness persistence"):
        ADAPTER.search(*repetition(), 1.0, 1, fail)


def test_zero_budget_and_no_logicals():
    events = []
    assert ADAPTER.search(*repetition(), 0.0, 1, lambda *event: events.append(event))["trials"] == 0
    own, opposite, _ = repetition()
    counters = ADAPTER.search(
        own, opposite, np.empty((0, 3), dtype=np.uint8), 10.0, 1, lambda *event: events.append(event)
    )
    assert counters["status"] == "no_logicals"
    assert not events
