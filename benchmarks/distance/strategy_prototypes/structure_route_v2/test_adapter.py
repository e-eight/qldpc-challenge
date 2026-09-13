"""Synthetic route and preservation tests; no corpus searches."""

import time

import numpy as np
import pytest
from strategy_prototypes.structure_route_v2 import adapter
from study_strategies import ris_native


def toy():
    first = np.zeros(31, dtype=np.uint8)
    first[[0, 1, 4, 8]] = 1
    a = np.asarray([np.roll(first, i) for i in range(31)])
    first[:] = 0
    first[[0, 2, 8, 13, 17, 21]] = 1
    b = np.asarray([np.roll(first, i) for i in range(31)])
    return np.concatenate([a, b], axis=1), np.concatenate([b.T, a.T], axis=1)


def test_recovered_route_returns_original_coordinate_witnesses():
    own, opposite = toy()
    permutation = np.random.default_rng(201).permutation(62)
    own, opposite = own[:, permutation], opposite[:, permutation]
    duals = ris_native.Prepared(own, opposite).logicals
    events = []
    result = adapter.search(own, opposite, duals, 0.10, 5, lambda *e: events.append(e))
    assert result["applicable"] and events
    assert result["exported"] == len(events)
    assert result["routes"][0]["block_size"] == 31
    for weight, support, stage in events:
        assert stage == "routed_single_block" and weight == len(support) == 31
        candidate = np.zeros(62, dtype=np.uint8)
        candidate[support] = 1
        assert not np.any(opposite @ candidate % 2)
        assert np.any(duals @ candidate % 2)
        assert set(permutation[support]) in (set(range(31)), set(range(31, 62)))


def test_two_equal_cycles_only():
    assert adapter.two_cycle_order(np.array([1, 2, 0, 4, 5, 3])).tolist() == list(range(6))
    assert adapter.two_cycle_order(np.arange(6)) is None
    assert adapter.two_cycle_order(np.array([1, 0, 3, 4, 5, 2])) is None
    assert adapter.two_cycle_order(None) is None


def test_no_route(monkeypatch):
    own, opposite = toy()
    monkeypatch.setattr(adapter, "recover_translation", lambda h: np.arange(h.shape[1]))
    events = []
    result = adapter.search(
        own, opposite, ris_native.Prepared(own, opposite).logicals, 0.10, 0, lambda *e: events.append(e)
    )
    assert result["status"] == "no_route" and not result["applicable"] and not events


def test_zero_budget():
    own, opposite = toy()
    events = []
    result = adapter.search(
        own, opposite, ris_native.Prepared(own, opposite).logicals, 0, 0, lambda *e: events.append(e)
    )
    assert result["status"] == "setup_deadline" and not events


def test_callback_failure_propagates():
    own, opposite = toy()

    def fail(*args):
        raise OSError("synthetic save failure")

    with pytest.raises(OSError, match="synthetic save failure"):
        adapter.search(own, opposite, ris_native.Prepared(own, opposite).logicals, 0.10, 5, fail)


def test_late_native_return_is_preserved(monkeypatch):
    monkeypatch.setattr(adapter, "recover_translation", lambda h: np.array([1, 2, 0, 4, 5, 3]))

    class Prepared:
        def __init__(self, *args):
            pass

        def applicable(self):
            return True

        def batch(self, *args):
            time.sleep(0.010)
            return 1, [0], 1

    monkeypatch.setattr(adapter.benchmark_native, "PreparedSearch", Prepared)
    rows = np.zeros((0, 6), dtype=np.uint8)
    events = []
    result = adapter.search(rows, rows, np.eye(6, dtype=np.uint8), 0.005, 0, lambda *e: events.append(e))
    assert events == [(1, [0], "routed_single_block")]
    assert result["elapsed_seconds"] > 0.005 and result["exported"] == 1
