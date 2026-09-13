"""Synthetic detector, mapping, scheduling, and persistence checks."""

import time

import numpy as np
import pytest
from dispatch_search import RoutedSession, Search, detector
from scipy.optimize import linear_sum_assignment
from strategy_prototypes.structure_route_v2.test_adapter import toy
from study_strategies import ris_native


def test_native_recovers_permuted_and_deleted_row_structure():
    own, _ = toy()
    columns = np.random.default_rng(910).permutation(62)
    for rows in (own, np.delete(own, [3, 9, 17, 25], axis=0)):
        result = detector.detect(np.ascontiguousarray(rows[:, columns]), 1.0)
        assert result["status"] == "accepted"
        assert result["confidence"] >= 0.5
        assert sorted(result["order"]) == list(range(62))
        assert set(columns[result["order"][:31]]) in (set(range(31)), set(range(31, 62)))


def test_matching_agrees_with_independent_assignment():
    rng = np.random.default_rng(811)
    for _ in range(20):
        rows = (rng.random((24, 18)) < 0.25).astype(np.uint8)
        result = detector.detect(rows, 1.0, 0)
        scores = rows[:-1].astype(np.int64).T @ rows[1:]
        a, b = linear_sum_assignment(scores, maximize=True)
        expected = scores[a, b].sum() / rows[:-1].sum()
        assert result["confidence"] == pytest.approx(expected)


def test_rejection_and_work_bounds():
    own, _ = toy()
    shuffled = np.ascontiguousarray(own[np.random.default_rng(919).permutation(len(own))])
    assert detector.detect(shuffled, 1.0)["status"] != "accepted"
    assert detector.detect(own, 0)["status"] == "deadline"
    assert detector.detect(np.ones((3, 66), dtype=np.uint8), 1)["status"] == "degree_cap"
    assert detector.detect(np.zeros((2, 4098), dtype=np.uint8), 1)["status"] == "shape_cap"
    large = np.zeros((4096, 4096), dtype=np.uint8)
    result = detector.detect(large, 0.000001)
    assert result["status"] == "deadline" and result["elapsed_seconds"] < 0.1


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan")])
def test_invalid_time(seconds):
    with pytest.raises(ValueError):
        detector.detect(np.zeros((2, 6), dtype=np.uint8), seconds)


def test_binary_and_layout_validation():
    with pytest.raises(ValueError, match="binary"):
        detector.detect(np.full((2, 6), 2, dtype=np.uint8), 1)
    with pytest.raises(TypeError):
        detector.detect(np.zeros((2, 6), dtype=np.float64), 1)


def test_actual_routed_kernel_maps_and_checks_witnesses():
    own, opposite = toy()
    columns = np.random.default_rng(515).permutation(62)
    own, opposite = np.ascontiguousarray(own[:, columns]), np.ascontiguousarray(opposite[:, columns])
    order = detector.detect(own, 1)["order"]
    duals = ris_native.Prepared(own, opposite).logicals
    events = []
    session = RoutedSession(own, opposite, duals, order, 44, lambda *e: events.append(e))
    session.advance_until(time.perf_counter() + 0.03)
    assert events and session.counters["exported"] == len(events)
    for weight, support, stage in events:
        word = np.zeros(62, dtype=np.uint8)
        word[support] = 1
        assert stage == "routed_single_block" and weight == len(support)
        assert not np.any(opposite @ word % 2) and np.any(duals @ word % 2)


def test_native_callback_failure_propagates():
    own, opposite = toy()

    def fail(*args):
        raise OSError("save failure")

    session = RoutedSession(
        own, opposite, ris_native.Prepared(own, opposite).logicals, detector.detect(own, 1)["order"], 0, fail
    )
    with pytest.raises(OSError, match="save failure"):
        session.advance_until(time.perf_counter() + 0.05)


def test_late_return_is_exported(monkeypatch):
    import dispatch_search

    class Prepared:
        def __init__(self, *args):
            pass

        def applicable(self):
            return True

        def batch(self, *args):
            time.sleep(0.01)
            return 1, [0], 1

    monkeypatch.setattr(dispatch_search.benchmark_native, "PreparedSearch", Prepared)
    rows = np.zeros((0, 6), dtype=np.uint8)
    events = []
    session = RoutedSession(rows, rows, np.eye(6, dtype=np.uint8), list(range(6)), 0, lambda *e: events.append(e))
    session.advance_until(time.perf_counter() + 0.001)
    assert events == [(1, [0], "routed_single_block")]


def test_unhelpful_route_yields_to_guided():
    own, opposite = toy()
    own, opposite = own.astype(np.int8), opposite.astype(np.int8)
    events = []
    result = Search(own, opposite).run(0.15, 11, lambda *e: events.append(e))
    assert result["route_decision"] == "pilot_did_not_improve"
    assert result["fallback_seconds"] > 0.05
    assert result["detection_seconds"] < 0.05
    assert result["pilot_best"] == 31 > result["initial_best"]
    assert min(e[0] for e in events) <= result["initial_best"]


def test_unsupported_shape_uses_native_fallback():
    rows = np.zeros((0, 5), dtype=np.uint8)
    events = []
    result = Search(rows, rows).run(0.05, 9, lambda *e: events.append(e))
    assert result["route_decision"] == "no_partition"
    assert result["fallback_seconds"] > 0.02
    assert min(e[0] for e in events) == 1


def test_improving_route_reserves_guided_time(monkeypatch):
    import dispatch_search

    def initialize_high(rows, pool, observe, deadline):
        observe(15, "initialization")
        return {"complete": True}

    def detected(*args):
        return {"status": "accepted", "order": list(range(6))}

    class ImprovingRoute:
        def __init__(self, own, opposite, duals, order, seed, emit):
            self.emit = emit
            self.counters = {"best": None, "applicable": True, "calls": 0}

        def advance_until(self, deadline):
            self.emit(1, [0], "routed_single_block")
            self.counters["best"] = 1
            self.counters["calls"] += 1
            time.sleep(max(0, deadline - time.perf_counter()))

    monkeypatch.setattr(dispatch_search, "initialize", initialize_high)
    monkeypatch.setattr(dispatch_search.detector, "detect", detected)
    monkeypatch.setattr(dispatch_search, "RoutedSession", ImprovingRoute)
    rows = np.zeros((2, 6), dtype=np.uint8)
    events = []
    result = Search(rows, rows).run(0.1, 0, lambda *e: events.append(e))
    assert result["route_decision"] == "continue_improving_route"
    assert result["route"]["calls"] == 2
    assert result["fallback_seconds"] >= 0.01
    assert any(stage == "guided" for _, _, stage in events)


def test_invalid_seed_rejected_before_emission():
    rows = np.zeros((0, 5), dtype=np.uint8)
    events = []
    with pytest.raises(ValueError, match="Seed"):
        Search(rows, rows).run(1, -1, lambda *e: events.append(e))
    assert not events


def test_decision_audit_rejects_corrupt_route():
    import copy

    from audit_dispatch import audit_worker

    own, opposite = toy()
    events = []
    start = time.perf_counter()
    counters = Search(own, opposite).run(
        0.15,
        11,
        lambda w, s, label: events.append(dict(weight=w, support=s, stage=label, seconds=time.perf_counter() - start)),
    )
    worker = dict(counters=counters, events=events, search_seconds=time.perf_counter() - start)
    record = dict(method="dispatch", budget_seconds=0.3, workers={"X": [worker]})
    assert audit_worker(record, "X") > 0
    bad = copy.deepcopy(record)
    bad["workers"]["X"][0]["counters"]["route_decision"] = "continue_improving_route"
    with pytest.raises(ValueError, match="Unhelpful"):
        audit_worker(bad, "X")
    bad = copy.deepcopy(record)
    bad["workers"]["X"][0]["counters"]["order"][0] = bad["workers"]["X"][0]["counters"]["order"][1]
    with pytest.raises(ValueError, match="Nonbijective"):
        audit_worker(bad, "X")
