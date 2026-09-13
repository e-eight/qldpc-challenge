"""Synthetic tests for competing sessions, delivery and decision auditing."""

import copy
import time
from types import SimpleNamespace

import allocation_search as search
import numpy as np
import pytest
from audit_allocation import audit_worker
from initialized_search import pack_rows
from strategy_prototypes.structure_route_v2.test_adapter import toy


@pytest.mark.parametrize(
    "route,guided,initial,winner",
    [
        (85, 84, 100, "guided"),
        (85, 90, 100, "guided"),
        (81, 90, 100, "route"),
        (28, 78, 96, "route"),
        (None, None, 10, "guided"),
        (10, 20, 10, "guided"),
        (85, None, 90, "guided"),
    ],
)
def test_conservative_preference(route, guided, initial, winner):
    assert search.preferred(route, guided, initial) == winner


def fake_competition(monkeypatch, emit=lambda *e: None, late=False):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(search.time, "perf_counter", lambda: clock.now)
    counts = dict(guided=0, route=0)

    class Engine:
        def __init__(self, name, callback):
            counts[name] += 1
            self.name, self.callback = name, callback
            self.counters = dict(batches=0, best=None, exported=0, applicable=True)
            clock.now += 0.001

        def advance_until(self, deadline):
            self.counters["batches"] += 1
            # Route initially dominates, guided discovers a better answer later.
            weight = 30 if self.name == "route" else (80 if self.counters["batches"] == 1 else 20)
            clock.now = max(clock.now, deadline + (1 if late else 0))
            self.callback(weight, list(range(weight)), "guided" if self.name == "guided" else "routed_single_block")
            self.counters["exported"] += 1
            self.counters["best"] = min(weight, self.counters["best"] or weight)

    monkeypatch.setattr(search, "GuidedSession", lambda a, b, seed, callback: Engine("guided", callback))
    monkeypatch.setattr(search, "RoutedSession", lambda a, b, d, o, seed, callback: Engine("route", callback))
    result = search.compete(None, None, None, [], 0, emit, 100, 0, 0.9)
    return result, counts


def test_policy_switches_without_restarting_sessions(monkeypatch):
    result, counts = fake_competition(monkeypatch)
    assert counts == dict(guided=1, route=1)
    assert result["decisions"][0]["preferred"] == "route"
    assert result["decisions"][-1]["preferred"] == "guided"
    assert all(s["engine"] != d["preferred"] for d, s in zip(result["decisions"], result["slices"][2::2]))
    assert result["timely_best"] == dict(guided=20, route=30)
    assert result["sessions"]["guided"]["batches"] > 1


def test_late_export_preserved_without_policy_credit(monkeypatch):
    events = []
    result, _ = fake_competition(monkeypatch, lambda *e: events.append(e), late=True)
    assert len(events) == 1
    assert result["timely_best"]["guided"] is None
    assert not result["decisions"]


def test_callback_failure_propagates(monkeypatch):
    def fail(*args):
        raise OSError("save failed")

    with pytest.raises(OSError, match="save failed"):
        fake_competition(monkeypatch, fail)


def test_guided_wrapper_preserves_native_session_and_all_improvements(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(search.time, "perf_counter", lambda: clock.now)

    class Native:
        workspace_bytes = 100
        calls = 0

        def advance(self, count):
            self.calls += 1
            clock.now += 0.003
            event = SimpleNamespace(weight=20 - self.calls, support=[self.calls])
            return SimpleNamespace(
                trials=count,
                improvements=[event],
                reductions=0,
                proposals=0,
                exchanges=0,
                accepted_children=0,
                immigrants=0,
            )

    native = Native()
    monkeypatch.setattr(search.GUIDED, "prepare", lambda a, b: SimpleNamespace(session=lambda seed: native))
    events = []
    session = search.GuidedSession(None, None, 1, lambda *e: events.append(e))
    session.advance_until(0.006)
    session.advance_until(0.012)
    assert native.calls == 4 and len(events) == 4
    assert session.counters["best"] == 16
    assert session.counters["scored_bases"] == 4


def test_native_competition_and_corrupt_audit():
    own, opposite = toy()
    events = []
    started = time.perf_counter()
    counters = search.Search(own, opposite).run(
        0.12,
        151,
        lambda w, s, stage: events.append(
            dict(weight=w, support=s, stage=stage, seconds=time.perf_counter() - started)
        ),
    )
    worker = dict(counters=counters, events=events, search_seconds=time.perf_counter() - started)
    record = dict(method="race", budget_seconds=0.24, workers={"X": [worker]})
    assert audit_worker(record, "X") > 0
    assert counters["route_decision"] == "competing_sessions"
    assert counters["decisions"] and all(d["preferred"] == "guided" for d in counters["decisions"])
    checks = pack_rows(opposite)
    duals = pack_rows(search.ris_native.Prepared(own, opposite).logicals)
    for event in events:
        word = sum(1 << q for q in event["support"])
        assert word.bit_count() == event["weight"]
        assert not any((word & row).bit_count() % 2 for row in checks)
        assert any((word & row).bit_count() % 2 for row in duals)
    bad = copy.deepcopy(record)
    bad["workers"]["X"][0]["counters"]["decisions"][0]["preferred"] = "route"
    with pytest.raises(ValueError, match="preferred"):
        audit_worker(bad, "X")
    bad = copy.deepcopy(record)
    bad["workers"]["X"][0]["counters"]["sessions"]["guided"]["exported"] += 1
    with pytest.raises(ValueError, match="export count"):
        audit_worker(bad, "X")


def test_rejected_shape_fallback():
    rows = np.zeros((0, 5), dtype=np.uint8)
    events = []
    result = search.Search(rows, rows).run(0.02, 10, lambda *e: events.append(e))
    assert result["route_decision"] == "no_partition"
    assert "sessions" not in result and result["fallback_seconds"] > 0
    assert min(e[0] for e in events) == 1


@pytest.mark.parametrize("seconds,seed", [(0, 0), (float("nan"), 0), (1, -1), (1, 1 << 64)])
def test_invalid_inputs_before_emission(seconds, seed):
    rows = np.zeros((0, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        search.Search(rows, rows).run(seconds, seed, lambda *e: pytest.fail("unexpected export"))
