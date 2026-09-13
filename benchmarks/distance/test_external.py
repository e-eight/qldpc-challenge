"""Protect external observation semantics, initialization and deadline credit."""

import json
import subprocess
import time

import numpy as np
import pytest
import run  # noqa: F401 -- establish numerical thread limits
from external_search import QD_PARAMS, Search, m4ri_search, qdist_module, qdist_search
from study_initialized import run_side
from study_strategies import ris_native, summarize


def toy():
    own = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.uint8)
    opposite = np.array([[1, 1, 1, 1, 0]], dtype=np.uint8)
    return own, opposite


def test_qdist_observation_preserves_upstream_fixed_work_result():
    own, opposite = toy()
    duals = ris_native.Prepared(own, opposite).logicals
    module = qdist_module()
    original = module.permMinRowsK
    params = dict(QD_PARAMS, iterCount=200, genCount=2)
    expected = module.QDistEvol(opposite, duals, tB=1, params=params.copy(), seed=71)
    events = []
    result = qdist_search(opposite, duals, 71, float("inf"), lambda w, s, stage: events.append((w, s)), params)
    assert module.permMinRowsK is original
    assert result["scored_bases"] == 200
    assert result["engine_best"] == expected[0] == min(w for w, _ in events)
    for weight, support in events:
        word = np.zeros(5, dtype=np.uint8)
        word[support] = 1
        assert weight == int(word.sum())
        assert not ((opposite @ word) % 2).any()
        assert ((duals @ word) % 2).any()


def test_qdist_deadline_keeps_final_witness_and_restores_hook():
    own, opposite = toy()
    module = qdist_module()
    original = module.permMinRowsK
    events = []
    result = qdist_search(
        opposite, ris_native.Prepared(own, opposite).logicals, 2, 0, lambda *args: events.append(args)
    )
    assert result["status"] == "deadline" and result["scored_bases"] == 1
    assert events and module.permMinRowsK is original


def test_m4ri_exports_all_supports_and_does_not_receive_a_target(tmp_path, monkeypatch):
    directory = tmp_path / "m4ri"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        (directory / "codewords.txt").write_text("2 1 3\n2 2 4\n")
        return subprocess.CompletedProcess(command, 0, stdout=b"0 2 17\n")

    monkeypatch.setattr("external_search.subprocess.run", fake_run)
    events = []
    result = m4ri_search(
        np.zeros((1, 4), dtype=np.uint8),
        np.eye(4, dtype=np.uint8),
        directory,
        12,
        time.perf_counter() + 1,
        lambda *args: events.append(args),
    )
    assert result["scored_bases"] == 17
    assert events == [(2, [0, 2], "search"), (2, [1, 3], "search")]
    assert "wmin=0" in calls[0] and "threads=1" in calls[0]
    assert json.loads((directory / "command.json").read_text()) == calls[0]


@pytest.mark.parametrize("method", ["incremental", "guided", "m4ri", "qdistevol"])
def test_common_initialization_and_valid_toy_witnesses(method, tmp_path):
    own, opposite = toy()
    engine = Search(own, opposite, method, tmp_path / "backend")
    result = run_side(engine, 0.15, 12, tmp_path / "events.jsonl")
    assert result["counters"]["initial_best"] == 1
    assert result["counters"]["initialization"]["complete"]
    assert summarize(result["events"], 0.15, 1)["best_in_budget"] == 1
    duals = ris_native.Prepared(own, opposite).logicals
    for event in result["events"]:
        word = np.zeros(5, dtype=np.uint8)
        word[event["support"]] = 1
        assert int(word.sum()) == event["weight"]
        assert not ((opposite @ word) % 2).any()
        assert ((duals @ word) % 2).any()


def test_late_external_bound_does_not_override_timely_initialization():
    events = [
        {"seconds": 0.01, "weight": 8, "stage": "initialization"},
        {"seconds": 5.01, "weight": 4, "stage": "search"},
    ]
    result = summarize(events, 5, 4)
    assert result["best_in_budget"] == 8 and result["best_returned"] == 4
    assert not result["target_hit"] and result["late_events"] == 1
