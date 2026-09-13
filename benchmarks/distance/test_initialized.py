"""Invariant and evidence tests for the initialized search policy."""

import json
import time

import numpy as np
import pytest
from initialized_search import LogicalPool, Search, initialize, refinement_allowance, support_of, word_of
from study_initialized import run_side
from study_strategies import ris_native


def test_refinement_budget_is_earned_and_respects_deadline():
    assert refinement_allowance(0.049, 0, 1) == 0
    assert refinement_allowance(0.060, 0, 1) == 0.005
    assert refinement_allowance(0.060, 0.0055, 1) == 0
    assert refinement_allowance(1, 0, 0.0049) == 0
    assert refinement_allowance(1, 0.099, 1) == 0


def test_pool_tracks_classes_not_just_supports():
    # X stabilizer {0,1}; a Z logical overlaps both, so adding the stabilizer
    # preserves the signature. The last qubit is a second independent class.
    duals = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    pool = LogicalPool(duals, 2)
    pool.offer(word_of([0, 2]))
    pool.offer(word_of([1]))
    assert pool.signature(word_of([0])) == pool.signature(word_of([1]))
    pool.offer(word_of([0]))
    assert len(pool.entries) == 2
    assert support_of(pool.take()) == [0]
    assert support_of(pool.take()) == [0, 2]
    assert pool.take() is None
    with pytest.raises(ValueError, match="stabilizer"):
        pool.offer(word_of([0, 1]))


@pytest.mark.parametrize("n", [7, 65, 129])
def test_initialization_matches_exhaustive_representative_span(n):
    own = np.zeros((1, n), dtype=np.uint8)
    own[0, [0, 1]] = 1
    opposite = np.zeros((n - 3, n), dtype=np.uint8)
    for i in range(n - 3):
        opposite[i, i + 2] = 1
    prepared = ris_native.Prepared(own, opposite)
    rows = ris_native.Prepared(opposite, own).logicals
    assert rows.shape == (2, n)
    pool = LogicalPool(prepared.logicals)
    events = []

    def observe(word, stage, **kwargs):
        events.append(word)

    stats = initialize(rows, pool, observe, time.perf_counter() + 1)
    assert stats["complete"] and stats["examined"] == 3
    assert min(w.bit_count() for w in events) == 1
    for word in events:
        vector = np.zeros(n, dtype=np.uint8)
        vector[support_of(word)] = 1
        assert not ((opposite @ vector) % 2).any()
        assert pool.signature(word)


def test_initialization_checks_deadline_before_scanning():
    pool = LogicalPool(np.eye(2, dtype=np.uint8))
    events = []
    stats = initialize(np.eye(2, dtype=np.uint8), pool, lambda *a, **k: events.append(a), 0)
    assert not stats["complete"] and not events


@pytest.mark.parametrize("method", ["incremental", "guided", "guided-refine"])
def test_end_to_end_toy_search_and_stage_retention(method, tmp_path):
    own = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.uint8)
    opposite = np.array([[1, 1, 1, 1, 0]], dtype=np.uint8)
    path = tmp_path / "events.jsonl"
    result = run_side(Search(own, opposite, method), 0.15, 99, path)
    assert result["events"] == [json.loads(line) for line in path.read_text().splitlines()]
    assert result["counters"]["initial_best"] == 1
    assert result["counters"]["best_weight"] == 1
    assert result["counters"]["scored_bases"] > 0
    if method == "guided-refine":
        assert result["counters"]["descent_calls"] > 0
    pool = LogicalPool(ris_native.Prepared(own, opposite).logicals)
    for event in result["events"]:
        vector = np.zeros(5, dtype=np.uint8)
        vector[event["support"]] = 1
        assert int(vector.sum()) == event["weight"]
        assert not ((opposite @ vector) % 2).any()
        assert pool.signature(word_of(event["support"]))


def test_empty_logical_space_and_zero_budget():
    own = np.eye(3, dtype=np.uint8)
    opposite = np.zeros((0, 3), dtype=np.uint8)
    events = []
    engine = Search(own, opposite, "guided-refine")
    assert engine.run(0, 0, lambda *args: events.append(args))["status"] == "zero_budget"
    assert engine.run(0.1, 0, lambda *args: events.append(args))["status"] == "not_applicable"
    assert not events
