"""Protect evidence retention and deadline scoring in the strategy pilot."""

import json

import pytest
from study_strategies import run_side, summarize


def test_deadline_censoring_keeps_late_witness():
    events = [
        {"seconds": 0.2, "weight": 40},
        {"seconds": 0.5, "weight": 30},
        {"seconds": 0.501, "weight": 20},
    ]
    result = summarize(events, 0.5, 20)
    assert result["best_in_budget"] == 30
    assert result["best_returned"] == 20
    assert not result["target_hit"]
    assert result["late_events"] == 1
    assert summarize(events, 0.5, 30)["time_to_target"] == 0.5
    assert summarize([], 1, 20)["best_in_budget"] is None


def test_partial_evidence_survives_search_failure(tmp_path):
    class Failing:
        def run(self, seconds, seed, emit):
            emit(2, [1, 3])
            raise RuntimeError("prototype failure")

    path = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError, match="prototype failure"):
        run_side(Failing(), 1, 10, path)
    event = json.loads(path.read_text())
    assert event["support"] == [1, 3]
    assert event["weight"] == 2


def test_retains_non_global_improvements_for_class_diversity(tmp_path):
    class Diverse:
        def run(self, seconds, seed, emit):
            emit(2, [1, 3])
            emit(3, [2, 4, 5])
            return {"attempts": 2}

    result = run_side(Diverse(), 1, 10, tmp_path / "events.jsonl")
    assert [e["weight"] for e in result["events"]] == [2, 3]
    assert result["counters"]["attempts"] == 2


def test_report_rejects_missing_run_and_changed_deadline_summary(tmp_path):
    from report_strategies import checked_records

    env = {"cases": ["fixture"], "methods": ["incremental"], "seed_start": 1, "seeds": 1, "seconds_per_code": 2}
    (tmp_path / "environment.json").write_text(json.dumps(env))
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": [{"id": "fixture", "paper_target": 2}]}))
    (tmp_path / "results.json").write_text("[]")
    with pytest.raises(ValueError, match="incomplete"):
        checked_records(tmp_path)
    record = {
        "case": "fixture",
        "method": "incremental",
        "seed": 1,
        "budget_seconds": 2,
        "target": 2,
        "validation_status": "passed",
        "saved_candidates": 0,
        "workers": {s: [{"events": []}] for s in ("X", "Z")},
        "sides": {s: summarize([], 1, 2) for s in ("X", "Z")},
    }
    folder = tmp_path / "fixture" / "incremental-s1"
    folder.mkdir(parents=True)
    for side in ("X", "Z"):
        (folder / f"{side}.jsonl").write_text("")
    (folder / "result.json").write_text(json.dumps(record))
    (tmp_path / "results.json").write_text(json.dumps([record]))
    assert len(checked_records(tmp_path)[1]) == 1
    record["sides"]["X"]["target_hit"] = True
    (folder / "result.json").write_text(json.dumps(record))
    (tmp_path / "results.json").write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="Deadline summary"):
        checked_records(tmp_path)


def test_logical_combination_gray_code_and_word_boundary():
    import numpy as np
    from study_logical_initialization import combinations

    rows = np.zeros((2, 65), dtype=np.uint8)
    rows[0, [0, 1, 64]] = 1
    rows[1, [0, 1]] = 1
    result = combinations(rows)
    assert result["examined"] == 3
    assert result["events"][-1]["support"] == [64]
    assert result["events"][-1]["weight"] == 1


def test_large_logical_basis_uses_bounded_pair_scan():
    import numpy as np
    from study_logical_initialization import combinations

    rows = np.zeros((17, 65), dtype=np.uint8)
    rows[np.arange(17), np.arange(17)] = 1
    rows[:, [63, 64]] = 1
    result = combinations(rows)
    assert result["examined"] == 17 + 17 * 16 // 2
    assert result["events"][-1]["weight"] == 2
    assert result["events"][-1]["support"] == [0, 1]
