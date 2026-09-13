"""Synthetic adapter, observer-equivalence and persistence-failure tests."""

import json
import subprocess
import time

import pytest
from run import gf2, np, read_codewords, write_matrix

# isort: split
from common import HERE, matrix_hash
from full_suite import study
from full_suite.adapter import METHODS, Search, m4ri
from full_suite.build_observer import DEST, MARKER
from study_strategies import ris_native, summarize


def steane():
    return np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]], dtype=np.int8)


@pytest.mark.parametrize("method", METHODS)
def test_adapters_valid_toy_events(tmp_path, method):
    h = steane()
    worker = study.run_side(Search(h, h, method, tmp_path / "m4ri"), 0.15, 421, tmp_path / "events.jsonl")
    assert worker["error"] is None
    assert worker["counters"]["initialization"]["complete"]
    assert worker["search_seconds"] < 2
    events = worker["events"]
    assert min(e["weight"] for e in events) == 3
    for e in events:
        v = np.zeros(7, dtype=np.int8)
        v[e["support"]] = 1
        assert int(v.sum()) == e["weight"] and gf2.commutes(v, h) and not gf2.in_rowspace(v, h)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_m4ri_observer_fixed_work_matches_original(tmp_path, seed):
    rng = np.random.default_rng(70)
    opposite = rng.integers(0, 2, (12, 31), dtype=np.int8)
    own = np.zeros((1, 31), dtype=np.int8)
    duals = ris_native.Prepared(own, opposite).logicals
    write_matrix(tmp_path / "H.mtx", opposite)
    write_matrix(tmp_path / "L.mtx", duals)
    records = []
    for name, binary in [("original", HERE / "cache/deps/dist-m4ri/src/dist_m4ri"), ("observer", DEST / "dist_m4ri")]:
        command = [
            str(binary),
            "method=1",
            f"finH={tmp_path / 'H.mtx'}",
            f"finL={tmp_path / 'L.mtx'}",
            "threads=1",
            "timeout=10",
            "steps=500",
            f"seed={seed}",
            "wmin=0",
            f"outC={tmp_path / (name + '.txt')}",
            "debug=0",
            "dW=0",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        records.append((result.stdout, sorted(read_codewords(tmp_path / (name + ".txt")))))
        if name == "observer":
            observations = [
                list(map(int, line.split()[1:])) for line in result.stderr.splitlines() if line.startswith(MARKER)
            ]
            assert observations and [r[0] for r in observations] == sorted({r[0] for r in observations}, reverse=True)
            assert min(r[0] for r in observations) == min(map(len, records[-1][1]))
            for weight, *support in observations:
                v = np.zeros(31, dtype=np.int8)
                v[support] = 1
                assert int(v.sum()) == weight and gf2.commutes(v, opposite) and np.any(duals @ v % 2)
    assert records[0] == records[1]


def test_live_m4ri_receipt_precedes_exit(tmp_path):
    h = steane()
    duals = ris_native.Prepared(h, h).logicals
    start = time.perf_counter()
    events = []
    stats = m4ri(
        h,
        duals,
        tmp_path / "m4ri",
        15,
        start + 0.4,
        lambda w, s, t: events.append(dict(seconds=time.perf_counter() - start, weight=w, support=s)),
    )
    assert events and events[0]["seconds"] < 0.2 and stats["exported"] == len(events)
    assert time.perf_counter() - start > 0.25


def test_balanced_checkpoint_excludes_late_event():
    events = [dict(seconds=14, weight=5, support=list(range(5))), dict(seconds=16, weight=3, support=list(range(3)))]
    assert summarize(events, 15, 3)["best_in_budget"] == 5
    assert summarize(events, 30, 3)["best_in_budget"] == 3


def test_raw_evidence_survives_search_error(tmp_path):
    class Broken:
        def run(self, seconds, seed, emit):
            emit(3, [0, 1, 2], "search")
            raise RuntimeError("synthetic failure")

    path = tmp_path / "raw.jsonl"
    worker = study.run_side(Broken(), 1, 1, path)
    assert "synthetic failure" in worker["error"]
    assert json.loads(path.read_text())["support"] == [0, 1, 2]


def test_save_failure_is_fatal_with_pending_record(tmp_path, monkeypatch):
    h = steane()
    (tmp_path / "matrices").mkdir()
    np.savez(tmp_path / "matrices/toy.npz", hx=h, hz=h)
    case = dict(
        id="toy", file="toy.npz", n=7, k=1, matrix_sha256=matrix_hash(h, h), reference={"X": [0, 1, 2], "Z": [0, 1, 2]}
    )

    def failed_save(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(study, "validate_and_stage", failed_save)
    with pytest.raises(OSError, match="disk failure"):
        study.job(tmp_path, case, "fresh", 0.1, 12)
    path = tmp_path / "runs/toy/fresh/attempt-000/result.json"
    r = json.loads(path.read_text())
    assert r["validation_status"] == "failed" and r["workers"]["X"][0]["events"]
    assert not (path.parents[1] / "completed.json").exists()


def test_preservation_api_receives_every_side_export(tmp_path, monkeypatch):
    h = steane()
    (tmp_path / "matrices").mkdir()
    np.savez(tmp_path / "matrices/toy.npz", hx=h, hz=h)
    case = dict(
        id="toy", file="toy.npz", n=7, k=1, matrix_sha256=matrix_hash(h, h), reference={"X": [0, 1, 2], "Z": [0, 1, 2]}
    )
    calls = []

    def save(hx, hz, c, sides, fallback, run_id):
        calls.append(sides)
        return 0, sum(len({tuple(e["support"]) for w in ws for e in w["events"]}) for ws in sides.values())

    monkeypatch.setattr(study, "validate_and_stage", save)
    r = study.job(tmp_path, case, "fresh", 0.1, 12)
    assert calls and all(calls[0][side][0]["events"] for side in ("X", "Z"))
    assert r["validation_status"] == "passed"
    assert (tmp_path / "runs/toy/fresh/completed.json").is_file()


def test_full_pipeline_and_final_audit(tmp_path, monkeypatch):
    import os
    import tarfile

    import run
    from common import sha256
    from full_suite import report

    # Synthetic-only full pipeline, including the real trusted kit packager.
    # Redirect saved candidate documents into the pytest temporary workspace.
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    h = steane()
    (tmp_path / "matrices").mkdir()
    (tmp_path / "inputs").mkdir()
    np.savez(tmp_path / "matrices/toy.npz", hx=h, hz=h)
    source = tmp_path / "inputs/toy.json"
    source.write_text("{}\n")
    case = dict(
        id="toy",
        file="toy.npz",
        n=7,
        k=1,
        family="synthetic",
        source="codes/toy.json",
        source_sha256=sha256(source),
        matrix_sha256=matrix_hash(h, h),
        paper_target=3,
        reference={"X": [0, 1, 2], "Z": [0, 1, 2]},
    )
    atomic_json = run.atomic_json
    atomic_json(tmp_path / "manifest.json", dict(cases=[case]))
    atomic_json(
        tmp_path / "environment.json",
        dict(
            source_hashes={},
            binary_hashes={},
            methods=list(METHODS),
            seed=12,
            cpus=[next(iter(os.sched_getaffinity(0)))],
            checkpoints=[30, 60],
            seconds_per_code=0.2,
        ),
    )
    for name in ("sources", "binaries"):
        with tarfile.open(tmp_path / (name + ".tar.gz"), "w:gz"):
            pass
    for method in METHODS + ("circulant",):
        study.job(tmp_path, case, method, 2 if method == "circulant" else 0.2, 12)
    report.finalize(tmp_path)
    completed = json.loads((tmp_path / "completed.json").read_text())
    assert completed["configurations"] == 6 and completed["saved_documents"] > 0
    assert json.loads((tmp_path / "audit.json").read_text())["status"] == "passed"
