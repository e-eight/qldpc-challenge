"""Check the independent engine against trusted algebra and retained witnesses."""

import json
import sys

import numpy as np
import pytest
from common import HERE, ROOT

sys.path.insert(0, str(ROOT / "native" / "ris"))
sys.path.insert(0, str(HERE / "build"))
import benchmark_native
import gf2
import ris_native
from run import validate_and_stage


def load_case(case_id):
    directory = HERE / "results" / "reference-corpus"
    cases = json.loads((directory / "manifest.json").read_text())["cases"]
    case = next(case for case in cases if case["id"] == case_id)
    with np.load(directory / case["file"]) as data:
        return case, data["hx"], data["hz"]


def preserve(case, hx, hz, batches, tag):
    sides = {}
    for side in ("X", "Z"):
        events = [
            {"weight": event.weight, "support": event.support}
            for batch in batches.get(side, [])
            for event in batch.improvements
        ]
        sides[side] = [{"events": events}]
    validate_and_stage(hx, hz, case, sides, case["reference"], f"native-unit-{tag}")


@pytest.mark.parametrize("case_id", ["board-72-12-6", "tanner-432_8_33", "mitten-975-195"])
@pytest.mark.parametrize("masked,block_size", [(False, 1), (True, 1), (False, 4), (False, 6)])
def test_matches_baseline_trials_and_validates_every_improvement(case_id, masked, block_size):
    case, hx, hz = load_case(case_id)
    batches = {}
    for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
        prepared = ris_native.Prepared(own, opposite)
        baseline = benchmark_native.PreparedSearch(own, opposite)
        session = ris_native.Session(prepared, 1, 123, masked=masked, block_size=block_size)
        batch = session.advance(12)
        batches[side] = [batch]
        expected, support, count = baseline.batch(12, 123, 8, 1, 0)
        assert batch.trials == count == 12
        assert batch.best_weight == expected
        assert batch.improvements[-1].support == support
        assert np.array_equal(prepared.kernel, gf2.kernel_basis(opposite))
        assert prepared.logicals.shape[0] == case["k"]
        assert not ((prepared.logicals @ own.T) % 2).any()
        assert gf2.rank(np.vstack((opposite, prepared.logicals))) == gf2.rank(opposite) + case["k"]
    preserve(case, hx, hz, batches, f"baseline-{case_id}-{masked}-{block_size}")


@pytest.mark.parametrize("threads", [1, 4])
def test_batches_preserve_stream_and_lifetime(threads):
    case, hx, hz = load_case("board-72-12-6")
    prepared = ris_native.Prepared(hx, hz)
    one = ris_native.Session(prepared, threads, 321)
    split = ris_native.Session(prepared, threads, 321)
    del prepared
    entire = one.advance(80)
    pieces = [split.advance(20) for _ in range(4)]

    def key(event):
        return event.worker, event.trial, event.weight, event.support

    assert sorted(map(key, entire.improvements)) == sorted(key(e) for b in pieces for e in b.improvements)
    assert entire.best_weight == pieces[-1].best_weight
    assert entire.trials == sum(b.trials for b in pieces)
    preserve(case, hx, hz, {"X": [entire, *pieces]}, f"batch-{threads}")


@pytest.mark.parametrize("n", [1, 63, 64, 65, 127, 128, 129])
def test_word_boundaries_and_asymmetric_repetition_code(n):
    hx = np.zeros((0, n), dtype=np.uint8)
    hz = np.zeros((n - 1, n), dtype=np.uint8)
    for row in range(n - 1):
        hz[row, row : row + 2] = 1
    case = {
        "id": f"repetition-{n}",
        "n": n,
        "matrix_sha256": "unit-fixture",
        "reference": {"X": list(range(n)), "Z": [0]},
    }
    batches = {}
    for side, own, opposite, weight in (("X", hx, hz, n), ("Z", hz, hx, 1)):
        p = ris_native.Prepared(own, opposite)
        result = ris_native.Session(p, 4, 11).advance(3)
        assert result.trials == 3
        assert result.best_weight == weight
        batches[side] = [result]
    preserve(case, hx, hz, batches, f"boundary-{n}")


def test_invalid_input_and_no_logicals():
    z = np.zeros((1, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="binary"):
        ris_native.Prepared(z + 2, z)
    with pytest.raises(ValueError, match="dtype"):
        ris_native.Prepared(z.astype(float), z)
    with pytest.raises(ValueError, match="widths"):
        ris_native.Prepared(z, z[:, :2])
    with pytest.raises(ValueError, match="commute"):
        ris_native.Prepared(np.ones((1, 3), dtype=np.int8), np.ones((1, 3), dtype=np.int8))
    with pytest.raises(ValueError):
        ris_native.Prepared(z[0], z)
    with pytest.raises(ValueError):
        ris_native.Prepared(z[:, :0], z[:, :0])
    p = ris_native.Prepared(np.eye(3, dtype=np.uint8), z)
    assert not p.applicable
    assert ris_native.Session(p).advance(10).trials == 0
    with pytest.raises(ValueError):
        ris_native.Session(p, threads=0)
    with pytest.raises(ValueError):
        ris_native.Session(p, pair_depth=-1)


def test_strided_inputs_are_copied_and_owned():
    _, hx, hz = load_case("board-72-12-6")
    p = ris_native.Prepared(hx[::-1, ::-1], hz[::-1, ::-1])
    expected = gf2.kernel_basis(hz[:, ::-1])
    hx.fill(0)
    hz.fill(0)
    assert np.array_equal(gf2.rref(p.kernel)[0], gf2.rref(expected)[0])


@pytest.mark.parametrize("seed", range(8))
def test_preparation_on_rank_deficient_css_matrices(seed):
    rng = np.random.default_rng(seed)
    own = rng.integers(0, 2, (9, 65), dtype=np.uint8)
    own = np.vstack((own, own[0], own[0] ^ own[1]))
    basis = gf2.kernel_basis(own)
    opposite = (rng.integers(0, 2, (13, len(basis)), dtype=np.uint8) @ basis) % 2
    p = ris_native.Prepared(own.astype(bool), opposite.astype(bool))
    assert np.array_equal(p.kernel, gf2.kernel_basis(opposite))
    assert gf2.rank(np.vstack((opposite, p.logicals))) == 65 - gf2.rank(own)
    assert len(p.logicals) == 65 - gf2.rank(own) - gf2.rank(opposite)


@pytest.mark.parametrize("failure", ["missing", "unvalidated", "support"])
def test_fixed_work_report_rejects_incomplete_or_changed_results(tmp_path, failure):
    from report_native import main

    source = tmp_path / "run"
    (source / "fixture").mkdir(parents=True)
    environment = {"methods": ["cpp", "ris-block6"], "seed_start": 100, "repeats": 1}
    (source / "environment.json").write_text(json.dumps(environment))
    (source / "corpus.json").write_text(json.dumps({"cases": [{"id": "fixture"}]}))
    for method in environment["methods"]:
        if failure == "missing" and method != "cpp":
            continue
        side = {"trials": 10, "seconds": 1, "best_weight": 1, "events": [{"weight": 1, "support": [0]}]}
        if failure == "support" and method != "cpp":
            side["events"][0]["support"] = [1]
        record = {
            "case": "fixture",
            "method": method,
            "seed": 100,
            "validation_status": "pending" if failure == "unvalidated" else "passed",
            "sides": {"X": side, "Z": side},
        }
        (source / "fixture" / f"{method}.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incomplete|mismatch"):
        main(source, tmp_path / "report")


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("interval", [1, 7, 64])
def test_incremental_continuity_counters_and_witnesses(threads, interval):
    case, hx, hz = load_case("board-72-12-6")
    prepared = ris_native.Prepared(hx, hz)
    options = dict(threads=threads, seed=784, block_size=6, restart_interval=interval, exchange_proposals=8)
    one, split = ris_native.Session(prepared, **options), ris_native.Session(prepared, **options)
    entire = one.advance(320)
    pieces = [split.advance(80) for _ in range(4)]

    def key(e):
        return e.worker, e.trial, e.weight, e.support

    assert sorted(map(key, entire.improvements)) == sorted(key(e) for b in pieces for e in b.improvements)
    expected_reductions = threads * ((320 // threads + interval - 1) // interval)
    assert entire.reductions == expected_reductions
    assert entire.proposals == (320 - expected_reductions) * 8
    assert 0 <= entire.exchanges <= entire.proposals
    empty = split.advance(0)
    for name in ("reductions", "proposals", "exchanges", "best_weight"):
        assert getattr(empty, name) == getattr(entire, name)
        assert getattr(pieces[-1], name) == getattr(entire, name)
    preserve(case, hx, hz, {"X": [entire, *pieces]}, f"incremental-{threads}-{interval}")


def test_incremental_options():
    _, hx, hz = load_case("board-72-12-6")
    prepared = ris_native.Prepared(hx, hz)
    with pytest.raises(ValueError):
        ris_native.Session(prepared, exchange_proposals=0)
    with pytest.raises(TypeError):
        ris_native.Session(prepared, restart_interval=-1)


def test_adjacent_study_seeds_do_not_reuse_cpp_batch_streams():
    from run import native_batch_seed

    # The previous seed + batch * 1000003 mapping made these identical.
    assert native_batch_seed(300 * 1000003, 1) != native_batch_seed(301 * 1000003, 0)
    seen = set()
    for seed in range(300, 310):
        for side in (0, 499979):
            streams = {native_batch_seed(seed * 1000003 + side, batch) for batch in range(256)}
            assert len(streams) == 256
            assert not (streams & seen)
            seen.update(streams)


def test_parallel_save_failure_is_a_hard_error(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import run

    case, hx, hz = load_case("board-72-12-6")
    support = case["reference"]["X"]
    sides = {"X": [{"events": [{"weight": len(support), "support": support}]}], "Z": []}

    def fail_save(*args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(run, "save_submission", fail_save)
    with ThreadPoolExecutor(2) as executor, pytest.raises(OSError, match="storage failure"):
        run.validate_and_stage(hx, hz, case, sides, case["reference"], "test-failed-save", executor)
