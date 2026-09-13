"""Independent tiny-space oracles for systematic collision joins and enumeration."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_collision_native as native  # noqa: E402


def packed(rows):
    return [sum(int(x) << j for j, x in enumerate(row)) for row in rows]


@pytest.mark.parametrize("stern", [False, True])
@pytest.mark.parametrize("k,n", [(6, 19), (8, 79)])
def test_one_trial_matches_complete_candidate_oracle(k, n, stern):
    rng = np.random.default_rng(183)
    basis = np.concatenate([np.eye(k, dtype=np.int8), rng.integers(0, 2, (k, n - k), dtype=np.int8)], axis=1)
    # Test active-coordinate compaction and a non-64-bit-aligned tail.
    basis[:, k + 1] = 0
    duals = rng.integers(0, 2, (3, n), dtype=np.int8)
    session = native.Session(basis, duals, 31, stern)
    events = session.advance(1)
    view = session.snapshot()
    rows = packed(view["basis"])
    tags = [sum(((word & d).bit_count() % 2) << i for i, d in enumerate(packed(duals))) for word in rows]
    assert tags == view["tags"]
    assert np.array_equal(view["basis"][:, view["pivots"]], np.eye(k))
    combos = [(i,) for i in range(k)] + list(itertools.combinations(range(k), 2))
    collisions = []
    if stern:
        order = view["row_order"]
        for left in itertools.combinations(order[: k // 2], 2):
            for right in itertools.combinations(order[k // 2 :], 2):
                selected = left + right
                value = 0
                for i in selected:
                    value ^= rows[i]
                if all(not (value >> q) & 1 for q in view["projection"]):
                    collisions.append(selected)
        combos += collisions
    eligible = []
    rejected = 0
    for selected in combos:
        value = tag = 0
        for i in selected:
            value ^= rows[i]
            tag ^= tags[i]
        if tag:
            eligible.append(value)
        else:
            rejected += 1
    stats = session.stats
    assert stats["best"] == min(w.bit_count() for w in eligible)
    assert stats["collisions"] == len(collisions)
    assert stats["trivial_rejections"] == rejected
    assert stats["single_scores"] == k and stats["pair_scores"] == k * (k - 1) // 2
    assert stats["exports"] == len(events)
    weights = [w for w, _, _ in events]
    assert weights == sorted(set(weights), reverse=True)
    for weight, support, _ in events:
        word = sum(1 << q for q in support)
        assert word in eligible and weight == len(support)
        assert k + 1 not in support
        original_word = 0
        for i, value in enumerate(packed(basis)):
            if word >> i & 1:
                original_word ^= value
        assert word == original_word


def test_exact_gray_chunks_match_bruteforce_and_filter_stabilizers():
    rng = np.random.default_rng(55)
    k, n = 8, 90
    basis = np.concatenate([np.eye(k, dtype=np.int8), rng.integers(0, 2, (k, n - k), dtype=np.int8)], axis=1)
    duals = np.eye(n, dtype=np.int8)[[0, k - 1]]
    rows = packed(basis)
    words = []
    for mask in range(1, 1 << k):
        value = 0
        for i in range(k):
            if mask >> i & 1:
                value ^= rows[i]
        if mask & 1 or mask >> (k - 1) & 1:
            words.append(value)
    session = native.Session(basis, duals, 0, False)
    events = []
    while not session.stats["enumeration_complete"]:
        events.extend(session.enumerate_chunk(17))
    assert session.stats["enumerated"] == (1 << k) - 1
    assert session.stats["enumerated_nontrivial"] == len(words)
    assert session.stats["best"] == min(w.bit_count() for w in words)
    assert all(sum(1 << q for q in support) in words for _, support, _ in events)
    assert session.enumerate_chunk(17) == []


def test_session_resumes_without_reexporting_incumbents():
    basis = np.eye(8, dtype=np.int8)
    session = native.Session(basis, basis, 77, True)
    first = session.advance(1)
    later = session.advance(3)
    assert first and not later
    assert session.stats["reductions"] == 4
    assert session.stats["best"] == 1


def test_collision_stage_can_improve_beyond_all_pairs():
    rng = np.random.default_rng(183)
    basis = np.concatenate([np.eye(8, dtype=np.int8), rng.integers(0, 2, (8, 71), dtype=np.int8)], axis=1)
    duals = np.eye(79, dtype=np.int8)[:8]
    improvements = []
    for seed in range(32):
        session = native.Session(basis, duals, seed, True)
        improvements += [e for e in session.advance(1) if e[2] == "collision4"]
    assert improvements


@pytest.mark.parametrize("method", ["full-pairs", "block-pairs", "full-stern", "block-stern", "block-exact"])
def test_adapter_exports_valid_synthetic_words(method):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from strategy_prototypes.block_collision.adapter import Search

    empty = np.zeros((0, 8), dtype=np.int8)
    events = []
    result = Search(empty, empty, method).run(0.03, 0, lambda *e: events.append(e))
    assert result["initialization"]["complete"]
    assert all(w == len(s) and len(s) == len(set(s)) and 0 < w <= 8 for w, s, _ in events)
    if method == "block-exact":
        assert all(s["status"] == "exhausted" for s in result["spaces"])


def test_callback_failure_propagates():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from strategy_prototypes.block_collision.adapter import Search

    def fail(*args):
        raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        Search(np.zeros((0, 4), dtype=np.int8), np.zeros((0, 4), dtype=np.int8), "block-stern").run(1, 0, fail)


@pytest.mark.parametrize("bad", ["dependent", "nonbinary", "width", "logical_cap"])
def test_invalid_inputs(bad):
    basis = np.eye(4, dtype=np.int8)
    duals = basis.copy()
    if bad == "dependent":
        basis[1] = basis[0]
    elif bad == "nonbinary":
        basis[0, 0] = 2
    elif bad == "width":
        duals = np.zeros((2, 5), dtype=np.int8)
    else:
        duals = np.zeros((65, 4), dtype=np.int8)
    with pytest.raises(ValueError):
        native.Session(basis, duals, 0)


def test_empty_and_trivial_spaces():
    empty = np.zeros((0, 8), dtype=np.int8)
    session = native.Session(empty, empty, 0)
    assert session.advance(1) == [] and session.enumerate_chunk(2) == []
    assert session.stats["best"] is None and session.stats["enumeration_complete"]
    trivial = native.Session(np.eye(8, dtype=np.int8), empty, 0)
    assert trivial.advance(1) == []
    with pytest.raises(ValueError):
        trivial.advance(0)
