"""Synthetic checks only: corpus witnesses belong to the parent saving harness."""

import numpy as np
from strategy_prototypes.matrix_structure_v2.adapter import (
    cycle_lengths,
    pack_rows,
    recover_translation,
    search,
    sector_maps,
    transform,
)
from study_strategies import ris_native


def circulant(length, support):
    first = np.zeros(length, dtype=np.uint8)
    first[support] = 1
    return np.asarray([np.roll(first, i) for i in range(length)])


def child():
    a = circulant(31, [0, 1, 4, 8, 13])
    full = np.concatenate([a, a], axis=1)
    return np.delete(full, [3, 12, 22], axis=0), full


def test_translation_completes_deleted_checks_after_qubit_relabeling():
    own, opposite = child()
    permutation = np.random.default_rng(812).permutation(own.shape[1])
    own, opposite = own[:, permutation], opposite[:, permutation]
    events = []
    duals = ris_native.Prepared(own, opposite).logicals
    result = search(own, opposite, duals, 2.0, 19, lambda *event: events.append(event))
    recovered = [event for event in events if event[2].startswith("orbit_") and "layout" not in event[2]]
    assert result["complete"] and recovered
    for weight, indices, _ in recovered:
        word = np.zeros(own.shape[1], dtype=np.uint8)
        word[indices] = 1
        assert weight == 10
        assert not np.any(opposite @ word % 2)
        assert np.any(duals @ word % 2)


def test_recovery_exact_on_distinct_column_signatures_and_equivariant():
    a, b = circulant(31, [0, 1, 4]), circulant(31, [0, 2, 8, 13, 17])
    rows = np.concatenate([a, b], axis=1)
    expected = np.r_[np.roll(np.arange(31), -1), np.roll(np.arange(31, 62), -1)]
    assert np.array_equal(recover_translation(rows), expected)
    relabel = np.random.default_rng(44).permutation(62)
    hidden = recover_translation(rows[:, relabel])
    assert np.array_equal(relabel[hidden], expected[relabel])
    assert cycle_lengths(hidden) == [31, 31]


def test_sector_exchange_verified_after_relabeling():
    a, b = circulant(15, [0, 1]), circulant(15, [0, 3])
    own, opposite = np.concatenate([a, b], axis=1), np.concatenate([b.T, a.T], axis=1)
    relabel = np.random.default_rng(3).permutation(30)
    own, opposite = own[:, relabel], opposite[:, relabel]
    maps = sector_maps(own, opposite)
    assert maps
    for permutation in maps:
        assert {transform(row, permutation) for row in pack_rows(own)} == set(pack_rows(opposite))
        assert {transform(row, permutation) for row in pack_rows(opposite)} == set(pack_rows(own))
    events = []
    duals = ris_native.Prepared(own, opposite).logicals
    result = search(own, opposite, duals, 2, 0, lambda *event: events.append(event))
    assert result["sector_maps"] > 0
    assert any(stage == "sector_transfer" for _, _, stage in events)
    for weight, indices, _ in events:
        word = np.zeros(30, dtype=np.uint8)
        word[indices] = 1
        assert weight == len(indices)
        assert not np.any(opposite @ word % 2)
        assert np.any(duals @ word % 2)


def test_zero_budget_emits_nothing():
    own, opposite = child()
    events = []
    result = search(own, opposite, np.zeros((0, 62), dtype=np.uint8), 0, 0, lambda *e: events.append(e))
    assert result["status"] == "deadline" and not events


def test_false_translation_never_assumed_to_commute():
    own = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    opposite = np.array([[0, 1, 0, 0]], dtype=np.uint8)
    events = []
    result = search(own, opposite, ris_native.Prepared(own, opposite).logicals, 1, 0, lambda *e: events.append(e))
    assert result["complete"] and not events


def test_row_shuffle_is_an_explicit_limitation():
    a, b = circulant(31, [0, 1, 4]), circulant(31, [0, 2, 8, 13, 17])
    rows = np.concatenate([a, b], axis=1)
    expected = recover_translation(rows)
    shuffled = rows[np.random.default_rng(75).permutation(31)]
    assert not np.array_equal(recover_translation(shuffled), expected)


def test_combined_qubit_and_independent_row_shuffle_control():
    a, b = circulant(31, [0, 1, 4]), circulant(31, [0, 2, 8, 13, 17])
    own, opposite = np.concatenate([a, b], axis=1), np.concatenate([b.T, a.T], axis=1)
    rng = np.random.default_rng(917)
    relabel = rng.permutation(62)
    expected = recover_translation(own[:, relabel])
    own = own[rng.permutation(31)][:, relabel]
    opposite = opposite[rng.permutation(31)][:, relabel]
    assert not np.array_equal(recover_translation(own), expected)
    assert not np.array_equal(recover_translation(opposite), expected)
    # Exact exchange exists abstractly, but this bounded detector cannot recover
    # its unknown row correspondence after independent row shuffles.
    assert sector_maps(own, opposite) == []
