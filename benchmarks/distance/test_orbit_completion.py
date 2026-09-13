"""Synthetic missing-check recovery and proposal rejection."""

import numpy as np
from orbit_completion import rotate_blocks, search_orbits
from study_strategies import ris_native


def test_rotation_crosses_machine_word_boundary_without_mixing_blocks():
    word = (1 << 64) | (1 << 65) | (1 << 129)
    rotated = rotate_blocks(word, 65)
    assert rotated == (1 << 0) | (1 << 66) | (1 << 65)
    for _ in range(64):
        rotated = rotate_blocks(rotated, 65)
    assert rotated == word


def test_deleted_check_orbit_yields_valid_logicals():
    # Full parent has rows {i,i+4}; remove one. Only child matrices enter search.
    full = np.concatenate([np.eye(4, dtype=np.uint8)] * 2, axis=1)
    own = full[1:]
    opposite = full.copy()
    events = []
    stats = search_orbits(
        own, opposite, ris_native.Prepared(own, opposite).logicals, lambda w, s, stage: events.append((w, s))
    )
    assert events == [(2, [0, 4])]
    assert stats["orbits"] == 1 and stats["proposals"] == 4
    assert stats["logical_witnesses"] == 1


def test_shift_is_not_assumed_to_preserve_syndrome():
    own = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    opposite = np.array([[0, 1, 0, 0]], dtype=np.uint8)
    events = []
    stats = search_orbits(own, opposite, ris_native.Prepared(own, opposite).logicals, lambda *args: events.append(args))
    assert not events
    assert stats["proposals"] == 2 and stats["zero_syndrome"] == 1
    assert stats["logical_witnesses"] == 0
