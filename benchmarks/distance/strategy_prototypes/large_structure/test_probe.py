"""Synthetic checks for restricted kernels and witness preservation."""

import itertools

import numpy as np
import pytest
from strategy_prototypes.large_structure import adapter


def test_restricted_kernel_matches_exhaustive_assignments():
    rng = np.random.default_rng(61)
    groups = [[0, 3], [1, 4], [2, 5], [6, 7]]
    for _ in range(12):
        h = rng.integers(0, 2, (3, 8), dtype=np.int8)
        space = adapter.native.Space(h, np.eye(8, dtype=np.int8), groups)
        expected = set()
        for bits in itertools.product((0, 1), repeat=len(groups)):
            v = np.zeros(8, dtype=np.int8)
            for bit, group in zip(bits, groups):
                v[group] = bit
            if not (h @ v % 2).any():
                expected.add(tuple(v))
        basis = space.basis
        actual = {
            tuple(np.array(bits, dtype=np.int8) @ basis % 2) for bits in itertools.product((0, 1), repeat=len(basis))
        }
        assert actual == expected
        assert space.logical_rank == space.dimension


def test_native_witness_and_logical_filter():
    h = np.zeros((0, 6), dtype=np.int8)
    space = adapter.native.Space(h, np.eye(6, dtype=np.int8), [[0, 3], [1, 4]])
    weight, support = space.batch(3, 2)
    assert weight == 2 and set(support) in ({0, 3}, {1, 4})
    trivial = adapter.native.Space(h, np.zeros((0, 6), dtype=np.int8), [[0, 3]])
    assert trivial.logical_rank == 0 and trivial.batch(1, 0)[1] == []


@pytest.mark.parametrize("groups", [[[0], [0]], [[6]], [[]]])
def test_invalid_embeddings(groups):
    with pytest.raises(ValueError):
        adapter.native.Space(np.zeros((0, 6), dtype=np.int8), np.eye(6, dtype=np.int8), groups)


@pytest.mark.parametrize(
    "spec,n",
    [
        ({"kind": "product", "group_order": 4}, 20),
        ({"kind": "torus", "length": 35, "twist": 12}, 70),
        ({"kind": "affine", "prime": 19, "order": 18, "action": 2}, 684),
    ],
)
def test_construction_proposals_are_disjoint(spec, n):
    proposals = list(adapter.proposals(spec, n))
    assert proposals
    for _, groups in proposals:
        flat = [q for group in groups for q in group]
        assert len(flat) == len(set(flat)) and min(flat) >= 0 and max(flat) < n


def test_callback_failure_is_fatal():
    rows = np.zeros((0, 20), dtype=np.int8)

    def fail(*args):
        raise OSError("save failed")

    with pytest.raises(OSError, match="save failed"):
        adapter.Search(rows, rows, "structure", {"kind": "product", "group_order": 4}).run(0.05, 0, fail)


def test_real_structural_exports():
    rows = np.zeros((0, 20), dtype=np.int8)
    events = []
    result = adapter.Search(rows, rows, "structure", {"kind": "product", "group_order": 4}).run(
        0.05, 0, lambda *e: events.append(e)
    )
    assert result["active_spaces"] == 4
    structured = [e for e in events if e[2].startswith("seed_blocks_")]
    assert structured and all(w == len(s) == 1 for w, s, _ in structured)
    assert sum(s["exports"] for s in result["spaces"]) == len(structured)
