"""Toy-only correctness tests; corpus evaluation belongs to the persisted harness."""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location("reduced_adapter", HERE / "adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
from study_strategies import ris_native  # noqa: E402


def multiply(a, b):
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def bicycle(length, a, b):
    def circulant(poly):
        return np.array(
            [[(poly >> ((j - i) % length)) & 1 for j in range(length)] for i in range(length)], dtype=np.uint8
        )

    aa, bb = circulant(a), circulant(b)
    return np.hstack((aa, bb)), np.hstack((bb.T, aa.T))


def check(events, own, opposite, duals):
    for weight, support, _ in events:
        vector = np.zeros(own.shape[1], dtype=np.uint8)
        vector[support] = 1
        assert vector.sum() == weight
        assert not ((opposite @ vector) & 1).any()
        assert ((duals @ vector) & 1).any()


def test_both_block_reduction_searches_combinations():
    own, opposite = bicycle(9, multiply(3, 124), multiply(3, 121))
    duals = ris_native.Prepared(own, opposite).logicals
    spaces, stats = adapter.polynomial_generators(own, opposite, time.perf_counter() + 10)
    assert stats["common_factor_degree"] == 1
    assert "reduced_both" in spaces
    original_weight = (124).bit_count() + (121).bit_count()
    events = []
    result = adapter.search(own, opposite, duals, 0.1, 101, lambda w, s, stage: events.append((w, s, stage)))
    check(events, own, opposite, duals)
    both = [w for w, _, stage in events if stage == "reduced_both"]
    assert both and min(both) < original_weight
    assert result["branches"]["reduced_both"]["trials"] > 0
    assert result["workspace_bytes"] < 1_000_000


def test_tags_and_cancellation_across_word_boundaries():
    n = 130
    words = [sum(1 << j for j in support) for support in ([0, 64, 65, 129], [1, 64, 65, 129])]
    duals = np.zeros((67, n), dtype=np.uint8)
    duals[66, 0] = 1
    session = adapter.reduced_space_native.Session(adapter.packed_with_tags(words, duals, n), n, 67, 55)
    events = session.advance()["events"]
    assert min(w for w, _ in events) == 2
    for weight, support in events:
        vector = np.zeros(n, dtype=np.uint8)
        vector[support] = 1
        assert vector.sum() == weight
        assert ((duals @ vector) & 1).any()
        assert sum(1 << j for j in support) in (words[0], words[0] ^ words[1])


def test_fixed_trial_reproducibility():
    own, opposite = bicycle(9, multiply(3, 124), multiply(3, 121))
    duals = ris_native.Prepared(own, opposite).logicals
    spaces, _ = adapter.polynomial_generators(own, opposite, time.perf_counter() + 10)
    packed = adapter.packed_with_tags(spaces["reduced_both"], duals, 18)
    a = adapter.reduced_space_native.Session(packed, 18, len(duals), 66)
    b = adapter.reduced_space_native.Session(packed, 18, len(duals), 66)
    for _ in range(4):
        assert a.advance() == b.advance()


def test_invalid_layout_never_exports_invalid_vectors():
    own, opposite = bicycle(9, multiply(3, 124), multiply(3, 121))
    perm = np.random.default_rng(93).permutation(18)
    own, opposite = own[:, perm], opposite[:, perm]
    duals = ris_native.Prepared(own, opposite).logicals
    events = []
    adapter.search(own, opposite, duals, 0.03, 102, lambda w, s, stage: events.append((w, s, stage)))
    check(events, own, opposite, duals)


def test_zero_budget_and_dimension_guard():
    own, opposite = bicycle(4, 3, 3)
    duals = ris_native.Prepared(own, opposite).logicals
    events = []
    result = adapter.search(own, opposite, duals, 0, 1, lambda *args: events.append(args))
    assert not events
    assert not result["branches"]
    huge = np.zeros((1, 4096), dtype=np.uint8)
    result = adapter.search(huge, huge, huge, 0.1, 1, lambda *args: events.append(args))
    assert result["status"] == "dimension_limit_or_no_logicals"


def test_native_rejects_overlarge_buffers():
    with pytest.raises(ValueError, match="dimensions"):
        adapter.reduced_space_native.Session(np.zeros((1, 66), dtype=np.uint64), 4096, 65, 1)
