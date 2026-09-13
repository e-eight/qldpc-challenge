"""Binary polynomial arithmetic and logical-quotient witness checks."""

import numpy as np
from polynomial_completion import divide, gcd, search_quotients
from study_strategies import ris_native


def multiply(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
    return result


def test_division_and_gcd_across_word_boundaries():
    a = (1 << 129) | (1 << 65) | 5
    b = (1 << 64) | 3
    q, r = divide(a, b)
    assert multiply(q, b) ^ r == a
    assert r.bit_length() < b.bit_length()
    f = (1 << 66) | 7
    assert gcd(multiply(f, 3), multiply(f, 7)) == f


def test_shared_factor_produces_nontrivial_two_block_logical():
    own = np.zeros((4, 8), dtype=np.uint8)
    for i in range(4):
        own[i, [i, (i + 1) % 4, 4 + i, 4 + (i + 1) % 4]] = 1
    opposite = own.copy()
    duals = ris_native.Prepared(own, opposite).logicals
    events = []
    stats = search_quotients(own, opposite, duals, lambda w, s, stage: events.append((w, s)))
    assert events and events[-1][0] == 2
    for weight, support in events:
        v = np.zeros(8, dtype=np.uint8)
        v[support] = 1
        assert weight == int(v.sum())
        assert not ((opposite @ v) % 2).any()
        assert ((duals @ v) % 2).any()
    assert stats["logical_witnesses"] == len(events)
