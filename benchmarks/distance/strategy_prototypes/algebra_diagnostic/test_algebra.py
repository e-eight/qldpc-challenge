import itertools

import numpy as np
import pytest
from run import gf2
from strategy_prototypes.algebra_diagnostic.algebra import (
    balanced_profiles,
    bezout,
    components,
    coupling_profile,
    decomposition,
    factor,
    irreducible,
    logical_rank,
    orbit_rows,
    packed_rows,
    poly_div,
    poly_mul,
    polynomial_model,
    preserves_rowspace,
    row_mix,
)


def test_decomposition_equals_exhaustive_mask_equations():
    rng = np.random.default_rng(80)
    for _ in range(20):
        h = rng.integers(0, 2, (4, 7), dtype=np.int8)
        g = gf2.kernel_basis(h)
        groups, _, _ = decomposition(h)
        accepted = []
        for bits in itertools.product(range(2), repeat=7):
            mask = np.array(bits, dtype=np.int8)
            if not np.any(h @ (g * mask).T % 2):
                accepted.append(mask)
        signatures = list(zip(*accepted))
        expected = {frozenset(j for j, s in enumerate(signatures) if s == signature) for signature in signatures}
        assert {frozenset(x["coordinates"]) for x in groups} == expected
        assert sum(x["dimension"] for x in groups) == 7 - gf2.rank(h)


def test_row_mixing_hides_graph_but_not_algebraic_components():
    h = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]], dtype=np.int8)
    mixed = h.copy()
    mixed[0] ^= mixed[1]
    assert len(components(mixed)) == 2
    assert decomposition(mixed)[0] == decomposition(h)[0]
    p = np.array([4, 2, 0, 3, 1])
    actual = {frozenset(p[x["coordinates"]]) for x in decomposition(row_mix(h, 19)[:, p])[0]}
    assert actual == {frozenset(x["coordinates"]) for x in decomposition(h)[0]}


def test_profile_matches_syndrome_intersections():
    rng = np.random.default_rng(90)
    h = rng.integers(0, 2, (4, 8), dtype=np.int8)
    order = rng.permutation(8).tolist()
    profile = coupling_profile(h, order)

    def span(a):
        values = {0}
        for column in packed_rows(a.T):
            values |= {v ^ column for v in list(values)}
        return values

    for i in range(9):
        assert len(span(h[:, order[:i]]) & span(h[:, order[i:]])) == 2 ** profile[i]
    for p in balanced_profiles(h):
        assert 2 <= p["cut"] <= 6


def test_logical_rank_matches_quotient_dimensions():
    own = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)
    opposite = np.array([[1, 1, 1, 1]], dtype=np.int8)
    for mask in itertools.product(range(2), repeat=4):
        s = [q for q, b in enumerate(mask) if b]
        small = gf2.kernel_basis(opposite[:, s])
        lifted = np.zeros((len(small), 4), dtype=np.int8)
        lifted[:, s] = small
        expected = gf2.rank(np.vstack([own, lifted])) - gf2.rank(own)
        assert logical_rank(own, opposite, s) == expected


def test_factorization_against_exhaustive_trial_division():
    for polynomial in range(2, 256):
        pieces = factor(polynomial)
        product = 1
        for piece in pieces:
            product = poly_mul(product, piece)
            assert irreducible(piece)
            for divisor in range(2, 1 << ((piece.bit_length() - 1) // 2 + 1)):
                assert poly_div(piece, divisor)[1] != 0
        assert product == polynomial


def test_bezout_and_repeated_factors():
    for a, b in [(0, 7), (15, 0), (25, 37), (91, 109)]:
        d, u, v = bezout(a, b)
        assert poly_mul(a, u) ^ poly_mul(b, v) == d
    assert factor((1 << 12) | 1) == [3] * 4 + [7] * 4


def test_verified_circulant_model_and_rejections():
    length = 7
    a = 3
    b = 9

    def reverse(p):
        return sum(1 << ((-i) % length) for i in range(length) if p >> i & 1)

    hx = orbit_rows(a | (b << length), length)
    hz = orbit_rows(reverse(b) | (reverse(a) << length), length)
    model = polynomial_model(hx, hz)
    assert model["status"] == "verified" and model["quantum_dimension"] == 2
    shift = list(range(1, length)) + [0] + list(range(length + 1, 2 * length)) + [length]
    assert preserves_rowspace(row_mix(hx, 21), shift)
    assert polynomial_model(hx[::-1], hz[::-1])["status"] == "verified"
    damaged = hz.copy()
    damaged[0, 0] ^= 1
    assert polynomial_model(hx, damaged)["status"] == "Z_reciprocal_mismatch"
    assert not preserves_rowspace(np.eye(3, 7, dtype=np.int8), list(range(1, 7)) + [0])


@pytest.mark.parametrize("bad", [[0, 0], [1, 0, 2]])
def test_bad_order(bad):
    with pytest.raises(ValueError):
        coupling_profile(np.zeros((0, 2), dtype=np.int8), bad)
