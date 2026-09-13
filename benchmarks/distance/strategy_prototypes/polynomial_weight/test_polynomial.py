"""Synthetic exhaustive oracles; never run real-code witness searches here."""

import itertools

import pytest
from run import gf2, np

# isort: split
from strategy_prototypes.algebra_diagnostic.algebra import factor
from strategy_prototypes.polynomial_weight.adapter import native
from strategy_prototypes.polynomial_weight.module import build, check_orbits, field_kernel, packed_with_tags, unpack


def cyclic_checks(base, length):
    words = []
    mask = (1 << length) - 1
    for polynomials in base:
        for j in range(length):
            blocks = [((p << j) & mask) | (p >> (length - j)) for p in polynomials]
            words.append(sum(p << (i * length) for i, p in enumerate(blocks)))
    return unpack(words, length * len(base[0]))


@pytest.mark.parametrize("length,base", [(3, [[3, 5]]), (7, [[11, 13]]), (7, [[1, 2, 4], [2, 4, 1]])])
def test_crt_complete_span(length, base):
    h = cyclic_checks(base, length)
    basis, widths, _ = build(h, length, factor((1 << length) | 1), 2)
    expected = gf2.kernel_basis(h)
    assert sum(widths) == len(basis) == len(expected)
    assert gf2.rank(np.vstack([basis, expected])) == len(expected)
    # Every exhaustive synthetic combination has zero original syndrome.
    for coefficients in itertools.product((0, 1), repeat=len(basis)):
        word = np.array(coefficients, dtype=np.uint8) @ basis % 2
        assert not np.any(h @ word % 2)


def test_reject_bad_structure():
    h = cyclic_checks([[3, 5]], 7)
    with pytest.raises(ValueError, match="factorization"):
        build(h, 7, [3])
    with pytest.raises(ValueError, match="orbits"):
        check_orbits(h[:2], 7)
    with pytest.raises(ValueError, match="Repeated"):
        build(h, 7, [3, 3])
    with pytest.raises(ValueError, match="Reducible"):
        build(h, 7, [5])


def test_field_kernel_oracle():
    matrix = [[1, 2, 3], [2, 3, 1]]
    from strategy_prototypes.polynomial_weight.module import poly_mul, remainder

    basis = field_kernel(matrix, 7)
    assert len(basis) == 2
    for v in basis:
        for row in matrix:
            assert remainder(poly_mul(row[0], v[0]) ^ poly_mul(row[1], v[1]) ^ poly_mul(row[2], v[2]), 7) == 0


@pytest.mark.parametrize("n,tags", [(13, 3), (65, 130), (128, 64)])
def test_exact_update_and_high_logical_tags(n, tags):
    rng = np.random.default_rng(721)
    # Only the highest tag bit is populated: loss of bits >64 would reject all.
    duals = np.zeros((tags, n), dtype=np.uint8)
    duals[-1, 0] = 1
    for _ in range(15):
        basis = rng.integers(0, 2, (5, n), dtype=np.uint8)
        seed = rng.integers(0, 2, (1, n), dtype=np.uint8)
        seed[0, 0] = 1
        from initialized_search import pack_rows

        packed = packed_with_tags(pack_rows(basis), duals, n)
        seeds = packed_with_tags(pack_rows(seed), duals, n)
        session = native.Session(packed, [3, 2], seeds, n, tags, 7)
        for group, subset in [(0, basis[:3]), (1, basis[3:])]:
            oracle = []
            for coefficients in itertools.product((0, 1), repeat=len(subset)):
                word = seed[0] ^ (np.array(coefficients, dtype=np.uint8) @ subset % 2)
                if word[0]:
                    oracle.append(int(word.sum()))
            out = session.probe(seeds[0].tolist(), group)
            physical = sum(int(w) << (64 * j) for j, w in enumerate(out[: (n + 63) // 64]))
            assert physical.bit_count() == min(oracle)
            assert physical & 1


def test_native_inputs_and_padding():
    seeds = np.array([[1, 1]], dtype=np.uint64)
    with pytest.raises(ValueError, match="width"):
        native.Session(seeds, [11], seeds, 13, 1, 1)
    with pytest.raises(ValueError, match="padding"):
        native.Session(np.array([[1 << 13, 1]], dtype=np.uint64), [1], seeds, 13, 1, 1)
    with pytest.raises(ValueError, match="padding"):
        native.Session(np.array([[1, 2]], dtype=np.uint64), [1], seeds, 13, 1, 1)
    with pytest.raises(ValueError, match="Trivial"):
        native.Session(seeds, [1], np.array([[1, 0]], dtype=np.uint64), 13, 1, 1)


def test_restart_exports_membership_and_accounting():
    n, tags = 9, 130
    basis = np.eye(n, dtype=np.uint8)
    duals = np.zeros((tags, n), dtype=np.uint8)
    duals[-1, 8] = 1
    seed = np.ones((1, n), dtype=np.uint8)
    from initialized_search import pack_rows

    packed = packed_with_tags(pack_rows(basis), duals, n)
    seeds = packed_with_tags(pack_rows(seed), duals, n)
    a = native.Session(packed, [3, 3, 3], seeds, n, tags, 891)
    b = native.Session(packed, [3, 3, 3], seeds, n, tags, 891)
    events = a.advance(100)
    assert events == b.advance(100)
    assert [w for w, _ in events] == sorted({w for w, _ in events}, reverse=True)
    assert min(w for w, _ in events) == 1
    assert all(len(s) == w and 8 in s and max(s) < n for w, s in events)
    assert a.stats["scored"] == 700 and a.stats["updates"] == 100
    assert a.stats["exports"] == len(events) and a.stats["restarts"] > 1
