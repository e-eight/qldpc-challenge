"""Full QC kernel via field nullspaces and CRT, with physical coordinates."""

from run import gf2, np

# isort: split
from initialized_search import pack_rows, support_of
from strategy_prototypes.algebra_diagnostic.algebra import bezout, irreducible, poly_div, poly_mul


def remainder(a, f):
    return poly_div(a, f)[1]


def field_kernel(matrix, f):
    a = [[remainder(x, f) for x in row] for row in matrix]
    if not a:
        raise ValueError("Expected nonempty check matrix")
    columns, pivots = len(a[0]), []
    for j in range(columns):
        pivot = next((i for i in range(len(pivots), len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        r = len(pivots)
        a[r], a[pivot] = a[pivot], a[r]
        d, inverse, _ = bezout(a[r][j], f)
        if d != 1:
            raise ValueError("Noninvertible field pivot")
        a[r] = [remainder(poly_mul(x, inverse), f) for x in a[r]]
        for i in range(len(a)):
            if i == r or not a[i][j]:
                continue
            multiple = a[i][j]
            a[i] = [x ^ remainder(poly_mul(multiple, y), f) for x, y in zip(a[i], a[r])]
        pivots.append(j)
    basis = []
    for j in range(columns):
        if j in pivots:
            continue
        v = [0] * columns
        v[j] = 1
        for i, pivot in enumerate(pivots):
            v[pivot] = a[i][j]
        basis.append(v)
    return basis


def reverse(p, length):
    return sum(1 << ((-i) % length) for i in support_of(p))


def check_orbits(h, length):
    n = h.shape[1]
    if n % length:
        raise ValueError("Lift does not divide length")
    mask = (1 << length) - 1
    rows = set(pack_rows(h))
    unused, base = set(rows), []
    while unused:
        word = initial = min(unused)
        orbit = set()
        for _ in range(length):
            orbit.add(word)
            blocks = [(word >> j) & mask for j in range(0, n, length)]
            word = sum((((p << 1) & mask) | (p >> (length - 1))) << (i * length) for i, p in enumerate(blocks))
        if not orbit <= rows:
            raise ValueError("Rows are not complete cyclic orbits")
        unused -= orbit
        base.append([reverse((initial >> j) & mask, length) for j in range(0, n, length)])
    return base


def unpack(words, n):
    out = np.zeros((len(words), n), dtype=np.uint8)
    for i, word in enumerate(words):
        out[i, support_of(word)] = 1
    return out


def packed_with_tags(words, duals, n):
    """Pack physical words followed by every logical detector parity bit."""
    detectors = pack_rows(duals)
    nw, tw = (n + 63) // 64, (len(detectors) + 63) // 64
    out = np.zeros((len(words), nw + tw), dtype=np.uint64)
    mask = (1 << 64) - 1
    for i, word in enumerate(words):
        tag = sum(((word & dual).bit_count() & 1) << j for j, dual in enumerate(detectors))
        for j in range(nw):
            out[i, j] = (word >> (64 * j)) & mask
        for j in range(tw):
            out[i, nw + j] = (tag >> (64 * j)) & mask
    return out


def build(h, length, factors, cap=10):
    if length % 2 == 0 or not 1 <= cap <= 10:
        raise ValueError("Odd squarefree lift and cap in 1..10 required")
    product = 1
    if len(set(factors)) != len(factors):
        raise ValueError("Repeated factor")
    for f in factors:
        if not irreducible(f):
            raise ValueError("Reducible factor")
        product = poly_mul(product, f)
    modulus = (1 << length) | 1
    if product != modulus:
        raise ValueError("Incomplete factorization")
    base = check_orbits(h, length)
    words, widths, groups = [], [], []
    for f in factors:
        degree = f.bit_length() - 1
        quotient, rem = poly_div(modulus, f)
        d, inverse, _ = bezout(quotient, f)
        if rem or d != 1:
            raise RuntimeError("Invalid CRT projector")
        projector = remainder(poly_mul(quotient, inverse), modulus)
        for i, v in enumerate(field_kernel(base, f)):
            lifted = [remainder(poly_mul(projector, value), modulus) for value in v]
            for start in range(0, degree, cap):
                width = min(cap, degree - start)
                widths.append(width)
                groups.append(dict(factor=hex(f), degree=degree, generator=i, first_coefficient=start, width=width))
                for j in range(start, start + width):
                    words.append(sum(remainder(p << j, modulus) << (b * length) for b, p in enumerate(lifted)))
    basis = unpack(words, h.shape[1])
    # Independent binary checks prove completeness even if module conventions err.
    if np.any((basis @ h.T) % 2):
        raise RuntimeError("CRT generator syndrome is nonzero")
    if gf2.rank(basis) != len(words) or len(words) != h.shape[1] - gf2.rank(h):
        raise RuntimeError("CRT generators do not span the complete binary kernel")
    return basis, widths, groups
