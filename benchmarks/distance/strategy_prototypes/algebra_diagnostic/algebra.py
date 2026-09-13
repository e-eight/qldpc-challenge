"""Exact binary structure diagnostics; no distance or witness search."""

from collections import deque

from run import gf2, np


def packed_rows(matrix):
    return [int.from_bytes(row.tobytes(), "little") for row in np.packbits(matrix, axis=1, bitorder="little")]


def components(matrix):
    """Find coordinate groups joined by the supplied rows."""
    n = matrix.shape[1]
    parent = list(range(n))

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for row in matrix:
        support = np.flatnonzero(row)
        for q in support[1:]:
            parent[find(int(q))] = find(int(support[0]))
    groups = {}
    for q in range(n):
        groups.setdefault(find(q), []).append(q)
    return sorted(groups.values(), key=lambda g: g[0])


def decomposition(h):
    """Return the finest kernel support decomposition, including fixed-zero columns."""
    reduced, pivots = gf2.rref(h)
    groups = components(reduced)
    pivotset = set(pivots)
    result = [dict(coordinates=g, dimension=len(g) - sum(q in pivotset for q in g)) for g in groups]
    return result, reduced, pivots


def logical_rank(own, opposite, coordinates, own_rank=None):
    n = own.shape[1]
    s = sorted(coordinates)
    t = sorted(set(range(n)) - set(s))
    r = gf2.rank(own) if own_rank is None else own_rank
    return len(s) - gf2.rank(opposite[:, s]) - r + gf2.rank(own[:, t])


def rank_prefixes(columns):
    basis = {}
    ranks = [0]
    for column in columns:
        value = column
        while value:
            pivot = value.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = value
                break
            value ^= basis[pivot]
        ranks.append(len(basis))
    return ranks


def coupling_profile(h, order):
    if sorted(order) != list(range(h.shape[1])):
        raise ValueError("Not a coordinate order")
    columns = packed_rows(h.T)
    sequence = [columns[q] for q in order]
    left = rank_prefixes(sequence)
    right = rank_prefixes(sequence[::-1])[::-1]
    return [a + b - left[-1] for a, b in zip(left, right)]


def candidate_orders(h):
    n = h.shape[1]
    reduced, _ = gf2.rref(h)
    adjacency = [set() for _ in range(n)]
    for row in reduced:
        support = np.flatnonzero(row).tolist()
        if not support:
            continue
        pivot = support[0]
        for q in support[1:]:
            adjacency[pivot].add(q)
            adjacency[q].add(pivot)
    yield "natural", list(range(n))
    yield "reverse", list(reversed(range(n)))
    for start in (0, n // 2, n - 1):
        order = []
        seen = set()
        for root in [start] + list(range(n)):
            if root in seen:
                continue
            queue = deque([root])
            seen.add(root)
            while queue:
                q = queue.popleft()
                order.append(q)
                for v in sorted(adjacency[q]):
                    if v not in seen:
                        seen.add(v)
                        queue.append(v)
        yield f"bfs{start}", order
    for seed in (1900, 1901, 1902):
        yield f"random{seed}", np.random.default_rng(seed).permutation(n).tolist()


def balanced_profiles(h):
    n = h.shape[1]
    edge = (n + 3) // 4
    results = []
    for label, order in candidate_orders(h):
        profile = coupling_profile(h, order)
        cut = min(range(edge, n - edge + 1), key=lambda i: (profile[i], abs(2 * i - n), i))
        results.append(
            dict(
                order_kind=label, order=order, profile=profile, cut=cut, coupling=profile[cut], coordinates=order[:cut]
            )
        )
    return results


def preserves_rowspace(h, permutation):
    if sorted(permutation) != list(range(h.shape[1])):
        raise ValueError("Not a permutation")
    reduced, pivots = gf2.rref(h)
    basis = packed_rows(reduced)
    for transformed in packed_rows(reduced[:, permutation]):
        word = transformed
        for pivot, row in zip(pivots, basis):
            if word >> pivot & 1:
                word ^= row
        if word:
            return False
    return True


def row_mix(h, seed):
    out = h.copy()
    rng = np.random.default_rng(seed)
    for _ in range(3 * len(h)):
        a, b = rng.choice(len(h), 2, replace=False)
        out[a] ^= out[b]
    return out[rng.permutation(len(h))]


def poly_div(a, b):
    if not b:
        raise ZeroDivisionError
    q = 0
    while a and a.bit_length() >= b.bit_length():
        shift = a.bit_length() - b.bit_length()
        q ^= 1 << shift
        a ^= b << shift
    return q, a


def poly_mul(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


def poly_gcd(a, b):
    while b:
        a, b = b, poly_div(a, b)[1]
    return a


def bezout(a, b):
    u, v, x, y = 1, 0, 0, 1
    while b:
        q, r = poly_div(a, b)
        a, b = b, r
        u, x = x, u ^ poly_mul(q, x)
        v, y = y, v ^ poly_mul(q, y)
    return a, u, v


def factor(f):
    """Factor over GF(2) with multiplicity via derivatives and Berlekamp splitting."""
    if f <= 0:
        raise ValueError("Positive polynomial required")
    degree = f.bit_length() - 1
    if degree <= 1:
        return [] if f == 1 else [f]
    derivative = sum(1 << (i - 1) for i in range(1, degree + 1, 2) if f >> i & 1)
    if not derivative:
        root = sum(1 << (i // 2) for i in range(0, degree + 1, 2) if f >> i & 1)
        return sorted(factor(root) * 2)
    repeated = poly_gcd(f, derivative)
    if repeated != 1:
        return sorted(factor(repeated) + factor(poly_div(f, repeated)[0]))
    q = np.zeros((degree, degree), dtype=np.int8)
    value = 1
    for j in range(degree):
        for i in range(degree):
            q[i, j] = (value >> i) & 1
        value = poly_div(value << 2, f)[1]
    q ^= np.eye(degree, dtype=np.int8)
    basis = packed_rows(gf2.kernel_basis(q))
    parts = [f]
    for splitter in basis:
        refined = []
        for part in parts:
            divisor = poly_gcd(part, splitter)
            if divisor not in (1, part):
                refined.extend([divisor, poly_div(part, divisor)[0]])
            else:
                refined.append(part)
        parts = refined
        if len(parts) == len(basis):
            break
    if len(parts) != len(basis):
        raise RuntimeError("Incomplete factorization")
    return sorted(parts)


def irreducible(f):
    degree = f.bit_length() - 1
    if degree < 1:
        return False
    value = 2
    for step in range(1, degree + 1):
        value = poly_div(poly_mul(value, value), f)[1]
        if step <= degree // 2 and poly_gcd(f, value ^ 2) != 1:
            return False
    return value == poly_div(2, f)[1]


def orbit_rows(word, length):
    mask = (1 << length) - 1
    rows = []
    for _ in range(length):
        rows.append([(word >> q) & 1 for q in range(2 * length)])
        a, b = word & mask, word >> length
        a = ((a << 1) & mask) | (a >> (length - 1))
        b = ((b << 1) & mask) | (b >> (length - 1))
        word = a | (b << length)
    return np.array(rows, dtype=np.int8)


def polynomial_model(hx, hz):
    n = hx.shape[1]
    if n % 2 or not len(hx):
        return dict(status="not_two_block_shape")
    length = n // 2
    word = packed_rows(hx[:1])[0]
    mask = (1 << length) - 1
    a, b = word & mask, word >> length

    def reciprocal(p):
        return sum(1 << ((-i) % length) for i in range(length) if p >> i & 1)

    reconstructed_x = orbit_rows(word, length)
    reconstructed_z = orbit_rows(reciprocal(b) | (reciprocal(a) << length), length)
    if not np.array_equal(gf2.rref(hx)[0], gf2.rref(reconstructed_x)[0]):
        return dict(status="X_orbit_does_not_span")
    if not np.array_equal(gf2.rref(hz)[0], gf2.rref(reconstructed_z)[0]):
        return dict(status="Z_reciprocal_mismatch")
    modulus = (1 << length) | 1
    d, u, v = bezout(a, b)
    common = poly_gcd(d, modulus)
    factors = factor(modulus)
    product = 1
    for f in factors:
        assert irreducible(f)
        product = poly_mul(product, f)
    assert product == modulus and poly_mul(u, a) ^ poly_mul(v, b) == d

    def degree(x):
        return x.bit_length() - 1

    predicted = 2 * degree(common)
    assert predicted == n - gf2.rank(hx) - gf2.rank(hz)
    return dict(
        status="verified",
        length=length,
        a=hex(a),
        b=hex(b),
        modulus=hex(modulus),
        gcd_ab=hex(d),
        gcd_common=hex(common),
        bezout_u=hex(u),
        bezout_v=hex(v),
        quantum_dimension=predicted,
        half_kernel_dimensions=[degree(poly_gcd(a, modulus)), degree(poly_gcd(b, modulus))],
        squarefree=length % 2 == 1,
        factors=[
            dict(polynomial=hex(f), degree=degree(f), divides_common=poly_div(common, f)[1] == 0) for f in factors
        ],
    )
