"""Finite two-block proposals from common factors of binary check polynomials."""

import numpy as np
from initialized_search import pack_rows, support_of
from orbit_completion import rotate_blocks
from study_strategies import ris_native


def divide(a, b):
    if not b:
        raise ValueError("Polynomial division by zero")
    quotient = 0
    while a and a.bit_length() >= b.bit_length():
        shift = a.bit_length() - b.bit_length()
        quotient ^= 1 << shift
        a ^= b << shift
    return quotient, a


def gcd(a, b):
    while b:
        a, b = b, divide(a, b)[1]
    return a


def search_quotients(own, opposite, duals, emit):
    n = own.shape[1]
    counters = {
        "proposals": 0,
        "zero_syndrome": 0,
        "logical_witnesses": 0,
        "valid_logical_proposals": 0,
        "factor_degrees": [],
    }
    if n % 2 or not own.shape[0]:
        return dict(counters, status="not_applicable")
    length = n // 2
    mask, modulus = (1 << length) - 1, (1 << length) | 1
    word = pack_rows(own[:1])[0]
    checks, logicals = pack_rows(opposite), pack_rows(duals)
    seen, degrees = set(), set()
    best = n + 1
    for _ in range(length):
        a, b = word & mask, word >> length
        common = gcd(a, b)
        for factor in sorted({common, gcd(common, modulus)}):
            if factor <= 1:
                continue
            degrees.add(factor.bit_length() - 1)
            qa, ra = divide(a, factor)
            qb, rb = divide(b, factor)
            if ra or rb:
                raise RuntimeError("Common factor did not divide both polynomials")
            candidate = qa | (qb << length)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            counters["proposals"] += 1
            if any((candidate & row).bit_count() & 1 for row in checks):
                continue
            counters["zero_syndrome"] += 1
            if any((candidate & row).bit_count() & 1 for row in logicals):
                counters["valid_logical_proposals"] += 1
                weight = candidate.bit_count()
                if weight < best:
                    best = weight
                    counters["logical_witnesses"] += 1
                    emit(weight, support_of(candidate), "polynomial")
        word = rotate_blocks(word, length)
    counters["factor_degrees"] = sorted(degrees)
    return dict(counters, status="completed")


class PolynomialSearch:
    def __init__(self, own, opposite):
        self.own, self.opposite = own, opposite

    def run(self, seconds, seed, emit):
        # Finite enumeration, not a deadline-limited adapter.
        prepared = ris_native.Prepared(self.own, self.opposite)
        representatives = ris_native.Prepared(self.opposite, self.own).logicals
        if not prepared.applicable:
            return {"status": "not_applicable"}
        first = np.flatnonzero(representatives[0]).tolist()
        emit(len(first), first, "packaging_basis")
        return search_quotients(self.own, self.opposite, prepared.logicals, emit)
