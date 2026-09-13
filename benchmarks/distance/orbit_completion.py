"""Propose missing checks by cyclic shifts within two equal coordinate blocks.

This is a layout-dependent diagnostic, using matrices only. A proposed shift is
not assumed to be a code symmetry: syndrome and logical parity are tested for
every proposal. The harness independently validates and saves every emission.
"""

import numpy as np
from initialized_search import pack_rows, support_of
from study_strategies import ris_native


def rotate_blocks(word, length):
    mask = (1 << length) - 1
    lo, hi = word & mask, word >> length
    lo = ((lo << 1) & mask) | (lo >> (length - 1))
    hi = ((hi << 1) & mask) | (hi >> (length - 1))
    return lo | (hi << length)


def search_orbits(own, opposite, duals, emit):
    n = own.shape[1]
    counters = {"proposals": 0, "orbits": 0, "zero_syndrome": 0, "logical_witnesses": 0}
    if n % 2:
        return dict(counters, status="not_applicable")
    seen = set()
    checks, logicals = pack_rows(opposite), pack_rows(duals)
    for original in pack_rows(own):
        word = original
        if word in seen:
            continue
        counters["orbits"] += 1
        for _ in range(n // 2):
            if word not in seen:
                seen.add(word)
                counters["proposals"] += 1
                if not any((word & row).bit_count() & 1 for row in checks):
                    counters["zero_syndrome"] += 1
                    if any((word & row).bit_count() & 1 for row in logicals):
                        counters["logical_witnesses"] += 1
                        emit(word.bit_count(), support_of(word), "orbit")
            word = rotate_blocks(word, n // 2)
    return dict(counters, status="completed")


class OrbitSearch:
    def __init__(self, own, opposite):
        self.own, self.opposite = own, opposite

    def run(self, seconds, seed, emit):
        # Full finite enumeration; seconds is intentionally unused. This is a
        # one-shot structure diagnostic, not a deadline-limited search adapter.
        prepared = ris_native.Prepared(self.own, self.opposite)
        representatives = ris_native.Prepared(self.opposite, self.own).logicals
        if not prepared.applicable:
            return {"status": "not_applicable"}
        first = np.flatnonzero(representatives[0]).tolist()
        emit(len(first), first, "packaging_basis")
        return search_orbits(self.own, self.opposite, prepared.logicals, emit)
