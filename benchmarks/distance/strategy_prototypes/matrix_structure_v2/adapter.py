"""Bounded matrix-only translation recovery, orbit completion and sector transfer."""

import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

MAX_COLUMNS = 4096


def pack_rows(rows):
    return [int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little") for row in rows]


def support(word):
    result = []
    while word:
        bit = word & -word
        result.append(bit.bit_length() - 1)
        word ^= bit
    return result


def transform(word, permutation):
    result = 0
    while word:
        bit = word & -word
        result |= 1 << int(permutation[bit.bit_length() - 1])
        word ^= bit
    return result


def recover_translation(rows):
    """Infer old-column -> new-column map from adjacent supplied check rows.

    Missing parent rows introduce outliers. No assertion of code symmetry is made.
    Matching is equivariant under qubit permutations when its optimum is unique.
    """
    if rows.shape[0] < 2 or rows.shape[1] > MAX_COLUMNS:
        return None
    earlier = csr_matrix(rows[:-1].astype(np.int32))
    later = csr_matrix(rows[1:].astype(np.int32))
    scores = (earlier.T @ later).toarray()
    _, permutation = linear_sum_assignment(scores, maximize=True)
    return permutation


def sector_maps(own, opposite):
    """Exact row-set exchange, using aligned/reversed row order as proposals."""
    if own.shape != opposite.shape:
        return []
    own_set, opposite_set = set(pack_rows(own)), set(pack_rows(opposite))
    result = []
    for reverse in (False, True):
        source = own[::-1] if reverse else own
        groups = defaultdict(list)
        for column, values in enumerate(opposite.T):
            groups[values.tobytes()].append(column)
        permutation = []
        for values in source.T:
            matches = groups[values.tobytes()]
            if not matches:
                break
            permutation.append(matches.pop())
        if len(permutation) != own.shape[1]:
            continue
        permutation = np.asarray(permutation, dtype=np.int64)
        inverse = np.argsort(permutation)
        if set(pack_rows(own[:, inverse])) != opposite_set:
            continue
        if set(pack_rows(opposite[:, inverse])) != own_set:
            continue
        if not any(np.array_equal(permutation, previous) for previous in result):
            result.append(permutation)
    return result


def cycle_lengths(permutation):
    seen, lengths = set(), []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current, length = start, 0
        while current not in seen:
            seen.add(current)
            length += 1
            current = int(permutation[current])
        lengths.append(length)
    return sorted(lengths)


def search(own, opposite, duals, seconds, seed, emit):
    """All analysis and callback delivery count against seconds; seed is unused."""
    del seed
    start = time.perf_counter()
    deadline = start + seconds
    counters = {
        "applicable": False,
        "complete": False,
        "status": "deadline",
        "translations": [],
        "sector_maps": 0,
        "proposals": 0,
        "zero_syndrome": 0,
        "logical_candidates": 0,
        "emitted": 0,
        "method": "adjacent-row overlap assignment plus checked orbit completion and exact sector transfer",
        "max_columns": MAX_COLUMNS,
        "row_order_assumption": True,
    }
    checks, logicals = pack_rows(opposite), pack_rows(duals)
    original_rows = pack_rows(own)
    emitted, tested = set(), set()

    def consider(word, stage, improvements_only=False):
        if word in tested:
            return
        tested.add(word)
        counters["proposals"] += 1
        if any((word & row).bit_count() & 1 for row in checks):
            return
        counters["zero_syndrome"] += 1
        if not any((word & row).bit_count() & 1 for row in logicals):
            return
        counters["logical_candidates"] += 1
        weight = word.bit_count()
        previous = counters.get("best")
        if improvements_only and previous is not None and weight >= previous:
            return
        if word not in emitted:
            emit(weight, support(word), stage)
            emitted.add(word)
            counters["emitted"] += 1
        counters["best"] = weight if previous is None else min(previous, weight)

    def finish(complete=False):
        counters["complete"] = complete
        counters["status"] = "completed" if complete else "deadline"
        counters["elapsed_seconds"] = time.perf_counter() - start
        return counters

    if time.perf_counter() >= deadline:
        return finish()
    if own.shape[1] > MAX_COLUMNS:
        counters["status"] = "size_cap"
        counters["elapsed_seconds"] = time.perf_counter() - start
        return counters

    # Exact sector exchange lets us reuse opposite-sector basis combinations.
    symmetries = sector_maps(own, opposite)
    counters["sector_maps"] = len(symmetries)
    counters["applicable"] = bool(symmetries)
    for permutation in symmetries:
        transferred = [transform(row, permutation) for row in logicals]
        for row in transferred:
            if time.perf_counter() >= deadline:
                return finish()
            consider(row, "sector_transfer", improvements_only=True)
        if len(transferred) <= 16:
            word, previous_gray = 0, 0
            for index in range(1, 1 << len(transferred)):
                if time.perf_counter() >= deadline:
                    return finish()
                gray = index ^ (index >> 1)
                word ^= transferred[(gray ^ previous_gray).bit_length() - 1]
                previous_gray = gray
                consider(word, "sector_transfer", improvements_only=True)
        else:
            for i, row in enumerate(transferred):
                for other in transferred[:i]:
                    if time.perf_counter() >= deadline:
                        return finish()
                    consider(row ^ other, "sector_transfer", improvements_only=True)

    # Orbit exports retain accepted proposals even if transfer scored them without
    # exporting because they did not improve its incumbent.
    tested.clear()

    permutations = []
    # Recovery first: its proposal policy does not depend on supplied columns.
    for label, matrix in (("own_adjacent_rows", own), ("opposite_adjacent_rows", opposite)):
        if time.perf_counter() >= deadline:
            return finish()
        analysis_start = time.perf_counter()
        permutation = recover_translation(matrix)
        if permutation is None or any(np.array_equal(permutation, old) for old in permutations):
            continue
        permutations.append(permutation)
        inverse = np.argsort(permutation)
        overlap = len(set(pack_rows(own[:, inverse])) & set(original_rows))
        counters["translations"].append(
            {
                "source": label,
                "cycle_lengths": cycle_lengths(permutation),
                "own_rowset_overlap": overlap,
                "own_rows": len(set(original_rows)),
                "analysis_seconds": time.perf_counter() - analysis_start,
            }
        )
    n = own.shape[1]
    if n % 2 == 0:
        length = n // 2
        layout = np.r_[np.roll(np.arange(length), -1), np.roll(np.arange(length, n), -1)]
        if not any(np.array_equal(layout, old) for old in permutations):
            permutations.append(layout)
            counters["translations"].append({"source": "supplied_two_block_layout", "cycle_lengths": [length] * 2})
    counters["analysis_and_transfer_seconds"] = time.perf_counter() - start
    counters["applicable"] = bool(permutations or symmetries)
    for permutation, description in zip(permutations, counters["translations"], strict=True):
        seen = set()
        stage = "orbit_" + description["source"]
        for row in original_rows:
            word = row
            # Arbitrary permutation orders can exceed n. Deliberately cap each orbit.
            for _ in range(n):
                if time.perf_counter() >= deadline:
                    return finish()
                if word in seen:
                    break
                seen.add(word)
                consider(word, stage)
                word = transform(word, permutation)
        description["orbit_proposals"] = len(seen)
    return finish(complete=True)
