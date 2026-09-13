"""Search verified polynomial-generated subspaces with packed higher-order RIS."""

import sys
import time
from pathlib import Path

import numpy as np
from initialized_search import pack_rows
from polynomial_completion import divide, gcd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reduced_space_native  # noqa: E402

CONFIG = {
    "max_total_words": 64,
    "beam_logical": 32,
    "beam_trivial": 32,
    "light_rows": 64,
    "max_combination_depth": 4,
    "schedule": ["reduced_both"] * 4 + ["reduced_single_left", "reduced_single_right"],
}


def rotate(word, length):
    mask = (1 << length) - 1
    a, b = word & mask, word >> length
    a = ((a << 1) & mask) | (a >> (length - 1))
    b = ((b << 1) & mask) | (b >> (length - 1))
    return a | (b << length)


def independent(words):
    pivots = {}
    for original in words:
        word = original
        while word:
            pivot = word.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = word
                break
            word ^= pivots[pivot]
    return list(pivots.values())


def polynomial_generators(own, opposite, deadline):
    """Finite matrix-only reduction; no witness exports or assumed layout correctness."""
    n = own.shape[1]
    stats = {"status": "not_applicable", "rejected_generators": 0, "raw_generator_counts": {}}
    if n % 2 or not own.shape[0]:
        return {}, stats
    length = n // 2
    word = pack_rows(own[:1])[0]
    mask, modulus = (1 << length) - 1, (1 << length) | 1
    a, b = word & mask, word >> length
    common = gcd(a, b)
    shared = gcd(common, modulus)
    stats.update(common_factor_degree=common.bit_length() - 1, cyclic_factor_degree=shared.bit_length() - 1)
    if common <= 1 or shared <= 1:
        return {}, stats
    qa, ra = divide(a, common)
    qb, rb = divide(b, common)
    cyclic, remainder = divide(modulus, shared)
    if ra or rb or remainder:
        raise RuntimeError("Invalid exact polynomial division")
    checks = pack_rows(opposite)
    seeds = {
        "reduced_both": qa | (qb << length),
        "reduced_single_left": cyclic,
        "reduced_single_right": cyclic << length,
    }
    spaces = {}
    for label, initial in seeds.items():
        candidate = initial
        vectors = []
        for _ in range(length):
            if time.perf_counter() >= deadline:
                stats["status"] = "deadline"
                return spaces, stats
            if not any((candidate & row).bit_count() & 1 for row in checks):
                vectors.append(candidate)
            else:
                stats["rejected_generators"] += 1
            candidate = rotate(candidate, length)
        stats["raw_generator_counts"][label] = len(vectors)
        basis = independent(vectors)
        if basis:
            spaces[label] = basis
    stats["status"] = "applicable" if spaces else "not_applicable"
    stats["dimensions"] = {label: len(rows) for label, rows in spaces.items()}
    return spaces, stats


def packed_with_tags(words, duals, n):
    detectors = pack_rows(duals)
    nw, tw = (n + 63) // 64, (len(detectors) + 63) // 64
    result = np.zeros((len(words), nw + tw), dtype=np.uint64)
    mask = (1 << 64) - 1
    for r, word in enumerate(words):
        for j in range(nw):
            result[r, j] = (word >> (64 * j)) & mask
        for j, detector in enumerate(detectors):
            if (word & detector).bit_count() & 1:
                result[r, nw + j // 64] |= np.uint64(1) << np.uint64(j % 64)
    return result


def search(own, opposite, duals, seconds, seed, emit):
    start = time.perf_counter()
    deadline = start + max(0, seconds)
    n = own.shape[1]
    stats = {"config": CONFIG, "branches": {}, "exported": 0}
    if not len(duals) or (n + 63) // 64 + (len(duals) + 63) // 64 > CONFIG["max_total_words"]:
        return dict(stats, status="dimension_limit_or_no_logicals", elapsed_seconds=time.perf_counter() - start)
    spaces, reduction = polynomial_generators(own, opposite, deadline)
    stats["reduction"] = reduction
    checks, detectors = pack_rows(opposite), pack_rows(duals)
    best = n + 1
    sessions = {}
    for i, (label, rows) in enumerate(spaces.items()):
        if time.perf_counter() >= deadline:
            break
        packed = packed_with_tags(rows, duals, n)
        if not packed[:, (n + 63) // 64 :].any():
            continue
        sessions[label] = reduced_space_native.Session(packed, n, len(duals), seed + i * 100003)
        stats["branches"][label] = {"trials": 0, "scored": 0, "seconds": 0.0, "rank": len(rows), "best": None}
    stats["setup_seconds"] = time.perf_counter() - start
    schedule = [label for label in CONFIG["schedule"] if label in sessions]
    step = 0
    while schedule and time.perf_counter() < deadline:
        label = schedule[step % len(schedule)]
        step += 1
        tick = time.perf_counter()
        result = sessions[label].advance()
        branch = stats["branches"][label]
        branch.update({key: result[key] for key in ("trials", "scored", "rank", "workspace_bytes")})
        for weight, support in result["events"]:
            word = sum(1 << j for j in support)
            if weight != word.bit_count() or any((word & row).bit_count() & 1 for row in checks):
                raise RuntimeError("Native reduced-space witness has invalid syndrome or weight")
            if not any((word & row).bit_count() & 1 for row in detectors):
                raise RuntimeError("Native reduced-space witness is logically trivial")
            # Preserve every branch's improvement, even when another branch is already lighter.
            emit(weight, support, label)
            stats["exported"] += 1
            branch["best"] = weight
            best = min(best, weight)
        branch["seconds"] += time.perf_counter() - tick
    stats["best"] = best if best <= n else None
    stats["workspace_bytes"] = sum(branch.get("workspace_bytes", 0) for branch in stats["branches"].values())
    stats["status"] = "deadline" if schedule else reduction["status"]
    stats["elapsed_seconds"] = time.perf_counter() - start
    return stats
