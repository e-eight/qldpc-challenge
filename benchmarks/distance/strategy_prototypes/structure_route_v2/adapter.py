"""Route inferred coordinate blocks to the existing single-block search core."""

import math
import time

from run import benchmark_native, native_batch_seed, np

# isort: split
from strategy_prototypes.matrix_structure_v2.adapter import pack_rows, recover_translation

CONFIG = {
    "translation_proposals": ["own_adjacent_rows", "opposite_adjacent_rows"],
    "required_cycles": 2,
    "batch_seconds": 0.050,
    "batch_cap": 4096,
    "pairs": 8,
    "threads": 1,
}


def two_cycle_order(permutation):
    """Return reordered-position -> input-position, or None for no two-block route."""
    if permutation is None or len(permutation) % 2:
        return None
    n = len(permutation)
    if not np.array_equal(np.sort(permutation), np.arange(n)):
        raise ValueError("Recovered mapping is not a column permutation")
    seen, cycles = set(), []
    for first in range(n):
        if first in seen:
            continue
        cycle, current = [], first
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = int(permutation[current])
        cycles.append(cycle)
        if len(cycles) > 2:
            return None
    if len(cycles) != 2 or any(len(cycle) != n // 2 for cycle in cycles):
        return None
    return np.asarray(cycles[0] + cycles[1], dtype=np.int64)


def partition_key(order):
    """Cycle starting point/direction and exchanging the two blocks add no route."""
    middle = len(order) // 2
    return tuple(sorted((tuple(sorted(order[:middle].tolist())), tuple(sorted(order[middle:].tolist())))))


def search(own, opposite, duals, seconds, seed, emit):
    start = time.perf_counter()
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    deadline = start + seconds
    counters = {
        "parameters": CONFIG,
        "applicable": False,
        "status": "no_route",
        "row_order_assumption": True,
        "routes": [],
        "recovery": [],
        "batches": 0,
        "scored_bases": 0,
        "exported": 0,
        "best": None,
    }
    n = own.shape[1]
    checks, detectors = pack_rows(opposite), pack_rows(duals)
    seen_partitions, sessions = set(), []
    for label, matrix in zip(CONFIG["translation_proposals"], (own, opposite), strict=True):
        if time.perf_counter() >= deadline:
            counters["status"] = "setup_deadline"
            break
        before = time.perf_counter()
        order = two_cycle_order(recover_translation(matrix))
        recovery = {"source": label, "seconds": time.perf_counter() - before, "two_equal_cycles": order is not None}
        counters["recovery"].append(recovery)
        if order is None:
            continue
        key = partition_key(order)
        if key in seen_partitions:
            recovery["duplicate_partition"] = True
            continue
        seen_partitions.add(key)
        if time.perf_counter() >= deadline:
            counters["status"] = "setup_deadline"
            break
        before = time.perf_counter()
        # Explicit block size bypasses only the layout recognizer. The existing
        # core constructs actual restricted kernels and actual logical detectors.
        prepared = benchmark_native.PreparedSearch(
            np.ascontiguousarray(own[:, order]), np.ascontiguousarray(opposite[:, order]), True, n // 2
        )
        route = {
            "source": label,
            "block_size": n // 2,
            "preparation_seconds": time.perf_counter() - before,
            "native_applicable": bool(prepared.applicable()),
            "batches": 0,
            "scored_bases": 0,
            "exported": 0,
            "best": None,
        }
        counters["routes"].append(route)
        if prepared.applicable():
            sessions.append({"prepared": prepared, "order": order, "route": route, "batch_size": 1})
    counters["setup_seconds"] = time.perf_counter() - start
    counters["applicable"] = bool(sessions)
    step = 0
    while sessions and time.perf_counter() < deadline:
        index = step % len(sessions)
        step += 1
        session = sessions[index]
        route, order = session["route"], session["order"]
        number, batch_size = route["batches"], session["batch_size"]
        before = time.perf_counter()
        weight, returned, completed = session["prepared"].batch(
            batch_size, native_batch_seed(seed + index * 100003, number), CONFIG["pairs"], CONFIG["threads"], number
        )
        duration = time.perf_counter() - before
        counters["batches"] += 1
        counters["scored_bases"] += int(completed)
        route["batches"] += 1
        route["scored_bases"] += int(completed)
        if returned:
            support = sorted(int(order[q]) for q in returned)
            word = sum(1 << q for q in support)
            if len(support) != len(set(support)) or weight != word.bit_count():
                raise RuntimeError("Native routed search returned malformed support")
            if any((word & row).bit_count() & 1 for row in checks):
                raise RuntimeError("Native routed search returned nonzero syndrome")
            if not any((word & row).bit_count() & 1 for row in detectors):
                raise RuntimeError("Native routed search returned a trivial logical")
            # Preserve every native batch return, including ties and late output.
            emit(int(weight), support, "routed_single_block")
            counters["exported"] += 1
            route["exported"] += 1
            counters["best"] = int(weight) if counters["best"] is None else min(counters["best"], int(weight))
            route["best"] = int(weight) if route["best"] is None else min(route["best"], int(weight))
        remaining = max(0, deadline - time.perf_counter())
        session["batch_size"] = max(
            1,
            min(CONFIG["batch_cap"], int(batch_size * min(CONFIG["batch_seconds"], remaining) / max(duration, 1e-9))),
        )
    if sessions:
        counters["status"] = "deadline"
    counters["elapsed_seconds"] = time.perf_counter() - start
    return counters
