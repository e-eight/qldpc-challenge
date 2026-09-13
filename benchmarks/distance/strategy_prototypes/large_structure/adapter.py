"""Construction-assisted restrictions; references and targets never enter search."""

import math
import sys
import time
from pathlib import Path

from run import native_batch_seed

# isort: split
from allocation_search import Search as ControlSearch
from candidate_search import native_search
from initialized_search import LogicalPool, initialize, support_of
from study_strategies import ris_native

sys.path.insert(0, str(Path(__file__).resolve().parent))
import large_structure_native as native  # noqa: E402

CONFIG = dict(
    setup_fraction=0.20,
    fallback_fraction=0.20,
    slice_seconds=0.020,
    batch_seconds=0.002,
    batch_cap=64,
    pairs=8,
    strip_vectors=6,
    strip_widths=[1, 2, 3, 4],
    strip_max_fraction=0.60,
)


def proposals(spec, n):
    """Yield disjoint encoder-column supports from construction parameters."""
    if spec["kind"] == "product":
        length = spec["group_order"]
        if n != 5 * length:
            raise ValueError("Product dimension mismatch")
        for first, second in ((0, 1), (2, 3), (0, 2), (1, 3)):
            support = list(range(first * length, (first + 1) * length)) + list(
                range(second * length, (second + 1) * length)
            )
            yield f"seed_blocks_{first}_{second}", [[q] for q in support]
    elif spec["kind"] == "torus":
        length, twist = spec["length"], spec["twist"]
        if n != 2 * length:
            raise ValueError("Torus dimension mismatch")
        limit = 2 * math.ceil(math.sqrt(length))
        vectors = [
            (x, y)
            for x in range(limit + 1)
            for y in range(-limit, limit + 1)
            if (x > 0 or y > 0) and (y - twist * x) % length == 0
        ]
        vectors.sort(key=lambda v: (v[0] * v[0] + v[1] * v[1], v))
        primitive = []
        for x, y in vectors:
            if any(x * b == y * a and x * a + y * b > 0 for a, b in primitive):
                continue
            primitive.append((x, y))
        # Interleave widths so every short cycle gets a narrow probe first.
        for width in CONFIG["strip_widths"]:
            for x, y in primitive[: CONFIG["strip_vectors"]]:
                steps = max(abs(x), abs(y))
                cells = {
                    (round(i * y / steps) + dy - twist * (round(i * x / steps) + dx)) % length
                    for i in range(steps + 1)
                    for dx in range(-width, width + 1)
                    for dy in range(-width, width + 1)
                    if abs(dx) + abs(dy) <= width
                }
                if len(cells) > CONFIG["strip_max_fraction"] * length:
                    continue
                support = sorted(cells) + [q + length for q in sorted(cells)]
                yield f"strip_{x}_{y}_w{width}", [[q] for q in support]
    elif spec["kind"] == "affine":
        prime, order, action = spec["prime"], spec["order"], spec["action"]
        length = prime * order
        if n != 2 * length:
            raise ValueError("Affine dimension mismatch")
        generators = [(0, order // d) for d in (2, 3, 6, 9, 18) if order % d == 0] + [(1, 0)]
        seen = set()
        for side in ("left", "right"):
            for a, b in generators:
                permutation = []
                for x in range(prime):
                    for y in range(order):
                        u, v = (
                            ((a + pow(action, b, prime) * x) % prime, (b + y) % order)
                            if side == "left"
                            else ((x + pow(action, y, prime) * a) % prime, (y + b) % order)
                        )
                        permutation.append(u * order + v)
                unused = set(range(length))
                groups = []
                while unused:
                    q = min(unused)
                    orbit = []
                    while q in unused:
                        unused.remove(q)
                        orbit.append(q)
                        q = permutation[q]
                    groups.append(sorted(orbit))
                groups += [[q + length for q in group] for group in groups.copy()]
                key = tuple(tuple(group) for group in groups)
                if key in seen:
                    continue
                seen.add(key)
                yield f"orbits_{side}_{a}_{b}", groups
    else:
        raise ValueError("Unknown structural hypothesis")


class Search:
    def __init__(self, own, opposite, method, spec):
        if method not in ("guided", "race", "structure"):
            raise ValueError("Unknown method")
        self.own, self.opposite, self.method, self.spec = own, opposite, method, spec

    def run(self, seconds, seed, emit):
        if self.method != "structure":
            return ControlSearch(self.own, self.opposite, self.method).run(seconds, seed, emit)
        if not math.isfinite(seconds) or seconds <= 0 or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
            raise ValueError("Invalid budget or seed")
        started = time.perf_counter()
        deadline = started + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        counters = dict(parameters=CONFIG, preparation_seconds=time.perf_counter() - started, spec=self.spec)
        if not prepared.applicable:
            return dict(counters, status="not_applicable")
        best = self.own.shape[1] + 1

        def observe(word, stage, improving_only=False):
            nonlocal best
            weight = word.bit_count()
            if improving_only and weight >= best:
                return
            best = min(best, weight)
            emit(weight, support_of(word), stage)

        tick = time.perf_counter()
        counters["initialization"] = initialize(rows, LogicalPool(prepared.logicals, 8), observe, deadline)
        counters["initialization_seconds"] = time.perf_counter() - tick
        counters["initial_best"] = best
        structure_start = time.perf_counter()
        remaining = max(0, deadline - structure_start)
        structure_deadline = deadline - CONFIG["fallback_fraction"] * remaining
        setup_deadline = min(structure_deadline, structure_start + CONFIG["setup_fraction"] * remaining)
        spaces = []
        counters["spaces"] = []
        for label, groups in proposals(self.spec, self.own.shape[1]):
            if time.perf_counter() >= setup_deadline:
                break
            space = native.Space(self.opposite, prepared.logicals, groups)
            entry = dict(
                label=label,
                groups=groups,
                dimension=space.dimension,
                logical_rank=space.logical_rank,
                workspace_bytes=space.workspace_bytes,
                batches=0,
                scored_bases=0,
                exports=0,
                best=None,
            )
            counters["spaces"].append(entry)
            if space.logical_rank:
                spaces.append([space, entry, 1])
        counters["setup_seconds"] = time.perf_counter() - structure_start
        counters["active_spaces"] = len(spaces)
        number = 0
        while spaces and time.perf_counter() < structure_deadline:
            index = number % len(spaces)
            number += 1
            space, entry, batch = spaces[index]
            slice_end = min(structure_deadline, time.perf_counter() + CONFIG["slice_seconds"])
            while time.perf_counter() < slice_end:
                tick = time.perf_counter()
                weight, support = space.batch(
                    batch, native_batch_seed(seed, number * 1000003 + entry["batches"]), CONFIG["pairs"]
                )
                duration = time.perf_counter() - tick
                entry["batches"] += 1
                entry["scored_bases"] += batch
                if support:
                    emit(int(weight), list(support), entry["label"])
                    entry["exports"] += 1
                    entry["best"] = int(weight) if entry["best"] is None else min(entry["best"], int(weight))
                batch = max(
                    1,
                    min(
                        CONFIG["batch_cap"],
                        int(
                            batch
                            * min(CONFIG["batch_seconds"], max(0, slice_end - time.perf_counter()))
                            / max(duration, 1e-9)
                        ),
                    ),
                )
            spaces[index][2] = batch
        counters["structure_seconds"] = time.perf_counter() - structure_start
        if time.perf_counter() < deadline:
            tick = time.perf_counter()
            counters["fallback"] = native_search(self.own, self.opposite, prepared, "guided", deadline, seed, emit)
            counters["fallback_seconds"] = time.perf_counter() - tick
        return dict(counters, status="completed", elapsed_seconds=time.perf_counter() - started)
