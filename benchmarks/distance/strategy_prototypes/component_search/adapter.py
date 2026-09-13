"""Bounded component analysis with unchanged native search engines."""

import time

from run import gf2, np

# isort: split
from allocation_search import GuidedSession
from allocation_search import Search as Guided
from dispatch_search import detector
from initialized_search import LogicalPool, initialize, support_of
from strategy_prototypes.block_collision.adapter import native, restriction, ris_native

METHODS = ("guided", "auto-components", "metadata-components")
CONFIG = dict(
    setup_cap=0.15,
    setup_fraction=0.15,
    detector_cap=0.03,
    detector_fraction=0.03,
    dimension_cap=64,
    exact_cap=20,
    max_spaces=32,
    slice_seconds=0.005,
    chunk=4096,
    component_fraction=0.5,
)


def components(checks, coordinates):
    """Find qubit components by joining columns sharing an original check."""
    coords = sorted(coordinates)
    if len(set(coords)) != len(coords) or any(q < 0 or q >= checks.shape[1] for q in coords):
        raise ValueError("Invalid coordinate subset")
    parent = list(range(len(coords)))

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for row in checks[:, coords]:
        support = np.flatnonzero(row)
        for q in support[1:]:
            parent[find(int(q))] = find(int(support[0]))
    groups = {}
    for q in range(len(coords)):
        groups.setdefault(find(q), []).append(coords[q])
    return sorted(groups.values(), key=lambda c: c[0])


def compress_duals(basis, duals):
    """Preserve exactly the zero/nonzero logical pairing on this basis span."""
    tags = (basis @ duals.T) % 2
    _, columns = gf2.rref(tags)
    return np.ascontiguousarray(duals[columns]), columns


class Search:
    def __init__(self, own, opposite, method, blocks=()):
        if method not in METHODS + ("exact-components",):
            raise ValueError("Unknown method")
        self.own, self.opposite, self.method, self.blocks = own, opposite, method, blocks

    def run(self, seconds, seed, emit):
        if self.method == "guided":
            return Guided(self.own, self.opposite, "guided").run(seconds, seed, emit)
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite budget required")
        started = time.perf_counter()
        deadline = started + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        counters = dict(parameters=CONFIG, method=self.method, preparation_seconds=time.perf_counter() - started)
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
        tick = time.perf_counter()
        setup_end = min(deadline, tick + min(CONFIG["setup_cap"], CONFIG["setup_fraction"] * seconds))
        exact = self.method == "exact-components"
        if exact:
            setup_end = deadline
        n = self.own.shape[1]
        proposals = [("whole", list(range(n)))]
        counters["detection"] = []
        if self.method == "auto-components":
            detect_end = min(
                setup_end, time.perf_counter() + min(CONFIG["detector_cap"], CONFIG["detector_fraction"] * seconds)
            )
            for label, matrix in [("own", self.own), ("opposite", self.opposite)]:
                if time.perf_counter() >= detect_end:
                    break
                result = detector.detect(
                    np.ascontiguousarray(matrix, dtype=np.uint8), max(0, detect_end - time.perf_counter()), 0.5
                )
                counters["detection"].append(dict(source=label, **result))
                if result["status"] == "accepted":
                    order = result["order"]
                    proposals.extend([(label + "0", sorted(order[: n // 2])), (label + "1", sorted(order[n // 2 :]))])
                    break
        else:
            proposals.extend((f"metadata{i}", list(block)) for i, block in enumerate(self.blocks))
        counters["proposals"] = [dict(label=label, coordinates=q) for label, q in proposals]
        counters["spaces"] = []
        seen = set()
        sessions = []
        for label, coords in proposals:
            if time.perf_counter() >= setup_end:
                break
            for group in components(self.opposite, coords):
                if time.perf_counter() >= setup_end or len(sessions) >= CONFIG["max_spaces"]:
                    break
                key = tuple(group)
                if key in seen:
                    continue
                seen.add(key)
                space = restriction.Space(self.opposite, prepared.logicals, [[q] for q in group])
                entry = dict(
                    label=f"component{len(counters['spaces'])}",
                    proposal=label,
                    coordinates=group,
                    dimension=space.dimension,
                    logical_rank=space.logical_rank,
                )
                counters["spaces"].append(entry)
                if not space.logical_rank:
                    entry["status"] = "no_logicals"
                    continue
                cap = CONFIG["exact_cap"] if exact else CONFIG["dimension_cap"]
                if space.dimension > cap:
                    entry["status"] = "dimension_cap"
                    continue
                duals, columns = compress_duals(space.basis, prepared.logicals)
                if len(duals) != space.logical_rank:
                    raise RuntimeError("Pairing rank mismatch")
                entry["dual_columns"] = columns
                session = native.Session(space.basis, duals, (seed + len(sessions) * 104729) % (1 << 64), False)
                entry["exact"] = space.dimension <= CONFIG["exact_cap"]
                entry["status"] = "active"
                entry["stats"] = session.stats
                sessions.append((session, entry))
        counters["setup_seconds"] = time.perf_counter() - tick
        counters["setup_complete"] = time.perf_counter() < setup_end and len(sessions) < CONFIG["max_spaces"]
        component_end = (
            deadline
            if exact
            else time.perf_counter() + CONFIG["component_fraction"] * max(0, deadline - time.perf_counter())
        )
        counters["component_deadline_seconds"] = component_end - started
        number = 0
        tick = time.perf_counter()
        while sessions and time.perf_counter() < component_end:
            session, entry = sessions[number % len(sessions)]
            number += 1
            until = min(component_end, time.perf_counter() + CONFIG["slice_seconds"])
            while time.perf_counter() < until:
                events = session.enumerate_chunk(CONFIG["chunk"]) if entry["exact"] else session.advance(1)
                for weight, support, stage in events:
                    emit(weight, support, entry["label"] + ":" + stage)
                if entry["exact"] and session.stats["enumeration_complete"]:
                    entry["status"] = "exhausted"
                    break
            entry["stats"] = session.stats
            if entry["status"] == "exhausted":
                sessions.remove((session, entry))
                number = 0
        counters["component_seconds"] = time.perf_counter() - tick
        if not exact and time.perf_counter() < deadline:
            tick = time.perf_counter()
            guided = GuidedSession(self.own, self.opposite, seed, emit)
            guided.advance_until(deadline)
            counters["fallback"] = guided.counters
            counters["fallback_seconds"] = time.perf_counter() - tick
        return dict(counters, status="completed", elapsed_seconds=time.perf_counter() - started)
