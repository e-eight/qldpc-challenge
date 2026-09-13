"""Controlled full/block searches; references and target weights are not inputs."""

import sys
import time
from pathlib import Path

from run import np

# isort: split
from allocation_search import Search as Guided
from initialized_search import LogicalPool, initialize, support_of
from strategy_prototypes.large_structure.adapter import native as restriction
from study_strategies import ris_native

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_collision_native as native  # noqa: E402

METHODS = ("guided", "full-pairs", "block-pairs", "full-stern", "block-stern")
CONFIG = dict(slice_seconds=0.010, trials_per_batch=1, enumeration_cap=20, enumeration_chunk=4096)


class Search:
    def __init__(self, own, opposite, method):
        if method not in METHODS + ("block-exact",):
            raise ValueError("Unknown method")
        self.own, self.opposite, self.method = own, opposite, method

    def run(self, seconds, seed, emit):
        if self.method == "guided":
            return Guided(self.own, self.opposite, "guided").run(seconds, seed, emit)
        if seconds <= 0 or not np.isfinite(seconds):
            raise ValueError("Positive finite budget required")
        started = time.perf_counter()
        deadline = started + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        counters = dict(parameters=CONFIG, preparation_seconds=time.perf_counter() - started, method=self.method)
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
        spaces = []
        counters["spaces"] = []
        n = self.own.shape[1]
        layouts = (
            [("full", list(range(n)))]
            if self.method.startswith("full")
            else [(f"block{half}", list(range(half * (n // 2), (half + 1) * (n // 2)))) for half in (0, 1)]
        )
        if self.method.startswith("block") and n % 2:
            raise ValueError("Two equal blocks required")
        for index, (label, coordinates) in enumerate(layouts):
            if time.perf_counter() >= deadline:
                break
            if label == "full":
                basis = prepared.kernel
                logical_rank = prepared.logicals.shape[0]
            else:
                restricted = restriction.Space(self.opposite, prepared.logicals, [[q] for q in coordinates])
                basis, logical_rank = restricted.basis, restricted.logical_rank
            entry = dict(label=label, coordinates=coordinates, dimension=len(basis), logical_rank=logical_rank)
            counters["spaces"].append(entry)
            if not logical_rank:
                entry["status"] = "no_logicals"
                continue
            if self.method == "block-exact" and len(basis) > CONFIG["enumeration_cap"]:
                entry["status"] = "dimension_cap"
                continue
            session = native.Session(
                basis, prepared.logicals, (seed + index * 104729) % (1 << 64), "stern" in self.method
            )
            entry["status"] = "active"
            entry["stats"] = session.stats
            spaces.append((session, entry))
        counters["setup_seconds"] = time.perf_counter() - tick
        number = 0
        while spaces and time.perf_counter() < deadline:
            session, entry = spaces[number % len(spaces)]
            number += 1
            until = min(deadline, time.perf_counter() + CONFIG["slice_seconds"])
            while time.perf_counter() < until:
                events = (
                    session.enumerate_chunk(CONFIG["enumeration_chunk"])
                    if self.method == "block-exact"
                    else session.advance(CONFIG["trials_per_batch"])
                )
                for weight, support, stage in events:
                    emit(weight, support, entry["label"] + ":" + stage)
                if self.method == "block-exact" and session.stats["enumeration_complete"]:
                    entry["status"] = "exhausted"
                    break
            entry["stats"] = session.stats
            if entry["status"] == "exhausted":
                spaces.remove((session, entry))
                number = 0
        return dict(counters, status="completed", elapsed_seconds=time.perf_counter() - started)
