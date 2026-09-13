"""Timed full-kernel CRT versus binary grouping experiment."""

import sys
import time
from pathlib import Path

from run import np

# isort: split
from allocation_search import Search as Guided
from initialized_search import LogicalPool, initialize, pack_rows, support_of
from strategy_prototypes.polynomial_weight.module import build, packed_with_tags
from study_strategies import ris_native

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polynomial_weight_native as native  # noqa: E402

METHODS = ("guided", "crt-groups", "binary-groups")
CONFIG = dict(group_cap=10, batch=1, max_passes=8, stall_passes=2, uniform_restart_period=4)


class Search:
    def __init__(self, own, opposite, method, structure):
        if method not in METHODS:
            raise ValueError("Unknown method")
        self.own, self.opposite, self.method, self.structure = own, opposite, method, structure

    def run(self, seconds, seed, emit):
        if self.method == "guided":
            return Guided(self.own, self.opposite, "guided").run(seconds, seed, emit)
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite budget required")
        start = time.perf_counter()
        deadline = start + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        stats = dict(parameters=CONFIG, preparation_seconds=time.perf_counter() - start)
        best = self.own.shape[1] + 1

        def observe(word, stage, improving_only=False):
            nonlocal best
            weight = word.bit_count()
            if improving_only and weight >= best:
                return
            best = min(best, weight)
            emit(weight, support_of(word), stage)

        tick = time.perf_counter()
        pool = LogicalPool(prepared.logicals, 8)
        stats["initialization"] = initialize(rows, pool, observe, deadline)
        stats["initialization_seconds"] = time.perf_counter() - tick
        stats["initial_best"] = best
        if time.perf_counter() >= deadline:
            return dict(stats, status="initialization_exhausted_budget")
        tick = time.perf_counter()
        basis, widths, groups = build(
            self.opposite,
            self.structure["length"],
            [int(f, 16) for f in self.structure["factors"]],
            CONFIG["group_cap"],
        )
        stats.update(groups=groups, dimension=len(basis), module_seconds=time.perf_counter() - tick)
        if self.method == "binary-groups":
            order = np.random.default_rng(seed ^ 0xBB67AE8584CAA73B).permutation(len(basis))
            basis = prepared.kernel[order]
            stats["binary_order"] = order.tolist()
        tick = time.perf_counter()
        seeds = sorted(pool.entries.values(), key=lambda w: (w.bit_count(), w))
        packed = packed_with_tags(pack_rows(basis), prepared.logicals, self.own.shape[1])
        packed_seeds = packed_with_tags(seeds, prepared.logicals, self.own.shape[1])
        session = native.Session(packed, widths, packed_seeds, self.own.shape[1], len(prepared.logicals), seed)
        stats["table_setup_seconds"] = time.perf_counter() - tick
        stats["setup_finished_seconds"] = time.perf_counter() - start
        tick = time.perf_counter()
        while time.perf_counter() < deadline:
            for weight, support in session.advance(CONFIG["batch"]):
                emit(weight, support, self.method)
        stats["optimizer_seconds"] = time.perf_counter() - tick
        stats["native"] = session.stats
        return dict(stats, status="completed", elapsed_seconds=time.perf_counter() - start)
