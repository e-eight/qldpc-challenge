"""Bounded native structure inference, restricted search, and guided fallback."""

import math
import operator
import sys
import time

from run import benchmark_native, native_batch_seed, np

# isort: split
from candidate_search import Search as ControlSearch
from candidate_search import native_search
from common import ROOT
from initialized_search import LogicalPool, initialize, pack_rows, support_of
from study_strategies import ris_native

sys.path.insert(0, str(ROOT / "native/structure_dispatch"))
import structure_dispatch_native as detector  # noqa: E402

CONFIG = {
    "detector_seconds_cap": 0.050,
    "detector_budget_fraction": 0.02,
    "detector_max_columns": 4096,
    "detector_max_rows": 4096,
    "minimum_confidence": 0.50,
    "pilot_seconds_cap": 0.100,
    "pilot_budget_fraction": 0.10,
    "guided_reserve_fraction": 0.20,
    "batch_seconds": 0.050,
    "batch_cap": 4096,
    "pairs": 8,
}


class RoutedSession:
    def __init__(self, own, opposite, duals, order, seed, emit):
        self.order = np.asarray(order, dtype=np.int64)
        self.seed, self.emit = seed, emit
        self.prepared = benchmark_native.PreparedSearch(
            np.ascontiguousarray(own[:, self.order]),
            np.ascontiguousarray(opposite[:, self.order]),
            True,
            own.shape[1] // 2,
        )
        self.checks, self.duals = pack_rows(opposite), pack_rows(duals)
        self.batch_size = 1
        self.counters = dict(
            batches=0, scored_bases=0, exported=0, best=None, applicable=bool(self.prepared.applicable())
        )

    def advance_until(self, deadline):
        while self.counters["applicable"] and time.perf_counter() < deadline:
            number = self.counters["batches"]
            before = time.perf_counter()
            weight, returned, completed = self.prepared.batch(
                self.batch_size, native_batch_seed(self.seed, number), CONFIG["pairs"], 1, number
            )
            duration = time.perf_counter() - before
            self.counters["batches"] += 1
            self.counters["scored_bases"] += int(completed)
            if returned:
                support = sorted(int(self.order[q]) for q in returned)
                word = sum(1 << q for q in support)
                if len(set(support)) != len(support) or weight != word.bit_count():
                    raise RuntimeError("Malformed native routed support")
                if any((word & check).bit_count() & 1 for check in self.checks):
                    raise RuntimeError("Routed support has nonzero syndrome")
                if not any((word & dual).bit_count() & 1 for dual in self.duals):
                    raise RuntimeError("Routed support has trivial logical parity")
                # Persist every returned batch winner, including ties and late output.
                self.emit(int(weight), support, "routed_single_block")
                self.counters["exported"] += 1
                best = self.counters["best"]
                self.counters["best"] = int(weight) if best is None else min(best, int(weight))
            remaining = max(0, deadline - time.perf_counter())
            self.batch_size = max(
                1,
                min(
                    CONFIG["batch_cap"],
                    int(self.batch_size * min(CONFIG["batch_seconds"], remaining) / max(duration, 1e-9)),
                ),
            )


class Search:
    """Same callback/timed interface as the existing witness-preserving harness.

    No metadata, targets or reference supports enter this class. Deadline checks
    are cooperative; native kernel preparation and one batch are interrupt units.
    """

    def __init__(self, own, opposite, method="dispatch"):
        if method not in {"dispatch", "guided", "incremental", "circulant"}:
            raise ValueError("Unknown dispatcher/control method")
        self.own, self.opposite, self.method = own, opposite, method

    def run(self, seconds, seed, emit):
        seed = operator.index(seed)
        if not 0 <= seed < 1 << 64:
            raise ValueError("Seed must fit an unsigned 64-bit integer")
        if self.method != "dispatch":
            return ControlSearch(self.own, self.opposite, self.method, {}).run(seconds, seed, emit)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite budget required")
        started = time.perf_counter()
        deadline = started + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        counters = dict(parameters=CONFIG, preparation_seconds=time.perf_counter() - started)
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

        init_start = time.perf_counter()
        counters["initialization"] = initialize(rows, LogicalPool(prepared.logicals, 8), observe, deadline)
        counters["initialization_seconds"] = time.perf_counter() - init_start
        counters["initial_best"] = best
        if time.perf_counter() >= deadline:
            return dict(counters, status="initialization_exhausted_budget")
        detection_start = time.perf_counter()
        allowance = min(
            CONFIG["detector_seconds_cap"],
            CONFIG["detector_budget_fraction"] * seconds,
            max(0, deadline - detection_start),
        )
        detection_deadline = detection_start + allowance
        counters["detection_allowance_seconds"] = allowance
        counters["detection"] = []
        order = None
        for label, matrix in (("own", self.own), ("opposite", self.opposite)):
            if time.perf_counter() >= detection_deadline:
                break
            m, n = matrix.shape
            if n < 4 or n % 2 or n > CONFIG["detector_max_columns"] or m < 2 or m > CONFIG["detector_max_rows"]:
                counters["detection"].append(
                    dict(source=label, status="shape_cap", elapsed_seconds=0, workspace_bytes=0)
                )
                continue
            array = np.ascontiguousarray(matrix, dtype=np.uint8)
            result = detector.detect(
                array, max(0, detection_deadline - time.perf_counter()), CONFIG["minimum_confidence"]
            )
            candidate_order = result.pop("order")
            counters["detection"].append(dict(source=label, **result))
            if result["status"] == "accepted" and time.perf_counter() < detection_deadline:
                order = candidate_order
                counters["order"] = order  # exact route retained for replay/audit
                break
        counters["detection_seconds"] = time.perf_counter() - detection_start
        counters["route_decision"] = "no_partition"
        remaining = max(0, deadline - time.perf_counter())
        if order is not None and remaining > 0:
            route_start = time.perf_counter()
            route_deadline = deadline - CONFIG["guided_reserve_fraction"] * remaining
            counters["guided_reserved_seconds"] = CONFIG["guided_reserve_fraction"] * remaining
            counters["route_deadline_seconds"] = route_deadline - started
            route = RoutedSession(self.own, self.opposite, prepared.logicals, order, seed, emit)
            counters["route_setup_seconds"] = time.perf_counter() - route_start
            pilot_start = time.perf_counter()
            pilot_deadline = min(
                route_deadline,
                pilot_start + min(CONFIG["pilot_seconds_cap"], CONFIG["pilot_budget_fraction"] * remaining),
            )
            route.advance_until(pilot_deadline)
            counters["pilot_seconds"] = time.perf_counter() - pilot_start
            counters["pilot_best"] = route.counters["best"]
            if route.counters["best"] is not None and route.counters["best"] < best:
                counters["route_decision"] = "continue_improving_route"
                route.advance_until(route_deadline)
            else:
                counters["route_decision"] = "pilot_did_not_improve" if route.counters["applicable"] else "no_logicals"
            counters["route"] = route.counters
            counters["route_seconds"] = time.perf_counter() - route_start
            counters["route_overrun_seconds"] = max(0, time.perf_counter() - route_deadline)
        if time.perf_counter() < deadline:
            fallback_start = time.perf_counter()
            counters["fallback"] = native_search(self.own, self.opposite, prepared, "guided", deadline, seed, emit)
            counters["fallback_seconds"] = time.perf_counter() - fallback_start
        counters["elapsed_seconds"] = time.perf_counter() - started
        return dict(counters, status="completed")
