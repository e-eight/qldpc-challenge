"""Resumable guided/structured competition with conservative time allocation."""

import math
import operator
import sys
import time

from run import np

# isort: split
from candidate_search import native_search
from common import ROOT
from dispatch_search import RoutedSession
from dispatch_search import Search as ControlSearch
from initialized_search import GUIDED, LogicalPool, initialize, support_of
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
    "exploration_fraction": 0.20,
    "route_advantage_fraction": 0.10,
    "allocation_epoch_seconds": 0.250,
    "batch_seconds": 0.050,
    "batch_cap": 4096,
    "pairs": 8,
}


def preferred(route_best, guided_best, initial_best):
    """Require a strict improvement and a 10% advantage to favor the route."""
    guided_best = min(initial_best, guided_best if guided_best is not None else initial_best)
    if (
        route_best is not None
        and route_best < initial_best
        and route_best <= (1 - CONFIG["route_advantage_fraction"]) * guided_best
    ):
        return "route"
    return "guided"


class GuidedSession:
    """Keep the native population, RNG and batch state across scheduling slices."""

    def __init__(self, own, opposite, seed, emit):
        self.session = GUIDED.prepare(own, opposite).session(seed)
        self.emit, self.batch_size = emit, 1
        self.counters = dict(
            best=None, exported=0, scored_bases=0, batches=0, workspace_bytes=self.session.workspace_bytes
        )

    def advance_until(self, deadline):
        while time.perf_counter() < deadline:
            before = time.perf_counter()
            result = self.session.advance(self.batch_size)
            duration = time.perf_counter() - before
            self.counters["batches"] += 1
            self.counters["scored_bases"] += result.trials
            for event in result.improvements:
                self.emit(event.weight, list(event.support), "guided")
                self.counters["exported"] += 1
                best = self.counters["best"]
                self.counters["best"] = event.weight if best is None else min(best, event.weight)
            remaining = max(0, deadline - time.perf_counter())
            self.batch_size = max(1, min(64, int(self.batch_size * min(0.002, remaining) / max(duration, 1e-9))))
            for key in ("reductions", "proposals", "exchanges", "accepted_children", "immigrants"):
                self.counters[key] = getattr(result, key)


def compete(own, opposite, duals, order, seed, emit, initial_best, started, deadline):
    counters = dict(route_decision="competing_sessions", slices=[], decisions=[])
    sessions, timely = {}, dict(route=None, guided=None)

    def callback(name):
        def receive(weight, support, stage):
            # Always export; only timely delivered outputs may influence policy.
            emit(weight, support, stage)
            if time.perf_counter() <= deadline:
                old = timely[name]
                timely[name] = weight if old is None else min(old, weight)

        return receive

    def advance(name, until, phase):
        begin = time.perf_counter()
        if begin >= deadline:
            return
        if name not in sessions:
            sessions[name] = (
                GuidedSession(own, opposite, seed, callback(name))
                if name == "guided"
                else RoutedSession(own, opposite, duals, order, seed, callback(name))
            )
        # Setup is charged to the slice. Never restart either search session.
        sessions[name].advance_until(min(until, deadline))
        counters["slices"].append(
            dict(
                engine=name,
                phase=phase,
                start_seconds=begin - started,
                deadline_seconds=min(until, deadline) - started,
                end_seconds=time.perf_counter() - started,
                best=timely[name],
                batches=sessions[name].counters["batches"],
            )
        )

    remaining = max(0, deadline - time.perf_counter())
    pilot = min(CONFIG["pilot_seconds_cap"], CONFIG["pilot_budget_fraction"] * remaining)
    counters["pilot_allowance_seconds"] = pilot
    for name in ("guided", "route"):
        advance(name, min(deadline, time.perf_counter() + pilot), "pilot")
    while time.perf_counter() < deadline:
        begin = time.perf_counter()
        winner = preferred(timely["route"], timely["guided"], initial_best)
        loser = "guided" if winner == "route" else "route"
        epoch = min(CONFIG["allocation_epoch_seconds"], deadline - begin)
        counters["decisions"].append(
            dict(
                seconds=begin - started,
                route_best=timely["route"],
                guided_best=timely["guided"],
                preferred=winner,
                epoch_seconds=epoch,
            )
        )
        # Explore first so a final overrun cannot erase the minority's entire slice.
        advance(loser, begin + CONFIG["exploration_fraction"] * epoch, "explore")
        advance(winner, begin + epoch, "exploit")
    counters["sessions"] = {name: session.counters for name, session in sessions.items()}
    counters["timely_best"] = timely
    return counters


class Search:
    """Same callback/timed interface as the existing witness-preserving harness.

    No metadata, targets or reference supports enter this class. Deadline checks
    are cooperative; native kernel preparation and one batch are interrupt units.
    """

    def __init__(self, own, opposite, method="race"):
        if method not in {"race", "dispatch", "guided"}:
            raise ValueError("Unknown dispatcher/control method")
        self.own, self.opposite, self.method = own, opposite, method

    def run(self, seconds, seed, emit):
        seed = operator.index(seed)
        if not 0 <= seed < 1 << 64:
            raise ValueError("Seed must fit an unsigned 64-bit integer")
        if self.method != "race":
            return ControlSearch(self.own, self.opposite, self.method).run(seconds, seed, emit)
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
        if order is None:
            if time.perf_counter() < deadline:
                fallback_start = time.perf_counter()
                counters["fallback"] = native_search(self.own, self.opposite, prepared, "guided", deadline, seed, emit)
                counters["fallback_seconds"] = time.perf_counter() - fallback_start
        else:
            counters.update(
                compete(self.own, self.opposite, prepared.logicals, order, seed, emit, best, started, deadline)
            )
        counters["elapsed_seconds"] = time.perf_counter() - started
        return dict(counters, status="completed")
