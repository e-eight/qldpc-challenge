"""Budgeted comparison of structural, reduced-space and decoder proposals."""

import math
import time

from run import benchmark_native, native_batch_seed

# isort: split
# run establishes numerical thread limits before numerical imports.
from initialized_search import GUIDED, LogicalPool, initialize, support_of
from study_strategies import load_adapter, ris_native

METHODS = ("incremental", "guided", "circulant", "matrix-structure", "reduced-space", "decoder")
ADAPTER_DIRS = {
    "matrix-structure": "matrix_structure_v2",
    "reduced-space": "reduced_space_v2",
    "decoder": "decoder_search_v2",
}


def load_candidates():
    return {name: load_adapter(directory) for name, directory in ADAPTER_DIRS.items()}


def native_search(own, opposite, prepared, method, deadline, seed, emit):
    started = time.perf_counter()
    session = (
        GUIDED.prepare(own, opposite).session(seed)
        if method == "guided"
        else ris_native.Session(prepared, 1, seed, block_size=6, restart_interval=64, exchange_proposals=8)
    )
    setup = time.perf_counter() - started
    batch_size, trials, result = 1, 0, None
    while time.perf_counter() < deadline:
        before = time.perf_counter()
        result = session.advance(batch_size)
        duration = time.perf_counter() - before
        trials += result.trials
        for event in result.improvements:
            emit(event.weight, list(event.support), method)
        remaining = max(0, deadline - time.perf_counter())
        batch_size = max(1, min(64, int(batch_size * min(0.002, remaining) / max(duration, 1e-9))))
    counters = dict(setup_seconds=setup, scored_bases=trials, workspace_bytes=session.workspace_bytes)
    if result is not None:
        for key in ("reductions", "proposals", "exchanges", "accepted_children", "immigrants"):
            if hasattr(result, key):
                counters[key] = getattr(result, key)
    return counters


def circulant_search(own, opposite, deadline, seed, emit):
    started = time.perf_counter()
    size = benchmark_native.circulant_size(own)
    if not size:
        return dict(applicable=False, setup_seconds=time.perf_counter() - started, scored_bases=0)
    prepared = benchmark_native.PreparedSearch(own, opposite, True, size)
    setup = time.perf_counter() - started
    if not prepared.applicable():
        return dict(applicable=False, setup_seconds=setup, scored_bases=0)
    batch_size, number, trials = 1, 0, 0
    while time.perf_counter() < deadline:
        before = time.perf_counter()
        weight, support, completed = prepared.batch(batch_size, native_batch_seed(seed, number), 8, 1, number)
        duration = time.perf_counter() - before
        trials += completed
        if support:
            emit(weight, list(support), "circulant")
        number += 1
        remaining = max(0, deadline - time.perf_counter())
        # The legacy API exports only its batch winner. Retain every return;
        # a 50-ms target limits repeated serialization of tied witnesses.
        batch_size = max(1, min(4096, int(batch_size * min(0.050, remaining) / max(duration, 1e-9))))
    return dict(applicable=True, setup_seconds=setup, scored_bases=trials, batches=number, batch_seconds=0.050)


class Search:
    def __init__(self, own, opposite, method, adapters):
        if method not in METHODS:
            raise ValueError("Unknown comparison method")
        self.own, self.opposite, self.method, self.adapters = own, opposite, method, adapters

    def run(self, seconds, seed, emit):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite budget required")
        started = time.perf_counter()
        deadline = started + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        counters = dict(preparation_seconds=time.perf_counter() - started)
        if not prepared.applicable:
            return dict(counters, status="not_applicable")
        init_start, best = time.perf_counter(), self.own.shape[1] + 1

        def observe(word, stage, improving_only=False):
            nonlocal best
            weight = word.bit_count()
            if improving_only and weight >= best:
                return
            best = min(best, weight)
            emit(weight, support_of(word), stage)

        pool = LogicalPool(prepared.logicals, 8)
        counters["initialization"] = initialize(rows, pool, observe, deadline)
        counters["initialization_seconds"] = time.perf_counter() - init_start
        counters["initial_best"] = best
        if time.perf_counter() >= deadline:
            return dict(counters, status="initialization_exhausted_budget")
        candidate_start = time.perf_counter()
        if self.method in ("guided", "incremental"):
            counters["native"] = native_search(self.own, self.opposite, prepared, self.method, deadline, seed, emit)
        elif self.method == "circulant":
            counters["candidate"] = circulant_search(self.own, self.opposite, deadline, seed, emit)
        else:
            counters["candidate"] = self.adapters[self.method].search(
                self.own,
                self.opposite,
                prepared.logicals,
                max(0, deadline - time.perf_counter()),
                seed,
                emit,
            )
        counters["candidate_seconds"] = time.perf_counter() - candidate_start
        if time.perf_counter() < deadline:
            fallback_start = time.perf_counter()
            counters["fallback"] = native_search(self.own, self.opposite, prepared, "guided", deadline, seed, emit)
            counters["fallback_seconds"] = time.perf_counter() - fallback_start
        counters["elapsed_seconds"] = time.perf_counter() - started
        return dict(counters, status="completed")
