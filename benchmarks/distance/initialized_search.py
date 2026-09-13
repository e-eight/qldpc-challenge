"""Common logical initialization and a budgeted guided/refinement experiment.

The native search kernels are unchanged. This module owns the experimental
policy and charges preparation, initialization, and refinement to one clock.
"""

import math
import time

import numpy as np
from study_strategies import load_adapter, ris_native

GUIDED = load_adapter("guided")
DESCENT = load_adapter("descent")
METHODS = ("incremental", "guided", "guided-refine")
CONFIG = {
    "pool_size": 8,
    "combination_limit": 16,
    "refinement_fraction": 0.10,
    "refinement_slice": 0.005,
    "batch_seconds": 0.002,
    "batch_cap": 64,
}


def pack_rows(rows):
    return [int.from_bytes(row.tobytes(), "little") for row in np.packbits(rows, axis=1, bitorder="little")]


def support_of(word):
    result = []
    while word:
        bit = word & -word
        result.append(bit.bit_length() - 1)
        word ^= bit
    return result


def word_of(support):
    word = 0
    for q in support:
        word |= 1 << int(q)
    return word


class LogicalPool:
    """Keep low-weight representatives of distinct logical parity signatures."""

    def __init__(self, duals, capacity=8):
        self.duals = pack_rows(duals)
        self.capacity = capacity
        self.entries = {}
        self.attempted = set()

    def signature(self, word):
        return sum(((word & dual).bit_count() & 1) << i for i, dual in enumerate(self.duals))

    def offer(self, word):
        weight = word.bit_count()
        if len(self.entries) == self.capacity and weight > max(w.bit_count() for w in self.entries.values()):
            return
        signature = self.signature(word)
        if not signature:
            raise ValueError("A stabilizer was offered to the logical pool")
        old = self.entries.get(signature)
        if old is not None and (old.bit_count(), old) <= (weight, word):
            return
        self.entries[signature] = word
        if len(self.entries) > self.capacity:
            worst = max(self.entries, key=lambda key: (self.entries[key].bit_count(), self.entries[key]))
            del self.entries[worst]

    def take(self):
        available = [word for word in self.entries.values() if word not in self.attempted]
        if not available:
            return None
        word = min(available, key=lambda w: (w.bit_count(), w))
        self.attempted.add(word)
        return word


def initialize(rows, pool, observe, deadline):
    """Score singles, then all combinations for small k or all pairs otherwise.

    Only improving witnesses and the final diverse seed pool are exported;
    enumeration intermediates are internal, as in native basis scoring.
    """
    values = pack_rows(rows)
    examined = 0
    complete = True

    def visit(word):
        nonlocal examined
        examined += 1
        observe(word, "initialization", improving_only=True)
        pool.offer(word)

    for word in values:
        if time.perf_counter() >= deadline:
            complete = False
            break
        visit(word)
    if complete and len(values) <= CONFIG["combination_limit"]:
        word = 0
        for index in range(1, 1 << len(values)):
            if time.perf_counter() >= deadline:
                complete = False
                break
            word ^= values[(index & -index).bit_length() - 1]
            # Singles were scored first so a short budget always sees them.
            gray = index ^ (index >> 1)
            if gray & (gray - 1):
                visit(word)
    elif complete:
        for i, word in enumerate(values):
            for other in values[i + 1 :]:
                if time.perf_counter() >= deadline:
                    complete = False
                    break
                visit(word ^ other)
            if not complete:
                break
    for word in sorted(pool.entries.values(), key=lambda w: (w.bit_count(), w)):
        observe(word, "initial_seed")
    return {"examined": examined, "complete": complete, "classes": len(pool.entries)}


def refinement_allowance(elapsed, spent, remaining):
    """Earn refinement time gradually; do not front-load a short run."""
    credit = CONFIG["refinement_fraction"] * elapsed - spent
    quantum = CONFIG["refinement_slice"]
    return quantum if min(credit, remaining) >= quantum else 0.0


class Search:
    def __init__(self, own, opposite, method):
        if method not in METHODS:
            raise ValueError("Unknown initialized search method")
        self.own, self.opposite, self.method = own, opposite, method

    def run(self, seconds, seed, emit):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Expected a finite nonnegative budget")
        started = time.perf_counter()
        deadline = started + seconds
        best = self.own.shape[1] + 1
        counters = {
            "config": CONFIG.copy(),
            "method": self.method,
            "scored_bases": 0,
            "preparation_seconds": 0.0,
            "initialization_seconds": 0.0,
            "ris_seconds": 0.0,
            "descent_seconds": 0.0,
            "descent_calls": 0,
            "descent_improvements": 0,
            "global_descent_improvements": 0,
            "steps": 0,
            "refinement_seeds": [],
        }

        def observe(word, stage, improving_only=False):
            nonlocal best
            weight = word.bit_count()
            improves = weight < best
            if improving_only and not improves:
                return
            best = min(best, weight)
            emit(weight, support_of(word), stage)
            if stage == "refinement":
                counters["descent_improvements"] += 1
                counters["global_descent_improvements"] += improves

        if seconds == 0:
            return dict(counters, status="zero_budget", best_weight=best)
        # Each side is standalone. Both matrix preparations are charged here;
        # a future paired-side API could share them to remove duplicated work.
        prepared = ris_native.Prepared(self.own, self.opposite)
        representatives = ris_native.Prepared(self.opposite, self.own).logicals
        counters["preparation_seconds"] = time.perf_counter() - started
        if not prepared.applicable:
            return dict(counters, status="not_applicable", best_weight=best)
        init_start = time.perf_counter()
        pool = LogicalPool(prepared.logicals, CONFIG["pool_size"])
        counters["initialization"] = initialize(representatives, pool, observe, deadline)
        counters["initialization_seconds"] = time.perf_counter() - init_start
        counters["initial_best"] = best
        if time.perf_counter() >= deadline:
            return dict(counters, status="initialization_exhausted_budget", best_weight=best)

        session_start = time.perf_counter()
        if self.method == "incremental":
            session = ris_native.Session(prepared, 1, seed, block_size=6, restart_interval=64, exchange_proposals=8)
        else:
            session = GUIDED.prepare(self.own, self.opposite).session(seed)
        descent = DESCENT.Descent(self.own) if self.method == "guided-refine" else None
        counters["session_setup_seconds"] = time.perf_counter() - session_start
        # A separate RNG ensures descent never consumes the basis-search stream.
        rng = np.random.default_rng(seed ^ 0x6A09E667F3BCC909)
        search_start = time.perf_counter()
        batch_size = 1
        batch = None
        while time.perf_counter() < deadline:
            before = time.perf_counter()
            batch = session.advance(batch_size)
            duration = time.perf_counter() - before
            counters["ris_seconds"] += duration
            counters["scored_bases"] += batch.trials
            for event in batch.improvements:
                word = word_of(event.support)
                observe(word, "search")
                pool.offer(word)
            now = time.perf_counter()
            allowance = refinement_allowance(now - search_start, counters["descent_seconds"], deadline - now)
            if descent is not None and allowance:
                word = pool.take()
                if word is not None:
                    # Initial pool members and search events have already been
                    # exported. Local histories are exported in full below.
                    local_seed = int(rng.bit_generator.random_raw())
                    counters["refinement_seeds"].append(
                        {
                            "support": support_of(word),
                            "seed": local_seed,
                            "seconds": time.perf_counter() - started,
                        }
                    )
                    before = time.perf_counter()
                    result = descent.run(support_of(word), allowance, local_seed)
                    counters["descent_seconds"] += time.perf_counter() - before
                    counters["descent_calls"] += 1
                    counters["steps"] += result["steps"]
                    for weight, support in result["improvements"]:
                        refined = word_of(support)
                        if refined.bit_count() != weight:
                            raise ValueError("Invalid refinement weight")
                        observe(refined, "refinement")
                        pool.offer(refined)
            remaining = max(0.0, deadline - time.perf_counter())
            batch_size = max(
                1,
                min(
                    CONFIG["batch_cap"], int(batch_size * min(CONFIG["batch_seconds"], remaining) / max(duration, 1e-9))
                ),
            )
        if batch is not None:
            for key in ("reductions", "proposals", "exchanges", "accepted_children", "immigrants"):
                if hasattr(batch, key):
                    counters[key] = getattr(batch, key)
        return dict(
            counters,
            best_weight=best,
            workspace_bytes=session.workspace_bytes,
            elapsed_seconds=time.perf_counter() - started,
            status="completed",
        )
