"""Matched initialization and deadline adapters for native and external search."""

import importlib
import math
import resource
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

from common import HERE, atomic_json
from run import np, read_codewords, write_matrix

# isort: split
# run sets numerical thread limits before initialized_search imports NumPy.
from initialized_search import GUIDED, LogicalPool, initialize, support_of
from study_strategies import ris_native

METHODS = ("incremental", "guided", "m4ri", "qdistevol")
QD_ROOT = HERE / "cache/deps/codedistance"
M4RI_BINARY = HERE / "cache/deps/dist-m4ri/src/dist_m4ri"
QD_PARAMS = {
    "method": "QDistEvol",
    "iterCount": 1000000,
    "genCount": 10000,
    "offspring": 10,
    "pMut": 2.0,
    "sMut": 1.0,
    "pMutScale": 50,
    "tabuLength": 0,
    "swapPivot": 1,
    "swapBlockorder": 0,
    "regroupPerm": 1,
    "GF4blockRep": 1,
    "HL": False,
    "maxErr": -1,
}


@lru_cache(maxsize=1)
def qdist_module():
    sys.path.insert(0, str(QD_ROOT))
    return importlib.import_module("codedistance.distance")


class DeadlineReached(Exception):
    pass


def qdist_search(opposite, duals, seed, deadline, emit, params=None):
    """Observe the pinned upstream routine without changing its population."""
    module = qdist_module()
    original = module.permMinRowsK
    trials, best = 0, opposite.shape[1] + 1

    def observed(*args):
        nonlocal trials, best
        result = original(*args)
        trials += 1
        weight, rows = result[:2]
        if len(rows) and int(weight) < best:
            support = np.flatnonzero(rows[0]).tolist()
            if support:
                best = int(weight)
                # Persist before checking the deadline, retaining late evidence.
                emit(best, support, "search")
        if time.perf_counter() >= deadline:
            raise DeadlineReached
        return result

    effective = QD_PARAMS.copy() if params is None else params.copy()
    module.permMinRowsK = observed
    try:
        module.QDistEvol(opposite, duals, tB=1, params=effective, seed=seed)
    except DeadlineReached:
        status = "deadline"
    else:
        status = "iteration_limit"
    finally:
        module.permMinRowsK = original
    return {"status": status, "scored_bases": trials, "engine_best": best, "parameters": effective}


def warm_qdist():
    """JIT on a synthetic CSS toy, never on an evaluation matrix or witness."""
    own = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.uint8)
    opposite = np.array([[1, 1, 1, 1, 0]], dtype=np.uint8)
    duals = ris_native.Prepared(own, opposite).logicals
    started = time.perf_counter()
    qdist_module().QDistEvol(opposite, duals, tB=1, params=dict(QD_PARAMS, iterCount=200, genCount=2), seed=990)
    return time.perf_counter() - started


def m4ri_search(opposite, duals, directory, seed, deadline, emit, binary=M4RI_BINARY):
    """Unmodified CLI; charge input/export overhead to the common deadline."""
    directory.mkdir(parents=True, exist_ok=False)
    write_matrix(directory / "H.mtx", opposite)
    write_matrix(directory / "L.mtx", duals)
    remaining = deadline - time.perf_counter()
    reserve = 0.05
    if remaining <= reserve:
        return {"status": "setup_exhausted_budget", "scored_bases": 0}
    command = [
        str(binary),
        "method=1",
        f"finH={directory / 'H.mtx'}",
        f"finL={directory / 'L.mtx'}",
        "threads=1",
        f"timeout={remaining - reserve:.9f}",
        "steps=2147483647",
        f"seed={seed % 2147483647}",
        "wmin=0",
        f"outC={directory / 'codewords.txt'}",
        "debug=0",
        "dW=0",
    ]
    atomic_json(directory / "command.json", command)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (directory / "stderr.txt").open("w") as stderr:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=stderr,
            timeout=max(0, deadline - time.perf_counter()) + 120,
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (directory / "stdout.txt").write_bytes(result.stdout)
    numeric = [
        line.split()
        for line in result.stdout.decode().splitlines()
        if len(line.split()) == 3 and all(s.lstrip("-").isdigit() for s in line.split())
    ]
    words = read_codewords(directory / "codewords.txt")
    # Save every returned word before checking the CLI's aggregate metadata.
    for support in words:
        emit(len(support), support, "search")
    if len(numeric) != 1:
        raise ValueError("Unexpected M4RI summary; raw output and supports retained")
    lower, weight, trials = map(int, numeric[0])
    if weight > 0 and (not words or min(map(len, words)) != weight):
        raise ValueError("M4RI bound has no matching exported support")
    return {
        "status": "completed",
        "scored_bases": trials,
        "engine_best": weight,
        "unused_lower_bound": lower,
        "export_reserve_seconds": reserve,
        "subprocess_cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "delivery_protocol": "CLI exit; serialization and parsing are charged",
    }


class Search:
    def __init__(self, own, opposite, method, directory):
        if method not in METHODS:
            raise ValueError("Unknown comparison method")
        self.own, self.opposite, self.method = own, opposite, method
        self.directory = Path(directory)

    def run(self, seconds, seed, emit):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite budget required")
        started = time.perf_counter()
        deadline = started + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        counters = {"preparation_seconds": time.perf_counter() - started}
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

        duals = prepared.logicals
        pool = LogicalPool(duals, 8)
        counters["initialization"] = initialize(rows, pool, observe, deadline)
        counters["initialization_seconds"] = time.perf_counter() - init_start
        counters["initial_best"] = best
        if time.perf_counter() >= deadline:
            return dict(counters, status="initialization_exhausted_budget")
        search_start = time.perf_counter()
        if self.method == "qdistevol":
            counters.update(qdist_search(self.opposite, duals, seed, deadline, emit))
        elif self.method == "m4ri":
            counters.update(m4ri_search(self.opposite, duals, self.directory, seed, deadline, emit))
        else:
            session = (
                GUIDED.prepare(self.own, self.opposite).session(seed)
                if self.method == "guided"
                else ris_native.Session(prepared, 1, seed, block_size=6, restart_interval=64, exchange_proposals=8)
            )
            counters["session_setup_seconds"] = time.perf_counter() - search_start
            count, batch_size, batch = 0, 1, None
            while time.perf_counter() < deadline:
                before = time.perf_counter()
                batch = session.advance(batch_size)
                duration = time.perf_counter() - before
                count += batch.trials
                for event in batch.improvements:
                    emit(event.weight, event.support, "search")
                remaining = max(0, deadline - time.perf_counter())
                batch_size = max(1, min(64, int(batch_size * min(0.002, remaining) / max(duration, 1e-9))))
            counters.update(status="completed", scored_bases=count, workspace_bytes=session.workspace_bytes)
            if batch is not None:
                for key in ("reductions", "proposals", "exchanges", "accepted_children", "immigrants"):
                    if hasattr(batch, key):
                        counters[key] = getattr(batch, key)
        counters["backend_seconds"] = time.perf_counter() - search_start
        counters["elapsed_seconds"] = time.perf_counter() - started
        return counters
