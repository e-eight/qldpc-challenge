"""Five leading search engines with common initialization and live checkpoints."""

import os
import resource
import selectors
import subprocess
import time

from run import benchmark_native, native_batch_seed, np, read_codewords, write_matrix

# isort: split
from common import atomic_json
from full_suite.build_observer import DEST, MARKER
from initialized_search import GUIDED, LogicalPool, initialize, support_of
from study_strategies import ris_native

METHODS = ("cpp", "fresh", "incremental", "guided", "m4ri")
CONFIG = dict(
    batch_seconds=0.002,
    batch_cap=64,
    pair_depth=8,
    block_size=6,
    restart_interval=64,
    exchange_proposals=8,
    structure_seconds=2.0,
)


def m4ri(opposite, duals, directory, seed, deadline, emit, steps=2147483647):
    directory.mkdir(parents=True, exist_ok=False)
    write_matrix(directory / "H.mtx", opposite)
    write_matrix(directory / "L.mtx", duals)
    remaining = deadline - time.perf_counter()
    if remaining <= 0.05:
        return dict(status="setup_exhausted_budget", scored_bases=0)
    command = [
        str(DEST / "dist_m4ri"),
        "method=1",
        f"finH={directory / 'H.mtx'}",
        f"finL={directory / 'L.mtx'}",
        "threads=1",
        f"timeout={remaining - 0.05:.9f}",
        f"steps={steps}",
        f"seed={seed % 2147483647}",
        "wmin=0",
        f"outC={directory / 'codewords.txt'}",
        "debug=0",
        "dW=0",
    ]
    atomic_json(directory / "command.json", command)
    cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    exports, best, buffer = 0, None, b""
    with (directory / "stdout.txt").open("wb") as stdout, (directory / "stderr.txt").open("wb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=subprocess.PIPE)
        selector = selectors.DefaultSelector()
        selector.register(proc.stderr, selectors.EVENT_READ)
        try:
            eof = False
            while not eof:
                if time.perf_counter() > deadline + 2:
                    raise TimeoutError("dist-m4ri exceeded deadline grace; raw witnesses retained")
                for key, _ in selector.select(0.1):
                    data = os.read(key.fd, 65536)
                    if not data:
                        eof = True
                        break
                    stderr.write(data)
                    stderr.flush()
                    buffer += data
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.startswith((MARKER + " ").encode()):
                            continue
                        values = list(map(int, line.split()[1:]))
                        weight, support = values[0], values[1:]
                        if len(support) != weight or (best is not None and weight >= best):
                            raise ValueError("Malformed/nonimproving M4RI observation")
                        emit(weight, support, "search")
                        best, exports = weight, exports + 1
            if proc.wait(timeout=2):
                raise RuntimeError("dist-m4ri failed; see raw stderr")
        finally:
            selector.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stderr.close()
            stderr.flush()
            os.fsync(stderr.fileno())
    cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    lines = [
        line.split()
        for line in (directory / "stdout.txt").read_text().splitlines()
        if len(line.split()) == 3 and all(s.lstrip("-").isdigit() for s in line.split())
    ]
    if len(lines) != 1:
        raise ValueError("Unexpected M4RI summary")
    lower, weight, trials = map(int, lines[0])
    words = read_codewords(directory / "codewords.txt")
    if weight > 0 and (best != weight or not words or min(map(len, words)) != weight):
        raise ValueError("M4RI final bound differs from observed/saved supports")
    return dict(
        status="completed",
        scored_bases=trials,
        engine_best=best,
        exported=exports,
        unused_lower_bound=lower,
        final_tied_codewords=len(words),
        subprocess_cpu_seconds=cpu1.ru_utime + cpu1.ru_stime - cpu0.ru_utime - cpu0.ru_stime,
        delivery_protocol="observer-only binary; parent receipt clock; final tied supports archived",
    )


class Search:
    def __init__(self, own, opposite, method, directory):
        if method not in METHODS + ("circulant",):
            raise ValueError("Unknown method")
        self.own, self.opposite, self.method, self.directory = own, opposite, method, directory

    def run(self, seconds, seed, emit):
        start = time.perf_counter()
        deadline = start + seconds
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite budget required")
        if self.method == "circulant":
            tick = time.perf_counter()
            session = benchmark_native.PreparedSearch(self.own, self.opposite, True)
            if not session.applicable():
                return dict(status="not_applicable", detection_seconds=time.perf_counter() - tick)
            stats = dict(preparation_seconds=time.perf_counter() - tick)
            # This separate existing structural pass deliberately has no common
            # initialization: attribution is exclusively the structural engine.
        else:
            prepared = ris_native.Prepared(self.own, self.opposite)
            rows = ris_native.Prepared(self.opposite, self.own).logicals
            stats = dict(parameters=CONFIG, preparation_seconds=time.perf_counter() - start)
            if not prepared.applicable:
                return dict(stats, status="not_applicable")
            best = self.own.shape[1] + 1

            def observe(word, stage, improving_only=False):
                nonlocal best
                weight = word.bit_count()
                if improving_only and weight >= best:
                    return
                best = min(best, weight)
                emit(weight, support_of(word), stage)

            tick = time.perf_counter()
            stats["initialization"] = initialize(rows, LogicalPool(prepared.logicals, 8), observe, deadline)
            stats["initialization_seconds"] = time.perf_counter() - tick
            stats["initial_best"] = best
            if time.perf_counter() >= deadline:
                return dict(stats, status="initialization_exhausted_budget")
            tick = time.perf_counter()
            if self.method == "m4ri":
                return dict(stats, **m4ri(self.opposite, prepared.logicals, self.directory, seed, deadline, emit))
            if self.method == "cpp":
                session = benchmark_native.PreparedSearch(self.own, self.opposite)
            elif self.method == "guided":
                session = GUIDED.prepare(self.own, self.opposite).session(seed)
            else:
                session = ris_native.Session(
                    prepared,
                    1,
                    seed,
                    block_size=6,
                    restart_interval=64 if self.method == "incremental" else 0,
                    exchange_proposals=8,
                )
            stats["session_setup_seconds"] = time.perf_counter() - tick
        batch_size, batches, count, best, exports = 1, 0, 0, self.own.shape[1] + 1, 0
        batch = None
        while time.perf_counter() < deadline:
            tick = time.perf_counter()
            if self.method in ("cpp", "circulant"):
                weight, support, trials = session.batch(batch_size, native_batch_seed(seed, batches), 8, 1, batches)
                events = [(weight, support)] if support and weight < best else []
                count += trials
            else:
                batch = session.advance(batch_size)
                events = [(e.weight, list(e.support)) for e in batch.improvements]
                count += batch.trials
            elapsed = time.perf_counter() - tick
            batches += 1
            for weight, support in events:
                best = min(best, weight)
                emit(weight, support, "search")
                exports += 1
            remaining = max(0, deadline - time.perf_counter())
            batch_size = max(1, min(CONFIG["batch_cap"], int(batch_size * min(0.002, remaining) / max(elapsed, 1e-9))))
        if batch is not None:
            for key in ("reductions", "proposals", "exchanges", "accepted_children", "immigrants"):
                if hasattr(batch, key):
                    stats[key] = getattr(batch, key)
        return dict(
            stats,
            status="completed",
            scored_bases=count,
            batches=batches,
            exported=exports,
            engine_best=best if best <= self.own.shape[1] else None,
            elapsed_seconds=time.perf_counter() - start,
        )
