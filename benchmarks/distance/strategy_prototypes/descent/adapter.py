"""Cold RIS seeding followed by native stabilizer tabu descent.

All seeding and local refinement is charged to run's elapsed-time budget.
Preparation is intentionally outside that budget, like the other warm adapters.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for path in (HERE, ROOT / "native" / "ris"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ris_native  # noqa: E402
from _stabilizer_descent import Descent  # noqa: E402

CONFIG = {
    "seed_bases": 128,
    "block_size": 6,
    "restart_interval": 64,
    "exchange_proposals": 8,
    "descent_slice_seconds": 0.005,
    "tabu_tenure": 7,
    "stagnation_steps": 64,
    "weight_excursion": 8,
    "kick_rows": [1, 3],
}


class Prepared:
    def __init__(self, own, opposite):
        started = time.perf_counter()
        self.ris = ris_native.Prepared(own, opposite)
        self.descent = Descent(np.asarray(own, dtype=np.uint8))
        self.n = own.shape[1]
        self.prepare_seconds = time.perf_counter() - started

    def run(self, seconds, seed, emit):
        started = time.perf_counter()
        deadline = started + seconds
        counters = dict(
            seed_sessions=0, seed_bases=0, ris_seconds=0.0,
            descent_seconds=0.0, steps=0, restarts=0,
            uphill_moves=0, neutral_moves=0, descent_improvements=0,
            global_descent_improvements=0,
            emitted=0, prepare_seconds=self.prepare_seconds, config=CONFIG.copy(),
        )
        if not self.ris.applicable:
            counters["elapsed_seconds"] = time.perf_counter() - started
            return counters
        best = self.n + 1
        # Independent child seeds avoid repeatedly refining the same global-best
        # logical class. A fresh Session exposes its own best seeding witness.
        rng = np.random.default_rng(seed)
        while time.perf_counter() < deadline:
            ris_start = time.perf_counter()
            session = ris_native.Session(
                self.ris, threads=1, seed=int(rng.bit_generator.random_raw()),
                block_size=6, restart_interval=64, exchange_proposals=8,
            )
            batch = session.advance(CONFIG["seed_bases"])
            counters["ris_seconds"] += time.perf_counter() - ris_start
            counters["seed_sessions"] += 1
            counters["seed_bases"] += batch.trials
            support = None
            for improvement in batch.improvements:
                support = improvement.support
                # Persist every seeding improvement, including useful evidence
                # from a different logical class that exceeds the global best.
                emit(improvement.weight, support)
                counters["emitted"] += 1
                best = min(best, improvement.weight)
            remaining = deadline - time.perf_counter()
            if support is not None and remaining > 0:
                local_start = time.perf_counter()
                result = self.descent.run(
                    support, min(CONFIG["descent_slice_seconds"], remaining),
                    int(rng.bit_generator.random_raw()),
                )
                counters["descent_seconds"] += time.perf_counter() - local_start
                for key in ("steps", "restarts", "uphill_moves", "neutral_moves"):
                    counters[key] += result[key]
                for weight, word in result["improvements"]:
                    emit(weight, word)
                    counters["emitted"] += 1
                    counters["descent_improvements"] += 1
                    counters["global_descent_improvements"] += weight < best
                    best = min(best, weight)
        counters["best_weight"] = best
        counters["elapsed_seconds"] = time.perf_counter() - started
        return counters


def prepare(own, opposite):
    return Prepared(own, opposite)
