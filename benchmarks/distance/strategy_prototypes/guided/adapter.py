"""Four-parent basis hill-climbing experiment; warm matrix preparation API."""

import importlib.util
from pathlib import Path
import time

import numpy as np


def _extension():
    paths = list(Path(__file__).parent.glob("ris_guided_native*.so"))
    if len(paths) != 1:
        raise RuntimeError("Build guided/setup.py build_ext --inplace first")
    spec = importlib.util.spec_from_file_location("ris_guided_native", paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = _extension()


class Prepared:
    def __init__(self, own, opposite):
        self.native = native.Prepared(np.asarray(own, dtype=np.uint8),
                                      np.asarray(opposite, dtype=np.uint8))

    def session(self, seed):
        return native.Session(self.native, threads=1, seed=seed, pair_depth=8,
                              block_size=6, restart_interval=64,
                              exchange_proposals=8)

    def run(self, seconds, seed, emit):
        start = time.perf_counter()
        deadline = start + seconds
        session = self.session(seed)
        trials = 0
        batch_size = 1
        batch = session.advance(0)
        while self.native.applicable and time.perf_counter() < deadline:
            batch_start = time.perf_counter()
            batch = session.advance(batch_size)
            batch_elapsed = time.perf_counter() - batch_start
            trials += batch.trials
            # Always deliver late witnesses; caller decides deadline credit.
            for event in batch.improvements:
                emit(event.weight, event.support)
            # Aim at 2 ms batches and cap each at 64 scored bases.
            remaining = max(0.0, deadline - time.perf_counter())
            batch_size = max(1, min(64, int(batch_size * min(.002, remaining) /
                                            max(batch_elapsed, 1e-9))))
        return {
            "trials": trials, "reductions": batch.reductions,
            "proposals": batch.proposals, "exchanges": batch.exchanges,
            "accepted_children": batch.accepted_children,
            "immigrants": batch.immigrants,
            "best_weight": batch.best_weight,
            "workspace_bytes": session.workspace_bytes,
            "elapsed_seconds": time.perf_counter() - start,
            "configuration": "elite4-exchange8-immigrant64-pairs8-block6",
        }


def prepare(own, opposite):
    return Prepared(own, opposite)
