"""Compare isolated strategy prototypes with a shared, witness-preserving clock."""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import random
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from run import (
    benchmark_native,
    initialize_validation,
    native_batch_seed,
    np,
    validate_and_stage,
    validation_ready,
)

sys.path.insert(0, str(ROOT / "native" / "ris"))
import ris_native

PROTOTYPES = HERE / "strategy_prototypes"
CASES = [
    "board-700-222-28",
    "board-682-172-79",
    "regression-690-182",
    "tanner-432_8_33",
    "mitten-975-195",
    "toric-1000",
    "fresh-bb-960",
]
METHODS = ["fresh", "incremental", "circulant", "guided", "descent", "structure"]


class NativeControl:
    def __init__(self, own, opposite, incremental):
        self.prepared = ris_native.Prepared(own, opposite)
        self.incremental = incremental

    def run(self, seconds, seed, emit):
        start = time.perf_counter()
        session = ris_native.Session(
            self.prepared,
            1,
            seed,
            block_size=6,
            restart_interval=64 if self.incremental else 0,
            exchange_proposals=8,
        )
        batch, count = 1, 0
        counters = {}
        while self.prepared.applicable and time.perf_counter() - start < seconds:
            before = time.perf_counter()
            result = session.advance(batch)
            duration = time.perf_counter() - before
            count += result.trials
            for event in result.improvements:
                emit(event.weight, event.support)
            counters = {key: getattr(result, key) for key in ("reductions", "proposals", "exchanges")}
            remaining = max(0, seconds - (time.perf_counter() - start))
            batch = max(1, min(4096, int(batch * min(0.005, remaining) / max(duration, 1e-9))))
        return dict(
            status="completed" if self.prepared.applicable else "not_applicable",
            scored_bases=count,
            workspace_bytes=session.workspace_bytes,
            **counters,
        )


class CirculantControl:
    def __init__(self, own, opposite, block_size):
        self.prepared = benchmark_native.PreparedSearch(own, opposite, True, block_size)

    def run(self, seconds, seed, emit):
        if not self.prepared.applicable():
            return {"status": "not_applicable", "scored_bases": 0}
        start = time.perf_counter()
        batch, number, count, best = 1, 0, 0, float("inf")
        while time.perf_counter() - start < seconds:
            before = time.perf_counter()
            weight, support, completed = self.prepared.batch(batch, native_batch_seed(seed, number), 8, 1, number)
            duration = time.perf_counter() - before
            count += completed
            if support:
                # This existing API exports a batch winner, not every internal improvement.
                emit(weight, support)
                best = min(best, weight)
            number += 1
            remaining = max(0, seconds - (time.perf_counter() - start))
            batch = max(1, min(4096, int(batch * min(0.005, remaining) / max(duration, 1e-9))))
        return {"status": "completed", "scored_bases": count, "batches": number}


def load_adapter(name):
    path = PROTOTYPES / name / "adapter.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"strategy_{name}_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(events, seconds, target):
    timely = [event for event in events if event["seconds"] <= seconds]
    hits = [event["seconds"] for event in timely if target is not None and event["weight"] <= target]
    return {
        "best_in_budget": min((event["weight"] for event in timely), default=None),
        "best_returned": min((event["weight"] for event in events), default=None),
        "target_hit": bool(hits),
        "time_to_target": min(hits, default=None),
        "late_events": sum(event["seconds"] > seconds for event in events),
    }


def run_side(engine, seconds, seed, path):
    events = []
    with path.open("x") as stream:
        start, cpu_start = time.perf_counter(), time.process_time()

        def emit(weight, support):
            event = {
                "seconds": time.perf_counter() - start,
                "weight": int(weight),
                "support": [int(q) for q in support],
            }
            events.append(event)
            stream.write(json.dumps(event) + "\n")
            stream.flush()

        try:
            counters = engine.run(seconds, seed, emit)
        finally:
            stream.flush()
            os.fsync(stream.fileno())
        elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_start
    return {
        "events": events,
        "search_seconds": elapsed,
        "cpu_seconds": cpu,
        "status": counters.get("status", "completed"),
        "counters": counters,
    }


def snapshot(output, methods):
    files = [Path(__file__), HERE / "common.py", HERE / "run.py", HERE / "native.cpp"]
    for base in [
        ROOT / "native" / "ris",
        *[PROTOTYPES / m for m in methods if m in ("guided", "descent", "structure")],
    ]:
        files.extend(
            p
            for p in base.rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".cpp", ".hpp", ".md", ".toml", ".txt")
            and not {"build", "__pycache__", ".pytest_cache"}.intersection(p.relative_to(base).parts)
        )
    files.extend([ROOT / "research" / "kit" / "submit.py"])
    hashes = {}
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for path in sorted(set(files)):
            relative = str(path.relative_to(ROOT))
            hashes[relative] = sha256(path)
            archive.add(path, arcname=relative)
    binaries = [Path(ris_native.__file__), Path(benchmark_native.__file__)]
    for method in methods:
        binaries.extend((PROTOTYPES / method).glob("*.so"))
    return hashes, {str(p.relative_to(ROOT)): sha256(p) for p in binaries}


def main(args):
    if args.seconds <= 0 or args.seeds < 1 or args.validation_workers < 1:
        raise ValueError("Positive budget and worker/seed counts required")
    if len(set(args.methods)) != len(args.methods) or len(set(args.cases)) != len(args.cases):
        raise ValueError("Duplicate methods/cases are not allowed")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    indexed = {case["id"]: case for case in manifest["cases"]}
    cases = [indexed[name] for name in args.cases]
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "manifest.json", {"sources": manifest["sources"], "cases": cases})
    available_cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu})
    adapters = {name: load_adapter(name) for name in args.methods if name in ("guided", "descent", "structure")}
    sources, binaries = snapshot(args.output, args.methods)
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": args.cpu,
        "validation_cpus": available_cpus,
        "threads": 1,
        "seconds_per_code": args.seconds,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "methods": args.methods,
        "cases": args.cases,
        "source_hashes": sources,
        "binary_hashes": binaries,
        "timing": "Warm matrices; cold search state; equal X/Z split; no target stopping; delivery-time scoring",
        "method_order": "shuffled within each case and seed",
        "random_seed_policy": "side seed = index * 1000003 + (0 for X, 499979 for Z)",
        "reference_policy": "Reference supports never passed to search; packaging uses a fresh logical-basis fallback",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(args.output / "environment.json", environment)
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.validation_workers)
    records = []
    with ProcessPoolExecutor(
        args.validation_workers,
        mp_context=context,
        initializer=initialize_validation,
        initargs=(available_cpus, barrier),
    ) as executor:
        futures = [executor.submit(validation_ready) for _ in range(args.validation_workers)]
        for future in futures:
            future.result()
        for case in cases:
            with np.load(args.corpus / case["file"], allow_pickle=False) as data:
                hx, hz = data["hx"], data["hz"]
            if matrix_hash(hx, hz) != case["matrix_sha256"]:
                raise ValueError("Corpus hash mismatch")
            matrix_dir = args.output / "matrices"
            matrix_dir.mkdir(exist_ok=True)
            np.savez_compressed(matrix_dir / case["file"], hx=hx, hz=hz)
            duals = {"X": ris_native.Prepared(hx, hz).logicals, "Z": ris_native.Prepared(hz, hx).logicals}
            fallback = {"X": np.flatnonzero(duals["Z"][0]).tolist(), "Z": np.flatnonzero(duals["X"][0]).tolist()}
            block_size = benchmark_native.circulant_size(hx)
            engines, setup = {}, {}
            for method in args.methods:
                engines[method], setup[method] = {}, {}
                for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
                    before = time.perf_counter()
                    if method in ("fresh", "incremental"):
                        engine = NativeControl(own, opposite, method == "incremental")
                    elif method == "circulant":
                        engine = CirculantControl(own, opposite, block_size)
                    else:
                        engine = adapters[method].prepare(own, opposite)
                    engines[method][side] = engine
                    setup[method][side] = time.perf_counter() - before
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                methods = list(args.methods)
                random.Random(f"strategy:{case['id']}:{seed}").shuffle(methods)
                for method in methods:
                    directory = args.output / case["id"] / f"{method}-s{seed}"
                    directory.mkdir(parents=True)
                    sides = {}
                    for side in ("X", "Z"):
                        side_seed = seed * 1000003 + (499979 if side == "Z" else 0)
                        sides[side] = [
                            run_side(engines[method][side], args.seconds / 2, side_seed, directory / f"{side}.jsonl")
                        ]
                    record = {
                        "case": case["id"],
                        "method": method,
                        "seed": seed,
                        "budget_seconds": args.seconds,
                        "target": code_target(case),
                        "setup_seconds": setup[method],
                        "workers": sides,
                        "sides": {
                            s: summarize(sides[s][0]["events"], args.seconds / 2, code_target(case)) for s in sides
                        },
                        "validation_status": "pending",
                    }
                    atomic_json(directory / "result.json", record)
                    validation_seconds, count = validate_and_stage(
                        hx,
                        hz,
                        case,
                        sides,
                        fallback,
                        f"{args.output.name}-{method}-s{seed}",
                        executor,
                    )
                    record.update(
                        validation_status="passed", validation_seconds=validation_seconds, saved_candidates=count
                    )
                    atomic_json(directory / "result.json", record)
                    records.append(record)
                    atomic_json(args.output / "results.json", records)
                    weights = [s["best_in_budget"] for s in record["sides"].values() if s["best_in_budget"] is not None]
                    print(
                        json.dumps(
                            {
                                "done": len(records),
                                "case": case["id"],
                                "method": method,
                                "seed": seed,
                                "best": min(weights, default=None),
                                "saved": count,
                            }
                        ),
                        flush=True,
                    )
    if len(records) != len(cases) * len(args.methods) * args.seeds:
        raise ValueError("Incomplete study")
    atomic_json(
        args.output / "completed.json",
        {"configurations": len(records), "saved_candidates": sum(r["saved_candidates"] for r in records)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "results" / "reference-corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=500)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=8)
    main(parser.parse_args())
