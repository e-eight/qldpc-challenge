"""Compare initialized native strategies, including preparation in the budget."""

import argparse
import json
import multiprocessing as mp
import os
import platform
import random
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from initialized_search import METHODS, Search
from run import initialize_validation, np, validate_and_stage, validation_ready
from study_strategies import CASES, snapshot, summarize


def run_side(engine, seconds, seed, path):
    events = []
    with path.open("x") as stream:
        started, cpu = time.perf_counter(), time.process_time()

        def emit(weight, support, stage):
            event = {
                "seconds": time.perf_counter() - started,
                "weight": int(weight),
                "support": list(support),
                "stage": stage,
            }
            events.append(event)
            stream.write(json.dumps(event) + "\n")
            stream.flush()

        try:
            counters = engine.run(seconds, seed, emit)
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "events": events,
        "search_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
        "status": counters.get("status", "completed"),
        "counters": counters,
    }


def main(args):
    if args.seconds <= 0 or args.seeds < 1 or args.validation_workers < 1:
        raise ValueError("Positive budget and counts required")
    if len(set(args.methods)) != len(args.methods) or len(set(args.cases)) != len(args.cases):
        raise ValueError("Duplicate methods/cases")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    indexed = {case["id"]: case for case in manifest["cases"]}
    cases = [indexed[name] for name in args.cases]
    # Refresh the evaluation target using the already saved diagnostic witness.
    # It is never passed to Search, which receives matrices and a method only.
    if "fresh-bb-960" in args.cases:
        diagnostic = json.loads((HERE / "results/strategy-study/prepared-basis/results.json").read_text())
        source = next(row for row in diagnostic if row["case"] == "fresh-bb-960")
        case = indexed["fresh-bb-960"]
        case["previous_reference"] = case["reference"]
        case["reference"] = {
            side: min(source["workers"][side][0]["events"], key=lambda e: e["weight"])["support"] for side in ("X", "Z")
        }
        case["reference_update"] = "Prepared logical-basis diagnostic; independently saved and validated"
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "manifest.json", {"sources": manifest["sources"], "cases": cases})
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu})
    sources, binaries = snapshot(args.output, ["guided", "descent"])
    extra_sources = [
        Path(__file__),
        HERE / "initialized_search.py",
        HERE / "test_initialized.py",
        HERE / "report_initialized.py",
        HERE / "strategy_prototypes/INITIALIZED_PLAN.md",
    ]
    with tarfile.open(args.output / "initialized-sources.tar.gz", "w:gz") as archive:
        for path in extra_sources:
            relative = str(path.relative_to(ROOT))
            sources[relative] = sha256(path)
            archive.add(path, arcname=relative)
    env = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": args.cpu,
        "threads": 1,
        "seconds_per_code": args.seconds,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "methods": args.methods,
        "cases": args.cases,
        "source_hashes": sources,
        "binary_hashes": binaries,
        "checkpoints": [t for t in (0.1, 2, 10, 30, 60) if t <= args.seconds],
        "timing": "Cold preparation, initialization and search within each equal side budget; imports excluded",
        "reference_policy": "Reference supports and target weights never passed to Search",
        "random_seed_policy": "side seed = index * 1000003 + (0 for X, 499979 for Z)",
        "checkpoint_policy": "min(X witness by t/2, Z witness by t/2), from the same continuing run",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.seconds not in env["checkpoints"]:
        env["checkpoints"].append(args.seconds)
    atomic_json(args.output / "environment.json", env)
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.validation_workers)
    records = []
    with ProcessPoolExecutor(
        args.validation_workers, mp_context=context, initializer=initialize_validation, initargs=(available, barrier)
    ) as executor:
        futures = [executor.submit(validation_ready) for _ in range(args.validation_workers)]
        for future in futures:
            future.result()
        for case in cases:
            with np.load(args.corpus / case["file"], allow_pickle=False) as data:
                hx, hz = data["hx"], data["hz"]
            if matrix_hash(hx, hz) != case["matrix_sha256"]:
                raise ValueError("Corpus matrix hash mismatch")
            matrix_dir = args.output / "matrices"
            matrix_dir.mkdir(exist_ok=True)
            shutil.copyfile(args.corpus / case["file"], matrix_dir / case["file"])
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                methods = list(args.methods)
                random.Random(f"initialized:{case['id']}:{seed}").shuffle(methods)
                for method in methods:
                    directory = args.output / case["id"] / f"{method}-s{seed}"
                    directory.mkdir(parents=True)
                    sides = {}
                    for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
                        side_seed = seed * 1000003 + (499979 if side == "Z" else 0)
                        sides[side] = [
                            run_side(
                                Search(own, opposite, method), args.seconds / 2, side_seed, directory / f"{side}.jsonl"
                            )
                        ]
                    record = {
                        "case": case["id"],
                        "method": method,
                        "seed": seed,
                        "budget_seconds": args.seconds,
                        "target": code_target(case),
                        "workers": sides,
                        "sides": {
                            s: summarize(sides[s][0]["events"], args.seconds / 2, code_target(case)) for s in sides
                        },
                        "validation_status": "pending",
                    }
                    atomic_json(directory / "result.json", record)
                    # Packaging fallback comes from this run's own output.
                    # Failure to produce either side remains a hard error, with
                    # the raw observations retained for recovery.
                    if any(not sides[s][0]["events"] for s in sides):
                        raise RuntimeError("No packaging witness on one side; increase the benchmark budget")
                    fallback = {s: sides[s][0]["events"][0]["support"] for s in sides}
                    validation_seconds, saved = validate_and_stage(
                        hx, hz, case, sides, fallback, f"{args.output.name}-{method}-s{seed}", executor
                    )
                    record.update(
                        validation_status="passed", saved_candidates=saved, validation_seconds=validation_seconds
                    )
                    atomic_json(directory / "result.json", record)
                    records.append(record)
                    atomic_json(args.output / "results.json", records)
                    print(
                        json.dumps(
                            {
                                "done": len(records),
                                "case": case["id"],
                                "method": method,
                                "seed": seed,
                                "best": min(s["best_in_budget"] for s in record["sides"].values()),
                                "saved": saved,
                            }
                        ),
                        flush=True,
                    )
    atomic_json(
        args.output / "completed.json",
        {"configurations": len(records), "saved_candidates": sum(r["saved_candidates"] for r in records)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "results/reference-corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=800)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=8)
    main(parser.parse_args())
