"""Compare fixed RIS work with identical seeds, retaining all returned witnesses."""

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from common import HERE, ROOT, atomic_json, matrix_hash, sha256

sys.path.insert(0, str(HERE / "build"))
sys.path.insert(0, str(ROOT / "native" / "ris"))
import benchmark_native
import ris_native
from run import validate_and_stage


def main(args):
    if args.trials < 1 or args.threads < 1 or args.repeats < 1:
        raise ValueError("positive trial, thread, and repeat counts required")
    if args.cpus:
        if len(set(args.cpus)) != args.threads:
            raise ValueError("choose one CPU per worker")
        os.sched_setaffinity(0, args.cpus)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if not args.cases or c["id"] in args.cases]
    if not cases or (args.cases and len(cases) != len(set(args.cases))):
        raise ValueError("unknown or empty case selection")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {
        "platform": platform.platform(),
        "threads": args.threads,
        "affinity": sorted(os.sched_getaffinity(0)),
        "trials_per_side": args.trials,
        "repeats": args.repeats,
        "seed_start": args.seed_start,
        "methods": args.methods,
        "ris_binary_sha256": sha256(ris_native.__file__),
        "ris_build": {"compiler": ris_native.compiler, "host_tuned": ris_native.host_tuned},
        "baseline_binary_sha256": sha256(benchmark_native.__file__),
        "corpus_manifest_sha256": sha256(args.corpus / "manifest.json"),
        "timing": "Preparation excluded; synchronous call includes dispatch and result conversion; no target stopping",
    }
    atomic_json(args.output / "environment.json", environment)
    atomic_json(args.output / "corpus.json", {**manifest, "cases": cases})
    for path in [
        Path(__file__),
        HERE / "native.cpp",
        HERE / "run.py",
        HERE / "setup_native.py",
        ROOT / "verify" / "gf2_fast.cpp",
        ROOT / "native" / "ris" / "setup.py",
        *(ROOT / "native" / "ris" / "src").glob("*.cpp"),
        *(ROOT / "native" / "ris" / "include" / "ris").glob("*.hpp"),
    ]:
        destination = args.output / "sources" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for case in cases:
        with np.load(args.corpus / case["file"]) as data:
            hx, hz = data["hx"], data["hz"]
        if matrix_hash(hx, hz) != case["matrix_sha256"]:
            raise ValueError("input hash mismatch")
        for repeat in range(args.repeats):
            # Rotate ordering to avoid always giving one method the first run.
            methods = args.methods[repeat % len(args.methods) :] + args.methods[: repeat % len(args.methods)]
            for method in methods:
                seed = args.seed_start + repeat
                run_id = f"{args.output.name}-{method}-t{args.threads}-s{seed}"
                record = {"case": case["id"], "method": method, "seed": seed, "threads": args.threads, "sides": {}}
                workers = {}
                for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
                    side_seed = seed + (0 if side == "X" else 499979)
                    start = time.perf_counter()
                    if method == "cpp":
                        prepared = benchmark_native.PreparedSearch(own, opposite)
                        session = None
                    else:
                        prepared = ris_native.Prepared(own, opposite)
                        session = ris_native.Session(
                            prepared,
                            args.threads,
                            side_seed,
                            masked=method == "ris-masked",
                            block_size=int(method[-1]) if method.startswith("ris-block") else 1,
                        )
                    preparation = time.perf_counter() - start
                    cpu_start, start = time.process_time(), time.perf_counter()
                    if method == "cpp":
                        weight, support, trials = prepared.batch(args.trials, side_seed, 8, args.threads, 0)
                        events = [{"weight": weight, "support": support}]
                    else:
                        result = session.advance(args.trials)
                        weight, trials = result.best_weight, result.trials
                        events = [
                            {"weight": e.weight, "support": e.support, "worker": e.worker, "worker_trial": e.trial}
                            for e in result.improvements
                        ]
                    elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_start
                    workers[side] = [{"events": events}]
                    record["sides"][side] = {
                        "seconds": elapsed,
                        "cpu_seconds": cpu,
                        "preparation_seconds": preparation,
                        "trials": trials,
                        "best_weight": weight,
                        "events": events,
                        "basis_bytes": prepared.basis_bytes if session else None,
                        "workspace_bytes": session.workspace_bytes if session else None,
                    }
                    # Persist immediately, including before starting the other side.
                    atomic_json(args.output / case["id"] / f"{method}-s{seed}.json", record)
                    del session, prepared
                seconds, count = validate_and_stage(hx, hz, case, workers, case["reference"], run_id)
                record.update(validation_status="passed", validation_seconds=seconds, saved_candidates=count)
                atomic_json(args.output / case["id"] / f"{method}-s{seed}.json", record)
                print(
                    f"{case['id']} {method} s{seed}: "
                    + " ".join(
                        f"{side}={r['best_weight']} {r['trials'] / r['seconds']:.0f}/s"
                        for side, r in record["sides"].items()
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "results" / "reference-corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cpp", "ris", "ris-masked", "ris-block4", "ris-block6"],
        default=["cpp", "ris", "ris-block4", "ris-block6"],
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpus", nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=100)
    main(parser.parse_args())
