"""Run a fixed-worker, witness-preserving CPU distance-search pilot or study."""

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Set before importing numerical libraries or creating worker processes.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qldpc-benchmark-matplotlib")

import numpy as np
from common import (
    HERE,
    ROOT,
    atomic_json,
    benchmark_schema_status,
    code_target,
    matrix_hash,
    sha256,
    submission_validator,
)

sys.path.insert(0, str(HERE / "build"))
import benchmark_native
import gf2
from submit import make_submission, save_submission
from workers import Recorder, initialize, ready, search

METHODS = ["cpp", "cpp-no-pairs", "m4ri", "qdistevol", "qdist-random", "numpy", "cpp-circulant"]
EXPERIMENTAL_METHODS = ["ris", "ris-masked", "ris-block4", "ris-block6", "ris-incremental"]


def ris_search(
    own,
    opposite,
    path,
    seconds,
    target,
    seed,
    threads,
    masked=False,
    block_size=1,
    restart_interval=0,
    exchange_proposals=8,
):
    """Observe the standalone engine, preserving every worker's improvements."""
    sys.path.insert(0, str(ROOT / "native" / "ris"))
    import ris_native

    before = time.perf_counter()
    prepared = ris_native.Prepared(own, opposite)
    session = ris_native.Session(
        prepared,
        threads,
        seed,
        masked=masked,
        block_size=block_size,
        restart_interval=restart_interval,
        exchange_proposals=exchange_proposals,
    )
    preparation = time.perf_counter() - before
    recorder = Recorder(path, seconds, target)
    batch = threads
    counters = {}
    try:
        if prepared.applicable:
            while True:
                before = time.perf_counter()
                result = session.advance(batch)
                elapsed = time.perf_counter() - recorder.start
                recorder.trials += result.trials
                counters = {key: getattr(result, key) for key in ("reductions", "proposals", "exchanges")}
                # All supports in the batch become observable together. Native
                # discovery time is not substituted for Python delivery time.
                for improvement in result.improvements:
                    event = {
                        "seconds": elapsed,
                        "weight": improvement.weight,
                        "support": improvement.support,
                        "trials": recorder.trials,
                        "worker": improvement.worker,
                        "worker_trial": improvement.trial,
                        "within_budget": elapsed <= seconds,
                    }
                    recorder.stream.write(json.dumps(event) + "\n")
                    recorder.events.append(event)
                if result.improvements:
                    recorder.stream.flush()
                    os.fsync(recorder.stream.fileno())
                if elapsed >= seconds or (target is not None and result.best_weight <= target):
                    break
                duration = time.perf_counter() - before
                batch = max(threads, min(8192, int(batch * 0.05 / max(duration, 1e-6))))
    finally:
        record = recorder.finish()
    record.update(
        setup_seconds=preparation,
        restart_interval=restart_interval,
        exchange_proposals=exchange_proposals,
        **counters,
        basis_bytes=prepared.basis_bytes,
        workspace_bytes=session.workspace_bytes,
        status="completed" if prepared.applicable else "not_applicable",
    )
    return record


def dependency_version(name):
    """Allow native-only experiments without installing optional controls."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_matrix(path, matrix):
    """Write MatrixMarket coordinates; indices in this format are one-based."""
    rows, columns = np.nonzero(matrix)
    with Path(path).open("w") as stream:
        stream.write("%%MatrixMarket matrix coordinate integer general\n")
        stream.write(f"{matrix.shape[0]} {matrix.shape[1]} {len(rows)}\n")
        for row, column in zip(rows, columns, strict=True):
            stream.write(f"{row + 1} {column + 1} 1\n")


def read_codewords(path):
    """Read dist-m4ri's one-based NZLIST supports, retaining all exported words."""
    words = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("%"):
            continue
        values = list(map(int, line.split()))
        if not values or values[0] != len(values) - 1:
            raise ValueError("Malformed NZLIST weight/support row")
        words.append([value - 1 for value in values[1:]])
    return words


def native_batch_seed(seed, number):
    """Remove systematic stream overlap between adjacent study seeds."""
    mask = (1 << 64) - 1
    data = (int(seed) & mask).to_bytes(8, "little") + (int(number) & mask).to_bytes(8, "little")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, person=b"qldpc-ris-batch").digest(), "little")


def native_search(prepared, path, seconds, target, seed, threads, pairs):
    """Observe batches of the existing C++ trial kernel with prepared bases."""
    recorder = Recorder(path, seconds, target)
    if not prepared.applicable():
        result = recorder.finish()
        result["status"] = "not_applicable"
        return result
    batch, number = threads, 0
    while True:
        before = time.perf_counter()
        weight, support, completed = prepared.batch(batch, native_batch_seed(seed, number), pairs, threads, number)
        if recorder.observe(weight, support, completed):
            break
        duration = time.perf_counter() - before
        # Aim for observations every 50ms. The final batch can overrun; its
        # witness is preserved but never credited before it was observed.
        batch = max(threads, min(8192, int(batch * 0.05 / max(duration, 1e-6))))
        number += 1
    result = recorder.finish()
    result["batch_seed_policy"] = "blake2b(study_seed,batch_index)"
    return result


def m4ri_search(binary, opposite, logicals, directory, seconds, target, seed, threads):
    """Use the public CLI and its native worker/deadline controls."""
    start = time.perf_counter()
    h_path, l_path = directory / "H.mtx", directory / "L.mtx"
    write_matrix(h_path, opposite)
    write_matrix(l_path, logicals)
    setup_seconds = time.perf_counter() - start
    export_reserve = min(0.05, seconds * 0.1)
    command = [
        str(binary),
        "method=1",
        f"finH={h_path}",
        f"finL={l_path}",
        f"threads={threads}",
        f"timeout={seconds - export_reserve}",
        "steps=2147483647",
        f"seed={seed % 2147483647}",
        f"wmin={target or 0}",
        f"outC={directory / 'codewords.txt'}",
        "debug=0",
        "dW=0",
    ]
    atomic_json(directory / "command.json", command)
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with (directory / "stderr.txt").open("w") as stderr:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=stderr, timeout=seconds + 120)
    elapsed = time.perf_counter() - start
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (directory / "stdout.txt").write_bytes(process.stdout)
    lines = process.stdout.decode().strip().splitlines()
    numeric = [
        line.split()
        for line in lines
        if len(line.split()) == 3 and all(part.lstrip("-").isdigit() for part in line.split())
    ]
    if len(numeric) != 1:
        raise ValueError(f"Unexpected dist-m4ri output in {directory}")
    lower, weight, trials = map(int, numeric[0])
    words = read_codewords(directory / "codewords.txt")
    if weight > 0 and (not words or min(map(len, words)) != weight):
        raise ValueError("dist-m4ri bound has no matching exported witness")
    events = [
        {
            "seconds": elapsed,
            "weight": len(word),
            "support": word,
            "trials": trials,
            "within_budget": elapsed <= seconds,
        }
        for word in words
    ]
    return {
        "status": "completed",
        "setup_seconds": setup_seconds,
        "search_seconds": elapsed,
        "trials": trials,
        "events": events,
        "cpu_seconds": cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime,
        "raw_file": str(directory / "codewords.txt"),
        "unused_lower_bound": lower,
        "export_reserve_seconds": export_reserve,
        "delivery_protocol": "unmodified CLI; all exported supports observed at process exit",
    }


def save_benchmark_witness(job):
    hx, hz, case, retained, run_id, sequence = job
    document = make_submission(
        hx,
        hz,
        name=f"Benchmark witness: {case['id']}",
        authors=["qldpc-challenge CPU benchmark"],
        construction=f"Distance benchmark {run_id}; input SHA256 {case['matrix_sha256']}",
        witnesses=retained,
        confidence="upper_bound",
    )
    directory = ROOT / "research" / "candidates" / "distance-benchmark" / case["id"] / run_id / str(sequence)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document['n']}-{document['k']}-{document['distance']['d']}.json"
    errors = save_submission(document, path)
    schema_status = benchmark_schema_status(document)
    if errors and schema_status != "above_size_cap":
        raise ValueError(f"Candidate save reported schema errors: {errors}")
    return str(path)


def initialize_validation(cpus, barrier):
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    barrier.wait(timeout=120)


def validation_ready():
    return os.getpid()


def validate_and_stage(hx, hz, case, sides, fallback, run_id, executor=None):
    """Validate every returned support and package improvements using the kit."""
    start = time.perf_counter()
    retained = dict(fallback)
    paths = []
    sequence = 0
    for side in ("X", "Z"):
        own, opposite = (hx, hz) if side == "X" else (hz, hx)
        seen = set()
        for worker in sides[side]:
            for event in worker["events"]:
                support = event["support"]
                if (
                    event["weight"] != len(support)
                    or len(support) != len(set(support))
                    or any(q < 0 or q >= case["n"] for q in support)
                ):
                    raise ValueError("Malformed benchmark witness")
                vector = np.zeros(case["n"], dtype=np.int8)
                vector[support] = 1
                if not support or not gf2.commutes(vector, opposite) or gf2.in_rowspace(vector, own):
                    raise ValueError(f"Invalid {side} witness: {case['id']}, {run_id}")
                key = tuple(support)
                if key in seen:
                    continue
                seen.add(key)
                # Each distinct returned support gets its own saved candidate;
                # the other side uses a valid basis witness only for packaging.
                retained[side] = support
                job = (hx, hz, case, dict(retained), run_id, sequence)
                paths.append(executor.submit(save_benchmark_witness, job) if executor else save_benchmark_witness(job))
                sequence += 1
    if executor:
        # Complete every save before any subsequent timed search. A failed save
        # remains a hard error; raw result.json was persisted before dispatch.
        for future in paths:
            future.result()
    return time.perf_counter() - start, len(paths)


def summarize_side(workers, target, seconds):
    """Compute deadline-censored results without dropping missed targets."""
    events = [event for worker in workers for event in worker["events"]]
    timely = [event for event in events if event["seconds"] <= seconds]
    hits = [event["seconds"] for event in timely if target is not None and event["weight"] <= target]
    return {
        "target": target,
        "best_in_budget": min((event["weight"] for event in timely), default=None),
        "best_returned": min((event["weight"] for event in events), default=None),
        "target_hit": bool(hits) if target is not None else None,
        "time_to_target": min(hits, default=None),
        "completed_trials": sum(worker["trials"] for worker in workers),
        "cpu_seconds": sum(worker["cpu_seconds"] for worker in workers),
        "elapsed_seconds": max(worker["search_seconds"] for worker in workers),
        "status": "not_applicable" if all(worker["status"] == "not_applicable" for worker in workers) else "completed",
    }


def main(args):
    """Run sequential methods with a fixed total number of CPU workers."""
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = [case for case in manifest["cases"] if not args.cases or case["id"] in args.cases]
    if not cases or (args.cases and set(args.cases) - {case["id"] for case in cases}):
        raise ValueError("No cases selected, or unknown case IDs")
    if args.threads < 1 or args.seconds <= 0 or args.seeds < 1 or args.seed_start < 0:
        raise ValueError("Threads, seconds, and seed count must be positive; seeds must be nonnegative")
    if args.restart_interval < 1 or args.exchange_proposals < 1:
        raise ValueError("Incremental restart interval and exchange proposals must be positive")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Repeated methods would overwrite benchmark evidence")
    validation_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if args.validation_workers < 1:
        raise ValueError("Validation worker count must be positive")
    if args.cpus is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("CPU affinity is unavailable on this platform")
        if len(set(args.cpus)) != args.threads:
            raise ValueError("Choose exactly one CPU per worker")
        os.sched_setaffinity(0, set(args.cpus))
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "workers": args.threads,
        "validation_workers": args.validation_workers,
        "validation_affinity": validation_cpus,
        "seconds_per_side": args.seconds / 2,
        "seed_start": args.seed_start,
        "cpp_batch_seed_policy": "blake2b(study_seed,batch_index)",
        "seeds": args.seeds,
        "incremental_restart_interval": args.restart_interval,
        "incremental_exchange_proposals": args.exchange_proposals,
        "m4ri_export_reserve_seconds": min(0.05, args.seconds * 0.05),
        "stop_at_target": not args.no_target_stop,
        "target_policy": "Same code-level minimum target on both sides; preserve smaller paper or analytic targets",
        "submission_size_cap": submission_validator().schema["properties"]["n"]["maximum"],
        "methods": args.methods,
        "cases": [case["id"] for case in cases],
        "corpus_manifest_sha256": sha256(manifest_path),
        "source_pins": manifest["sources"],
        "reference_policy": "frozen supplied references; paper targets labeled separately",
        "native_source_sha256": sha256(ROOT / "verify" / "gf2_fast.cpp"),
        "native_binary_sha256": sha256(benchmark_native.__file__),
        "adapter_sha256": sha256(HERE / "native.cpp"),
        "runner_sha256": sha256(HERE / "run.py"),
        "workers_sha256": sha256(HERE / "workers.py"),
        "submit_source_sha256": sha256(ROOT / "research" / "kit" / "submit.py"),
        "m4ri_binary_sha256": sha256(args.m4ri) if "m4ri" in args.methods else None,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "dependencies": {
            name: dependency_version(name)
            for name in ("numpy", "numba", "scipy", "codedistance", "pybind11", "threadpoolctl")
        },
    }
    if set(args.methods).intersection(EXPERIMENTAL_METHODS):
        sys.path.insert(0, str(ROOT / "native" / "ris"))
        import ris_native

        environment["ris_binary_sha256"] = sha256(ris_native.__file__)
        environment["ris_build"] = {"compiler": ris_native.compiler, "host_tuned": ris_native.host_tuned}
    if "m4ri" in args.methods:
        build_metadata = args.m4ri.parent.parent.parent / "build.json"
        if build_metadata.exists():
            metadata = json.loads(build_metadata.read_text())
            if metadata.get("m4ri_binary_sha256") == environment["m4ri_binary_sha256"]:
                environment["m4ri_build"] = metadata
    atomic_json(args.output / "environment.json", environment)
    atomic_json(args.output / "corpus.json", manifest)
    # Preserve the exact adapter sources and matrix bytes before any search.
    for path in [
        *HERE.glob("*.py"),
        HERE / "native.cpp",
        HERE / "sources.json",
        HERE / "requirements.txt",
        ROOT / "research" / "kit" / "submit.py",
        ROOT / "verify" / "gf2_fast.cpp",
        ROOT / "schema" / "code.schema.json",
        *sorted((ROOT / "native" / "ris" / "src").glob("*.cpp")),
        *sorted((ROOT / "native" / "ris" / "include" / "ris").glob("*.hpp")),
        *sorted((ROOT / "native" / "ris").glob("*.py")),
        *sorted((ROOT / "native" / "ris").glob("*.toml")),
        ROOT / "native" / "ris" / "CMakeLists.txt",
        ROOT / "native" / "ris" / "README.md",
        ROOT / "native" / "ris" / "LICENSE",
        ROOT / "native" / "ris" / "MANIFEST.in",
        ROOT / "native" / "ris" / "tests" / "test_core.cpp",
    ]:
        destination = args.output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (args.output / "matrices").mkdir()
    atomic_json(args.output / "matrices" / "manifest.json", manifest)
    for case in manifest["cases"]:
        shutil.copyfile(args.corpus / case["file"], args.output / "matrices" / case["file"])
    python_methods = {"qdistevol", "qdist-random", "numpy"}
    pool = None
    if python_methods.intersection(args.methods):
        startup = time.perf_counter()
        context = mp.get_context("spawn")
        barrier, stop = context.Barrier(args.threads), context.Event()
        pool = context.Pool(
            args.threads, initialize, (barrier, stop, bool({"qdistevol", "qdist-random"}.intersection(args.methods)))
        )
        # Wait for every import/JIT to finish before any timed native work.
        pool.map_async(ready, range(args.threads)).get(timeout=180)
        environment["worker_startup_seconds"] = time.perf_counter() - startup
        atomic_json(args.output / "environment.json", environment)
    validation_executor = None
    if args.validation_workers > 1:
        context = mp.get_context("spawn")
        barrier = context.Barrier(args.validation_workers)
        validation_executor = ProcessPoolExecutor(
            max_workers=args.validation_workers,
            mp_context=context,
            initializer=initialize_validation,
            initargs=(validation_cpus, barrier),
        )
        futures = [validation_executor.submit(validation_ready) for _ in range(args.validation_workers)]
        for future in futures:
            future.result(timeout=180)
    try:
        for case in cases:
            with np.load(args.corpus / case["file"], allow_pickle=False) as data:
                hx, hz = data["hx"], data["hz"]
            if matrix_hash(hx, hz) != case["matrix_sha256"]:
                raise ValueError(f"Corpus matrix changed: {case['id']}")
            before = time.perf_counter()
            prepared = {"X": benchmark_native.PreparedSearch(hx, hz), "Z": benchmark_native.PreparedSearch(hz, hx)}
            # Full quotient bases supply the required nontriviality tests.
            # Neither paper witnesses nor benchmark target supports are used.
            logicals = {side: prepared[side].opposite_logicals() for side in ("X", "Z")}
            fallback = {"X": np.flatnonzero(logicals["Z"][0]).tolist(), "Z": np.flatnonzero(logicals["X"][0]).tolist()}
            shared_preparation = time.perf_counter() - before
            structural = None
            before = time.perf_counter()
            if "cpp-circulant" in args.methods:
                block_size = benchmark_native.circulant_size(hx)
                structural = {
                    "X": benchmark_native.PreparedSearch(hx, hz, True, block_size),
                    "Z": benchmark_native.PreparedSearch(hz, hx, True, block_size),
                }
            structural_preparation = time.perf_counter() - before
            for seed_index in range(args.seed_start, args.seed_start + args.seeds):
                methods = list(args.methods)
                random.Random(f"{case['id']}:{seed_index}").shuffle(methods)
                for method in methods:
                    run_id = f"{args.output.name}-{method}-t{args.threads}-s{seed_index}"
                    directory = args.output / case["id"] / f"{method}-s{seed_index}"
                    directory.mkdir(parents=True)
                    start = time.perf_counter()
                    results, summaries = {}, {}
                    for side in ("X", "Z"):
                        own, opposite = (hx, hz) if side == "X" else (hz, hx)
                        side_dir = directory / side
                        side_dir.mkdir()
                        target = code_target(case)
                        if args.no_target_stop:
                            stopping_target = None
                        else:
                            stopping_target = target
                        seed = seed_index * 1000003 + (0 if side == "X" else 499979)
                        if method in EXPERIMENTAL_METHODS:
                            workers = [
                                ris_search(
                                    own,
                                    opposite,
                                    side_dir / "events.jsonl",
                                    args.seconds / 2,
                                    stopping_target,
                                    seed,
                                    args.threads,
                                    masked=method == "ris-masked",
                                    block_size=(
                                        6
                                        if method == "ris-incremental"
                                        else int(method[-1])
                                        if method.startswith("ris-block")
                                        else 1
                                    ),
                                    restart_interval=args.restart_interval if method == "ris-incremental" else 0,
                                    exchange_proposals=args.exchange_proposals,
                                )
                            ]
                        elif method.startswith("cpp"):
                            engine = structural[side] if method == "cpp-circulant" else prepared[side]
                            workers = [
                                native_search(
                                    engine,
                                    side_dir / "events.jsonl",
                                    args.seconds / 2,
                                    stopping_target,
                                    seed,
                                    args.threads,
                                    0 if method == "cpp-no-pairs" else 8,
                                )
                            ]
                        elif method == "m4ri":
                            workers = [
                                m4ri_search(
                                    args.m4ri,
                                    opposite,
                                    logicals[side],
                                    side_dir,
                                    args.seconds / 2,
                                    stopping_target,
                                    seed,
                                    args.threads,
                                )
                            ]
                        else:
                            stop.clear()
                            jobs = [
                                {
                                    "method": method,
                                    "own": own,
                                    "opposite": opposite,
                                    "logicals": logicals[side],
                                    "seconds": args.seconds / 2,
                                    "target": stopping_target,
                                    "seed": seed + worker * 15485863,
                                    "path": str(side_dir / f"worker-{worker}.jsonl"),
                                }
                                for worker in range(args.threads)
                            ]
                            workers = pool.map_async(search, jobs).get(timeout=args.seconds + 180)
                        results[side] = workers
                        summaries[side] = summarize_side(workers, target, args.seconds / 2)
                        summaries[side]["target_evidence"] = (
                            "validated_code_witness"
                            if any(len(support) == target for support in case.get("reference", {}).values())
                            else "analytic"
                            if "analytic_target" in case
                            else "paper_unverified"
                            if target
                            else "none"
                        )
                    search_and_dispatch = time.perf_counter() - start
                    record = {
                        "case": case["id"],
                        "method": method,
                        "seed": seed_index,
                        "threads": args.threads,
                        "budget_seconds": args.seconds,
                        "shared_preparation_seconds": shared_preparation,
                        "structural_preparation_seconds": structural_preparation if method == "cpp-circulant" else 0,
                        "search_and_dispatch_seconds": search_and_dispatch,
                        "workers": results,
                        "sides": summaries,
                        "validation_status": "pending",
                    }
                    atomic_json(directory / "result.json", record)
                    validation_seconds, saved = validate_and_stage(
                        hx, hz, case, results, fallback, run_id, validation_executor
                    )
                    record.update(
                        validation_status="passed",
                        validation_seconds=validation_seconds,
                        saved_candidates=saved,
                        submission_schema_status=(
                            "above_size_cap" if case["n"] > environment["submission_size_cap"] else "valid"
                        ),
                    )
                    atomic_json(directory / "result.json", record)
                    print(
                        f"{case['id']} {method} seed={seed_index}: "
                        f"X={summaries['X']['best_in_budget']} Z={summaries['Z']['best_in_budget']} "
                        f"trials={sum(summaries[s]['completed_trials'] for s in ('X', 'Z'))}",
                        flush=True,
                    )
    finally:
        if validation_executor:
            validation_executor.shutdown(wait=True)
        if pool:
            pool.terminate()
            pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "cache" / "corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m4ri", type=Path, default=HERE / "cache" / "deps" / "dist-m4ri" / "src" / "dist_m4ri")
    parser.add_argument("--methods", nargs="+", choices=METHODS + EXPERIMENTAL_METHODS, default=METHODS)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpus", type=int, nargs="+", help="Linux CPU affinity; choose one physical core per worker")
    parser.add_argument("--seconds", type=float, default=10, help="Total per-code budget, split equally by side")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--no-target-stop", action="store_true", help="Continue beyond reference targets to search for tighter bounds"
    )
    parser.add_argument("--restart-interval", type=int, default=64)
    parser.add_argument("--exchange-proposals", type=int, default=8)
    parser.add_argument(
        "--validation-workers", type=int, default=1, help="Parallel witness saves outside search timing"
    )
    main(parser.parse_args())
