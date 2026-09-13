"""Resumable, four-worker overnight suite with immediate witness persistence."""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
import traceback
from concurrent import futures
from pathlib import Path

from run import np, validate_and_stage

# isort: split
from common import ROOT, atomic_json, code_target, matrix_hash, sha256
from full_suite.adapter import Search
from study_strategies import summarize


def pin(queue):
    os.sched_setaffinity(0, {queue.get()})


def run_side(engine, seconds, seed, path):
    events = []
    start, cpu = time.perf_counter(), time.process_time()
    with path.open("x") as stream:

        def emit(weight, support, stage):
            event = dict(
                seconds=time.perf_counter() - start, weight=int(weight), support=list(map(int, support)), stage=stage
            )
            events.append(event)
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        try:
            counters = engine.run(seconds, seed, emit)
            error = None
        except Exception:
            error = traceback.format_exc()
            counters = dict(status="error")
    return dict(
        events=events,
        counters=counters,
        status=counters.get("status", "completed"),
        error=error,
        search_seconds=time.perf_counter() - start,
        cpu_seconds=time.process_time() - cpu,
    )


def job(output, case, method, seconds, seed):
    output = Path(output)
    base = output / "runs" / case["id"] / method
    base.mkdir(parents=True, exist_ok=True)
    attempt = len(list(base.glob("attempt-*")))
    directory = base / f"attempt-{attempt:03}"
    directory.mkdir()
    sides, errors = {"X": [], "Z": []}, []
    started = time.time()
    with np.load(output / "matrices" / case["file"], allow_pickle=False) as d:
        hx, hz = d["hx"], d["hz"]
    if matrix_hash(hx, hz) != case["matrix_sha256"]:
        raise ValueError("Frozen matrix differs")
    for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
        side_seed = seed * 1000003 + (499979 if side == "Z" else 0)
        worker = run_side(
            Search(own, opposite, method, directory / (side + "-m4ri")),
            seconds / 2,
            side_seed,
            directory / (side + ".jsonl"),
        )
        sides[side] = [worker]
        if worker["error"]:
            errors.append(worker["error"])
            break
    record = dict(
        case=case["id"],
        method=method,
        seed=seed,
        budget_seconds=seconds,
        target=code_target(case),
        workers=sides,
        started_unix=started,
        cpu=next(iter(os.sched_getaffinity(0))),
        attempt=attempt,
        directory=str(directory.relative_to(output)),
        validation_status="pending",
        checkpoints={
            str(b): {
                s: summarize(sides[s][0]["events"] if sides[s] else [], b / 2, code_target(case)) for s in ("X", "Z")
            }
            for b in ([30, 60] if method != "circulant" else [2])
        },
        errors=errors,
    )
    atomic_json(directory / "result.json", record)
    run_id = f"{output.name}-{method}-s{seed}-a{attempt:03}"
    # References are used exclusively to package the unsearched other sector
    # after an exception. They are never passed to any Search object.
    try:
        duration, saved = validate_and_stage(hx, hz, case, sides, case["reference"], run_id)
        record.update(validation_status="passed", validation_seconds=duration, saved_candidates=saved, run_id=run_id)
    except Exception:
        record.update(validation_status="failed", validation_error=traceback.format_exc())
        atomic_json(directory / "result.json", record)
        raise
    record["finished_unix"] = time.time()
    atomic_json(directory / "result.json", record)
    if errors:
        raise RuntimeError(f"Search failed after saving all exports: {directory}: {errors[0]}")
    atomic_json(
        base / "completed.json",
        dict(result=record["directory"] + "/result.json", sha256=sha256(directory / "result.json")),
    )
    return record


def checked_pins(output):
    env = json.loads((output / "environment.json").read_text())
    for field in ("source_hashes", "binary_hashes"):
        for path, expected in env[field].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f"Frozen {field} changed: {path}")
    return env


def main(output, resume=False):
    output = output.resolve()
    env = checked_pins(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cases = manifest["cases"]
    if (output / "progress.json").exists() and not resume:
        raise ValueError("Existing run requires --resume")
    if (output / "completed.json").exists():
        raise ValueError("Run already completed")
    seed = env["seed"]
    jobs = [(c, m, 60, seed) for c in cases for m in env["methods"]]
    jobs += [(c, "circulant", 2, seed) for c in cases]
    random.Random(2101).shuffle(jobs)
    finished, pending = [], []
    for c, m, b, s in jobs:
        marker = output / "runs" / c["id"] / m / "completed.json"
        if marker.exists():
            meta = json.loads(marker.read_text())
            if sha256(output / meta["result"]) != meta["sha256"]:
                raise ValueError("Completed record changed")
            r = json.loads((output / meta["result"]).read_text())
            if r["validation_status"] != "passed" or r["errors"]:
                raise ValueError("Invalid completion marker")
            finished.append(r)
        else:
            pending.append((c, m, b, s))
    start = time.time()
    existing = len(finished)

    def progress(status, running=(), error=None):
        elapsed = time.time() - start
        newly = finished[existing:]
        completed_allocation = sum(r["budget_seconds"] for r in finished)
        new_allocation = sum(r["budget_seconds"] for r in newly)
        total_allocation = sum(b for _, _, b, _ in jobs)
        eta = (total_allocation - completed_allocation) * elapsed / new_allocation if new_allocation else None
        atomic_json(
            output / "progress.json",
            dict(
                status=status,
                pid=os.getpid(),
                updated_unix=time.time(),
                session_started_unix=start,
                total_jobs=len(jobs),
                completed_jobs=len(finished),
                completed_general_jobs=sum(r["method"] != "circulant" for r in finished),
                total_general_jobs=len(cases) * len(env["methods"]),
                saved_documents=sum(r["saved_candidates"] for r in finished),
                session_elapsed_seconds=elapsed,
                remaining_seconds_estimate=eta,
                running=[dict(case=c["id"], method=m) for c, m, _, _ in running],
                error=error,
            ),
        )

    progress("running")
    context = mp.get_context("spawn")
    queue = context.Queue()
    for cpu in env["cpus"]:
        queue.put(cpu)
    with futures.ProcessPoolExecutor(
        len(env["cpus"]), mp_context=context, initializer=pin, initargs=(queue,)
    ) as executor:
        active = {}
        todo = iter(pending)

        def schedule():
            while len(active) < len(env["cpus"]):
                item = next(todo, None)
                if item is None:
                    break
                active[executor.submit(job, output, *item)] = item

        schedule()
        progress("running", active.values())
        failure = None
        while active:
            done, _ = futures.wait(active, timeout=30, return_when=futures.FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                try:
                    record = future.result()
                    finished.append(record)
                    print(
                        json.dumps(
                            dict(
                                done=len(finished),
                                total=len(jobs),
                                case=record["case"],
                                method=record["method"],
                                saved=record["saved_candidates"],
                                seconds=round(record["finished_unix"] - record["started_unix"], 2),
                            )
                        ),
                        flush=True,
                    )
                except Exception:
                    failure = traceback.format_exc()
            if failure is None:
                schedule()
            progress("draining_after_error" if failure else "running", active.values(), failure)
        if failure:
            progress("failed", error=failure)
            raise RuntimeError(failure)
    progress("auditing")
    from full_suite.report import finalize, freeze

    finalize(output)
    progress("complete")
    freeze(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    try:
        main(args.output, args.resume)
    except Exception:
        atomic_json(args.output / "driver-error.json", dict(error=traceback.format_exc(), time_unix=time.time()))
        raise
