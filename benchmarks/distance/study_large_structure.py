"""Frozen, witness-preserving comparison of bounded structural dispatch."""

import argparse
import importlib.metadata
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

from run import initialize_validation, np, validate_and_stage, validation_ready

# isort: split
from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from inspect_dispatch import create_corpus
from strategy_prototypes.large_structure.adapter import CONFIG, Search, native
from study_allocation import archive_sources as allocation_archive
from study_external import run_side
from study_strategies import summarize

METHODS = ("guided", "race", "structure")


def archive_sources(output):
    sources, binaries = allocation_archive(output)
    extra = [HERE / "study_large_structure.py", HERE / "audit_large_structure.py"]
    extra += list((HERE / "strategy_prototypes/large_structure").glob("*.py"))
    extra += [
        HERE / "strategy_prototypes/large_structure/search.cpp",
        HERE / "strategy_prototypes/large_structure/PLAN.md",
    ]
    with tarfile.open(output / "large-structure-sources.tar.gz", "w:gz") as archive:
        for path in extra:
            relative = str(path.relative_to(ROOT))
            sources[relative] = sha256(path)
            archive.add(path, arcname=relative)
    binary = Path(native.__file__)
    binaries[str(binary.relative_to(ROOT))] = sha256(binary)
    return sources, binaries


def main(args):
    if args.seconds <= 0 or args.seeds < 1 or args.validation_workers < 1:
        raise ValueError("Positive budget and counts required")
    if args.cases is None:
        args.cases = [c["id"] for c in json.loads((args.corpus / "manifest.json").read_text())["cases"]]
    if len(set(args.methods)) != len(args.methods) or len(set(args.cases)) != len(args.cases):
        raise ValueError("Duplicate method or case")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    indexed = {c["id"]: c for c in manifest["cases"]}
    cases = [indexed[c] for c in args.cases]
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "manifest.json", dict(sources=manifest["sources"], cases=cases))
    shutil.copyfile(args.corpus / "transforms.json", args.output / "transforms.json")
    sources, binaries = archive_sources(args.output)
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError("Numerical library exceeds one thread")
    env = dict(
        platform=platform.platform(),
        python=sys.version,
        cpu=args.cpu,
        threads=1,
        validation_cpus=available,
        seconds_per_code=args.seconds,
        seed_start=args.seed_start,
        seeds=args.seeds,
        methods=args.methods,
        cases=args.cases,
        checkpoints=[args.seconds],
        source_hashes=sources,
        binary_hashes=binaries,
        dispatcher_parameters=CONFIG,
        threadpools=pools,
        package_versions={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "ldpc", "pybind11")},
        reference_policy="Only matrices and prepared logical detectors enter search; no reference targets",
        timing="Preparation, initialization, analysis, search and witness delivery inside half-budget per side",
        fallback="Construction-assisted restricted kernels followed by guided fallback. "
        "No reference witnesses enter the search.",
        random_seed_policy="side seed = index * 1000003 + (0 for X, 499979 for Z)",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    atomic_json(args.output / "environment.json", env)
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.validation_workers)
    records = []
    with ProcessPoolExecutor(
        args.validation_workers,
        mp_context=context,
        initializer=initialize_validation,
        initargs=(available, barrier),
    ) as executor:
        for future in [executor.submit(validation_ready) for _ in range(args.validation_workers)]:
            future.result()
        for case in cases:
            with np.load(args.corpus / case["file"], allow_pickle=False) as data:
                hx, hz = data["hx"], data["hz"]
            if matrix_hash(hx, hz) != case["matrix_sha256"]:
                raise ValueError("Corpus matrix mismatch")
            (args.output / "matrices").mkdir(exist_ok=True)
            shutil.copyfile(args.corpus / case["file"], args.output / "matrices" / case["file"])
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                methods = list(args.methods)
                random.Random(f"large-structure-v1:{case['id']}:{seed}").shuffle(methods)
                for method in methods:
                    folder = args.output / case["id"] / f"{method}-s{seed}"
                    folder.mkdir(parents=True)
                    sides = {}
                    for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
                        side_seed = seed * 1000003 + (499979 if side == "Z" else 0)
                        sides[side] = [
                            run_side(
                                Search(own, opposite, method, case["structure_spec"]),
                                args.seconds / 2,
                                side_seed,
                                folder / f"{side}.jsonl",
                            )
                        ]
                    record = dict(
                        case=case["id"],
                        method=method,
                        seed=seed,
                        budget_seconds=args.seconds,
                        target=code_target(case),
                        workers=sides,
                        sides={s: summarize(sides[s][0]["events"], args.seconds / 2, code_target(case)) for s in sides},
                        validation_status="pending",
                    )
                    atomic_json(folder / "result.json", record)
                    if any(not sides[s][0]["events"] for s in sides):
                        raise RuntimeError("No packaging witness; raw evidence retained")
                    fallback = {s: sides[s][0]["events"][0]["support"] for s in sides}
                    duration, saved = validate_and_stage(
                        hx, hz, case, sides, fallback, f"{args.output.name}-{method}-s{seed}", executor
                    )
                    record.update(validation_status="passed", saved_candidates=saved, validation_seconds=duration)
                    atomic_json(folder / "result.json", record)
                    records.append(record)
                    atomic_json(args.output / "results.json", records)
                    print(
                        json.dumps(
                            dict(
                                done=len(records),
                                case=case["id"],
                                method=method,
                                seed=seed,
                                best=min(s["best_in_budget"] for s in record["sides"].values()),
                                saved=saved,
                            )
                        ),
                        flush=True,
                    )
    atomic_json(
        args.output / "completed.json",
        dict(configurations=len(records), saved_candidates=sum(r["saved_candidates"] for r in records)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-corpus", type=Path)
    parser.add_argument("--corpus", type=Path, default=HERE / "results/large-structure-study/corpus")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1600)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=8)
    args = parser.parse_args()
    if args.make_corpus:
        create_corpus(args.make_corpus)
    elif args.output:
        main(args)
    else:
        parser.error("Provide --output or --make-corpus")
