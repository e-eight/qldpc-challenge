"""Witness-preserving screen of three independent candidate generators."""

import argparse
import copy
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

from candidate_search import ADAPTER_DIRS, METHODS, Search, load_candidates
from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from run import initialize_validation, np, validate_and_stage, validation_ready
from study_external import run_side
from study_strategies import CASES, snapshot, summarize

HARD = CASES[:3]


def create_corpus(directory):
    """Copy frozen inputs and relabel reference supports only for evaluation."""
    directory.mkdir(parents=True, exist_ok=False)
    original = HERE / "results/reference-corpus"
    reference = json.loads((HERE / "results/external-refresh/t1-10s/manifest.json").read_text())
    cases, transforms = [], {}
    for source in reference["cases"]:
        with np.load(original / source["file"], allow_pickle=False) as data:
            hx, hz = data["hx"], data["hz"]
        if matrix_hash(hx, hz) != source["matrix_sha256"]:
            raise ValueError("Source matrix hash mismatch")
        cases.append(copy.deepcopy(source))
        shutil.copyfile(original / source["file"], directory / source["file"])
        if source["id"] not in HARD:
            continue
        rng = np.random.default_rng(1230)
        columns = rng.permutation(hx.shape[1])
        inverse = np.argsort(columns)
        for variant in ("columns", "columns-rows"):
            rows_x = rng.permutation(hx.shape[0]) if variant == "columns-rows" else np.arange(hx.shape[0])
            rows_z = rng.permutation(hz.shape[0]) if variant == "columns-rows" else np.arange(hz.shape[0])
            x, z = hx[rows_x][:, columns], hz[rows_z][:, columns]
            case = copy.deepcopy(source)
            case.update(
                id=source["id"] + "-" + variant,
                source_case=source["id"],
                variant=variant,
                matrix_sha256=matrix_hash(x, z),
            )
            case["file"] = case["id"] + ".npz"
            for field in ("reference", "previous_reference"):
                if field in case:
                    case[field] = {s: sorted(inverse[case[field][s]].tolist()) for s in ("X", "Z")}
            cases.append(case)
            np.savez_compressed(directory / case["file"], hx=x, hz=z)
            transforms[case["id"]] = dict(
                source_case=source["id"], columns=columns.tolist(), rows_x=rows_x.tolist(), rows_z=rows_z.tolist()
            )
    atomic_json(directory / "manifest.json", dict(sources=reference["sources"], cases=cases))
    atomic_json(directory / "transforms.json", transforms)


def archive_sources(output):
    sources, binaries = snapshot(output, ["guided", "descent"])
    extra = [
        Path(__file__),
        HERE / "candidate_search.py",
        HERE / "initialized_search.py",
        HERE / "study_external.py",
        HERE / "report_candidates.py",
        HERE / "test_candidates.py",
        HERE / "external_search.py",
        HERE / "workers.py",
        HERE / "setup_native.py",
        ROOT / "verify/gf2_fast.cpp",
        ROOT / "verify/validator_manifest.json",
        ROOT / "schema/code.schema.json",
        HERE / "polynomial_completion.py",
        HERE / "orbit_completion.py",
        HERE / "strategy_prototypes/CANDIDATES_PLAN.md",
    ]
    extra.extend((ROOT / "verify").glob("*.py"))
    extra.extend((ROOT / "research/kit").glob("*.py"))
    for name in ADAPTER_DIRS.values():
        folder = HERE / "strategy_prototypes" / name
        extra.extend(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".cpp", ".hpp", ".h", ".md", ".toml", ".txt")
            and not {"build", "__pycache__", ".pytest_cache"}.intersection(p.relative_to(folder).parts)
        )
        binaries.update({str(p.relative_to(ROOT)): sha256(p) for p in folder.glob("*.so")})
    import ldpc

    for p in Path(ldpc.__file__).parent.rglob("*.so"):
        binaries[str(p.relative_to(ROOT))] = sha256(p)
    with tarfile.open(output / "candidate-sources.tar.gz", "w:gz") as archive:
        for p in sorted(set(extra)):
            relative = str(p.relative_to(ROOT))
            sources[relative] = sha256(p)
            archive.add(p, arcname=relative)
    return sources, binaries


def main(args):
    if args.seconds <= 0 or args.seeds < 1 or args.validation_workers < 1:
        raise ValueError("Positive budget and counts required")
    if len(set(args.methods)) != len(args.methods) or len(set(args.cases)) != len(args.cases):
        raise ValueError("Duplicate method or case")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    indexed = {c["id"]: c for c in manifest["cases"]}
    cases = [indexed[c] for c in args.cases]
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu})
    adapters = load_candidates()
    warmups = {m: a.warmup() for m, a in adapters.items() if hasattr(a, "warmup")}
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
        warmups=warmups,
        threadpools=pools,
        package_versions={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "ldpc", "pybind11")},
        reference_policy="Only matrices and prepared logical detectors enter search; no reference targets",
        timing="Preparation, initialization, analysis, search and witness delivery inside half-budget per side",
        fallback="Finite/inapplicable candidate returns use remaining time for guided; all stages retained",
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
                random.Random(f"candidates-v2:{case['id']}:{seed}").shuffle(methods)
                for method in methods:
                    folder = args.output / case["id"] / f"{method}-s{seed}"
                    folder.mkdir(parents=True)
                    sides = {}
                    for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
                        side_seed = seed * 1000003 + (499979 if side == "Z" else 0)
                        sides[side] = [
                            run_side(
                                Search(own, opposite, method, adapters),
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
    parser.add_argument("--corpus", type=Path, default=HERE / "results/candidate-study-v2/corpus")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1200)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=8)
    args = parser.parse_args()
    if args.make_corpus:
        create_corpus(args.make_corpus)
    elif args.output:
        main(args)
    else:
        parser.error("Provide --output or --make-corpus")
