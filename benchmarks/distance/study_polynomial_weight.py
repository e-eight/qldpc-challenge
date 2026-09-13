"""Frozen full-kernel polynomial weight comparison."""

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

from run import initialize_validation, np, validate_and_stage, validation_ready

# isort: split
from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from strategy_prototypes.polynomial_weight.adapter import CONFIG, METHODS, Search, native
from study_components import archive_sources as prior_archive
from study_external import run_side
from study_strategies import summarize


def archive_sources(output):
    sources, binaries = prior_archive(output)
    extra = [HERE / "study_polynomial_weight.py", HERE / "audit_polynomial_weight.py"]
    prototype = HERE / "strategy_prototypes/polynomial_weight"
    extra += [p for p in prototype.iterdir() if p.suffix in (".py", ".cpp", ".md", ".json")]
    extra += [HERE / "strategy_prototypes/algebra_diagnostic/algebra.py"]
    with tarfile.open(output / "polynomial-weight-sources.tar.gz", "w:gz") as archive:
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
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if not args.cases or c["id"] in args.cases]
    manifest = dict(manifest, cases=cases)
    structure = json.loads((HERE / "strategy_prototypes/polynomial_weight/structure.json").read_text())
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "manifest.json", manifest)
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
        cases=[c["id"] for c in cases],
        checkpoints=[args.seconds],
        source_hashes=sources,
        binary_hashes=binaries,
        parameters=CONFIG,
        threadpools=pools,
        reference_policy="Only matrices, method, verified lift and factors enter Search; no reference witnesses",
        timing=(
            "Preparation, initialization, CRT lifting and binary validation, table building, scoring "
            "and export included; validation between runs; prior lift discovery and factorization excluded"
        ),
        random_seed_policy="side seed = index * 1000003 + (0 for X, 499979 for Z)",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    atomic_json(args.output / "environment.json", env)
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.validation_workers)
    records = []
    with ProcessPoolExecutor(
        args.validation_workers, mp_context=context, initializer=initialize_validation, initargs=(available, barrier)
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
                random.Random(f"polynomial-weight-v1:{case['id']}:{seed}").shuffle(methods)
                for method in methods:
                    folder = args.output / case["id"] / f"{method}-s{seed}"
                    folder.mkdir(parents=True)
                    sides = {}
                    for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
                        side_seed = seed * 1000003 + (499979 if side == "Z" else 0)
                        sides[side] = [
                            run_side(
                                Search(own, opposite, method, structure[case["id"]]),
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
    parser.add_argument("--corpus", type=Path, default=HERE / "results/component-study/corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--cases", nargs="+", default=["682-182-76", "664-170-18"])
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=8)
    main(parser.parse_args())
