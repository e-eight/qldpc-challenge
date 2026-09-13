"""Separate persisted exploratory comparison of supplied and inferred blocks."""

import argparse
import json
import multiprocessing as mp
import os
import platform
import random
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from candidate_search import Search  # noqa: E402
from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256  # noqa: E402
from run import initialize_validation, np, validate_and_stage, validation_ready  # noqa: E402
from study_candidates import HARD, archive_sources  # noqa: E402
from study_external import run_side  # noqa: E402
from study_strategies import load_adapter, summarize  # noqa: E402

METHODS = ["circulant", "routed"]


def main(args):
    if args.seconds <= 0 or args.seed < 0 or args.validation_workers < 1:
        raise ValueError("Positive time/workers and nonnegative seed required")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    indexed = {case["id"]: case for case in manifest["cases"]}
    names = [name + suffix for name in HARD for suffix in ("", "-columns", "-columns-rows")]
    cases = [indexed[name] for name in names]
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu})
    routed = load_adapter("structure_route_v2")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "manifest.json", dict(sources=manifest["sources"], cases=cases))
    shutil.copyfile(args.corpus / "transforms.json", args.output / "transforms.json")
    sources, binaries = archive_sources(args.output)
    with tarfile.open(args.output / "route-sources.tar.gz", "w:gz") as archive:
        for path in sorted(Path(__file__).parent.iterdir()):
            if not path.is_file() or path.suffix not in (".py", ".md"):
                continue
            relative = str(path.relative_to(ROOT))
            sources[relative] = sha256(path)
            archive.add(path, arcname=relative)
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
        seed_start=args.seed,
        seeds=1,
        methods=METHODS,
        cases=names,
        checkpoints=[args.seconds],
        source_hashes=sources,
        binary_hashes=binaries,
        threadpools=pools,
        scope="Separate exploratory route diagnostic, after frozen main comparison",
        override={
            "recorded_method": "routed",
            "wrapper_method": "matrix-structure",
            "adapter_mapping": {"matrix-structure": "structure_route_v2"},
        },
        timing="Equal X/Z split; preparation, initialization, recovery, native setup, search and delivery charged",
        fallback="Reuse unchanged candidate_search.Search; unused route budget goes to fresh guided",
        reference_policy="Only matrices and native logical detectors enter search; no evaluation targets/supports",
        seed_policy="side seed = seed*1000003 + (0 for X, 499979 for Z)",
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
            methods = METHODS.copy()
            random.Random(f"structure-route:{case['id']}:{args.seed}").shuffle(methods)
            for method in methods:
                folder = args.output / case["id"] / f"{method}-s{args.seed}"
                folder.mkdir(parents=True)
                sides = {}
                for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
                    side_seed = args.seed * 1000003 + (499979 if side == "Z" else 0)
                    wrapper_method = "matrix-structure" if method == "routed" else "circulant"
                    engine = Search(own, opposite, wrapper_method, {"matrix-structure": routed})
                    sides[side] = [run_side(engine, args.seconds / 2, side_seed, folder / f"{side}.jsonl")]
                record = dict(
                    case=case["id"],
                    method=method,
                    seed=args.seed,
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
                    hx,
                    hz,
                    case,
                    sides,
                    fallback,
                    f"{args.output.name}-{method}-s{args.seed}",
                    executor,
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
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path, default=HERE / "results/candidate-study-v2/corpus")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=8)
    main(parser.parse_args())
