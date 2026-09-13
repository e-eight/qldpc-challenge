"""Post-hoc deterministic order-six probe; fixed before testing sibling codes.

Enumerate all words in the same left-y^3 orbit-constant kernel that succeeded in
the main affine experiment. Exactness applies ONLY to this restricted space.
"""

import argparse
import json
import os
import platform
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "benchmarks/distance"))
from run import np, validate_and_stage  # noqa: E402

# isort: split
from audit_candidates import check_initialization, require  # noqa: E402
from audit_strategies import main as standard_audit  # noqa: E402
from common import atomic_json, matrix_hash, sha256  # noqa: E402
from corpus import supports_matrix  # noqa: E402
from initialized_search import LogicalPool, initialize, pack_rows, support_of  # noqa: E402
from strategy_prototypes.large_structure.adapter import native, proposals  # noqa: E402
from study_external import run_side  # noqa: E402
from study_large_structure import archive_sources  # noqa: E402
from study_strategies import ris_native, summarize  # noqa: E402

SPEC = dict(kind="affine", prime=19, order=18, action=2)
CASES = ["684-10-101", "684-12-73", "684-8-85"]
LABEL = "orbits_left_0_3"
MAX_DIMENSION = 20


class ExactSearch:
    def __init__(self, own, opposite):
        self.own, self.opposite = own, opposite

    def run(self, seconds, seed, emit):
        started = time.perf_counter()
        deadline = started + seconds
        prepared = ris_native.Prepared(self.own, self.opposite)
        rows = ris_native.Prepared(self.opposite, self.own).logicals
        counters = dict(preparation_seconds=time.perf_counter() - started)
        best = self.own.shape[1] + 1

        def observe(word, stage, improving_only=False):
            nonlocal best
            weight = word.bit_count()
            if improving_only and weight >= best:
                return
            best = min(best, weight)
            emit(weight, support_of(word), stage)

        tick = time.perf_counter()
        counters["initialization"] = initialize(rows, LogicalPool(prepared.logicals, 8), observe, deadline)
        counters["initialization_seconds"] = time.perf_counter() - tick
        counters["initial_best"] = best
        groups = next(groups for label, groups in proposals(SPEC, self.own.shape[1]) if label == LABEL)
        space = native.Space(self.opposite, prepared.logicals, groups)
        counters.update(
            groups=groups,
            dimension=space.dimension,
            logical_rank=space.logical_rank,
            label=LABEL,
            exhaustive=False,
            visited=0,
            nontrivial_visited=0,
            restricted_best=None,
            restricted_exports=0,
        )
        if space.dimension > MAX_DIMENSION or time.perf_counter() >= deadline:
            return dict(counters, status="dimension_or_time_cap")
        basis = pack_rows(space.basis)
        duals = pack_rows(prepared.logicals)
        tags = [sum(((word & d).bit_count() % 2) << i for i, d in enumerate(duals)) for word in basis]
        word = tag = 0
        restricted_best = self.own.shape[1] + 1
        tick = time.perf_counter()
        for index in range(1, 1 << len(basis)):
            if index % 1024 == 0 and time.perf_counter() >= deadline:
                break
            bit = (index & -index).bit_length() - 1
            word ^= basis[bit]
            tag ^= tags[bit]
            counters["visited"] += 1
            if tag:
                counters["nontrivial_visited"] += 1
                weight = word.bit_count()
                if weight < restricted_best:
                    emit(weight, support_of(word), LABEL)
                    restricted_best = weight
                    counters["restricted_best"] = weight
                    counters["restricted_exports"] += 1
        counters["exhaustive"] = counters["visited"] == (1 << len(basis)) - 1
        counters["enumeration_seconds"] = time.perf_counter() - tick
        return dict(counters, status="completed", elapsed_seconds=time.perf_counter() - started)


def main(output):
    os.sched_setaffinity(0, {0})
    output.mkdir(parents=True, exist_ok=False)
    (output / "matrices").mkdir()
    sources, binaries = archive_sources(output)
    with tarfile.open(output / "finite-probe-source.tar.gz", "w:gz") as archive:
        for path in (Path(__file__), Path(__file__).with_name("test_finite_probe.py")):
            relative = str(path.resolve().relative_to(REPO))
            sources[relative] = sha256(path)
            archive.add(path, arcname=relative)
    atomic_json(
        output / "environment.json",
        dict(
            methods=["exact"],
            cases=CASES,
            seeds=1,
            seed_start=0,
            seconds_per_code=4,
            checkpoints=[4],
            threads=1,
            cpu=0,
            source_hashes=sources,
            binary_hashes=binaries,
            python=sys.version,
            platform=platform.platform(),
            scope="Post-hoc order-six exhaustive restricted-space check and two sibling controls",
        ),
    )
    records = []
    cases = []
    for name in CASES:
        source = REPO / "codes" / f"{name}.json"
        doc = json.loads(source.read_text())
        hx, hz = [supports_matrix(doc["checks"][s], doc["n"]) for s in ("X", "Z")]
        case = dict(
            id=name,
            n=doc["n"],
            k=doc["k"],
            file=f"{name}.npz",
            source=str(source.relative_to(REPO)),
            source_sha256=sha256(source),
            matrix_sha256=matrix_hash(hx, hz),
            family=doc.get("family", ""),
            reference={s: doc["distance"][s]["witness"] for s in ("X", "Z")},
        )
        cases.append(case)
        atomic_json(output / "manifest.json", dict(sources={"repository": "local frozen board"}, cases=cases))
        np.savez_compressed(output / "matrices" / case["file"], hx=hx, hz=hz)
        folder = output / name / "exact-s0"
        folder.mkdir(parents=True)
        sides = {}
        for side, own, opposite in [("X", hx, hz), ("Z", hz, hx)]:
            sides[side] = [run_side(ExactSearch(own, opposite), 2, 0, folder / f"{side}.jsonl")]
        target = min(len(s) for s in case["reference"].values())
        record = dict(
            case=name,
            method="exact",
            seed=0,
            budget_seconds=4,
            target=target,
            workers=sides,
            sides={s: summarize(sides[s][0]["events"], 2, target) for s in sides},
            validation_status="pending",
        )
        atomic_json(folder / "result.json", record)
        fallback = {s: sides[s][0]["events"][0]["support"] for s in sides}
        duration, saved = validate_and_stage(hx, hz, case, sides, fallback, output.name + "-exact-s0")
        record.update(validation_status="passed", saved_candidates=saved, validation_seconds=duration)
        atomic_json(folder / "result.json", record)
        records.append(record)
        atomic_json(output / "results.json", records)
        for side, ws in sides.items():
            c = ws[0]["counters"]
            es = [e for e in ws[0]["events"] if e["stage"] == LABEL]
            require(c["restricted_exports"] == len(es), "Export count mismatch")
            require(c["restricted_best"] == min((e["weight"] for e in es), default=None), "Restricted best mismatch")
            require(c["exhaustive"] and c["visited"] == (1 << c["dimension"]) - 1, "Incomplete enumeration")
            require(
                c["nontrivial_visited"] == (1 << c["dimension"]) - (1 << (c["dimension"] - c["logical_rank"])),
                "Wrong number of nontrivial words",
            )
            require(
                all(not (set(e["support"]) & set(g)) or set(g) <= set(e["support"]) for e in es for g in c["groups"]),
                "Not orbit constant",
            )
            print(
                name,
                side,
                {
                    key: c[key]
                    for key in ("dimension", "logical_rank", "restricted_best", "visited", "enumeration_seconds")
                },
                flush=True,
            )
    atomic_json(
        output / "completed.json",
        dict(configurations=len(records), saved_candidates=sum(r["saved_candidates"] for r in records)),
    )
    initial = check_initialization(records)
    standard_audit(output)
    atomic_json(
        output / "finite-audit.json",
        dict(
            status="passed",
            scope="Exhaustive only inside a fixed orbit-constant subspace",
            configurations=len(records),
            initialization=initial,
            enumeration_counts_match=True,
            all_exports_saved=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output.resolve())
