"""Audit CRT completeness, native accounting, saved witnesses and deadline credit."""

import argparse
import json
from pathlib import Path

from run import gf2, np

# isort: split
from audit_candidates import INITIAL, archive_hashes, check_initialization, require
from audit_dispatch import audit_worker
from audit_strategies import main as standard_audit
from common import HERE, atomic_json
from initialized_search import pack_rows
from report_strategies import best, checked_records
from strategy_prototypes.polynomial_weight.adapter import CONFIG
from strategy_prototypes.polynomial_weight.module import build


def main(directory):
    env, records = checked_records(directory)
    archived = {}
    for path in directory.glob("*.tar.gz"):
        if path.name == "supplemental-sources.tar.gz":
            continue
        for key, digest in archive_hashes(path).items():
            require(key not in archived or archived[key] == digest, "Conflicting archives")
            archived[key] = digest
    require(archived == env["source_hashes"], "Source archive mismatch")
    initial = check_initialization(records)
    structures = json.loads((HERE / "strategy_prototypes/polynomial_weight/structure.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    matrices, layouts, rowspaces = {}, {}, {}
    for case in manifest["cases"]:
        with np.load(directory / "matrices" / case["file"]) as d:
            hx, hz = d["hx"], d["hz"]
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            key = case["id"], side
            matrices[key] = pack_rows(opposite)
            reduced, pivots = gf2.rref(own)
            rowspaces[key] = list(zip(pivots, pack_rows(reduced)))
            structure = structures[case["id"]]
            basis, widths, groups = build(opposite, structure["length"], [int(f, 16) for f in structure["factors"]])
            layouts[key] = len(basis), widths, groups
    count, sessions = 0, []
    for r in records:
        for side in ("X", "Z"):
            key = r["case"], side
            w = r["workers"][side][0]
            times = [e["seconds"] for e in w["events"]]
            require(times == sorted(times) and all(0 <= t <= w["search_seconds"] for t in times), "Event time")
            # Independently check every raw event using trusted reduced rowspace,
            # not the experimental module basis or its logical tags.
            for e in w["events"]:
                support = e["support"]
                require(support == sorted(set(support)) and len(support) == e["weight"], "Weight/support")
                n = next(c["n"] for c in manifest["cases"] if c["id"] == r["case"])
                require(support and min(support) >= 0 and max(support) < n, "Coordinate range")
                word = sum(1 << q for q in support)
                require(not any((word & check).bit_count() & 1 for check in matrices[key]), "Nonzero syndrome")
                reduced = word
                for pivot, row in rowspaces[key]:
                    if reduced >> pivot & 1:
                        reduced ^= row
                require(reduced != 0, "Trivial logical witness")
                count += 1
            if r["method"] == "guided":
                audit_worker(r, side)
                continue
            c = w["counters"]
            require(c["parameters"] == CONFIG, "Policy differs")
            dim, widths, groups = layouts[key]
            require(c["dimension"] == dim and c["groups"] == groups, "Full kernel/group layout mismatch")
            if r["method"] == "binary-groups":
                require(sorted(c["binary_order"]) == list(range(dim)), "Invalid binary ablation order")
            es = [e for e in w["events"] if e["stage"] not in INITIAL]
            require(all(e["stage"] == r["method"] for e in es), "Unexpected fallback/attribution")
            weights = [e["weight"] for e in es]
            require(weights == sorted(set(weights), reverse=True), "Nonimproving native exports")
            native = c["native"]
            require(native["exports"] == len(es) and native["best"] == min(weights, default=None), "Export count")
            require(native["groups"] == len(widths), "Native group count")
            require(native["logical_words"] == 3, "Lost high logical bits")
            require(all(e["seconds"] >= c["setup_finished_seconds"] for e in es), "Export before setup")
            cycles, remainder = divmod(native["updates"], len(widths))
            per_pass = sum((1 << width) - 1 for width in widths)
            lower = cycles * per_pass + sum(sorted((1 << width) - 1 for width in widths)[:remainder])
            upper = cycles * per_pass + sum(sorted(((1 << width) - 1 for width in widths), reverse=True)[:remainder])
            require(lower <= native["scored"] <= upper, "Score accounting")
            sessions.append(
                dict(case=r["case"], method=r["method"], side=side, seed=r["seed"], initial=c["initial_best"], **native)
            )
    standard_audit(directory)
    completed = json.loads((directory / "completed.json").read_text())
    require(completed["configurations"] == len(records), "Completion mismatch")
    require(completed["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Save count")
    cells = []
    for case in env["cases"]:
        for method in env["methods"]:
            rs = sorted((r for r in records if r["case"] == case and r["method"] == method), key=lambda r: r["seed"])
            cells.append(dict(case=case, method=method, weights=[best(r) for r in rs]))
    atomic_json(
        directory / "audit-polynomial.json",
        dict(
            status="passed",
            configurations=len(records),
            saved_documents=completed["saved_candidates"],
            independently_checked_events=count,
            initialization=initial,
            sessions=sessions,
        ),
    )
    atomic_json(directory / "summary.json", dict(cells=cells))
    print(
        json.dumps(dict(status="passed", configurations=len(records), saved=completed["saved_candidates"], cells=cells))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    main(p.parse_args().directory.resolve())
