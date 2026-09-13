"""Replay component geometry/algebra and audit all kit-saved witnesses."""

import argparse
import json
from pathlib import Path

from run import np

# isort: split
from audit_candidates import INITIAL, archive_hashes, check_initialization, require
from audit_dispatch import audit_worker as control_worker
from audit_strategies import main as standard_audit
from common import atomic_json
from report_strategies import best, checked_records
from strategy_prototypes.component_search.adapter import CONFIG, components, compress_duals, restriction, ris_native


def main(directory):
    directory = directory.resolve()
    env, records = checked_records(directory)
    complete = json.loads((directory / "completed.json").read_text())
    require(complete["configurations"] == len(records), "Incomplete grid")
    require(complete["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Save count")
    archived = {}
    for name in (
        "sources",
        "candidate-sources",
        "dispatch-sources",
        "allocation-sources",
        "large-structure-sources",
        "block-collision-sources",
        "component-sources",
    ):
        for path, digest in archive_hashes(directory / (name + ".tar.gz")).items():
            require(path not in archived or archived[path] == digest, "Conflicting archive")
            archived[path] = digest
    require(archived == env["source_hashes"], "Source archive mismatch")
    initial = check_initialization(records)
    cases = {c["id"]: c for c in json.loads((directory / "manifest.json").read_text())["cases"]}
    matrices = {}
    prepared = {}
    cache = {}
    sessions = []
    for case in cases.values():
        with np.load(directory / "matrices" / case["file"]) as d:
            matrices[case["id"]] = (d["hx"], d["hz"])
        for side, own, opposite in [("X", *matrices[case["id"]]), ("Z", *matrices[case["id"]][::-1])]:
            prepared[case["id"], side] = ris_native.Prepared(own, opposite)
    for r in records:
        for side in ("X", "Z"):
            if r["method"] == "guided":
                control_worker(r, side)
                continue
            w = r["workers"][side][0]
            c = w["counters"]
            events = w["events"]
            case = cases[r["case"]]
            require(c["parameters"] == CONFIG, "Wrong policy")
            proposals = {p["label"]: p["coordinates"] for p in c["proposals"]}
            n = case["n"]
            require(proposals["whole"] == list(range(n)), "Wrong full graph")
            if r["method"] == "auto-components":
                accepted = [d for d in c["detection"] if d["status"] == "accepted"]
                for d in accepted:
                    require(sorted(d["order"]) == list(range(n)), "Non-bijective detector order")
                    require(proposals[d["source"] + "0"] == sorted(d["order"][: n // 2]), "Wrong detected half")
                    require(proposals[d["source"] + "1"] == sorted(d["order"][n // 2 :]), "Wrong detected half")
                require(len(proposals) == 1 + 2 * len(accepted), "Unexpected automatic proposal")
            else:
                require(
                    proposals
                    == dict(whole=list(range(n)), **{f"metadata{i}": b for i, b in enumerate(case["blocks"])}),
                    "Metadata differs",
                )
            opposite = matrices[r["case"]][1 if side == "X" else 0]
            duals = prepared[r["case"], side].logicals
            spaces = {s["label"]: s for s in c["spaces"]}
            require(len(spaces) == len(c["spaces"]), "Duplicate labels")
            for s in spaces.values():
                key = (r["case"], side, tuple(s["coordinates"]))
                geometry = (r["case"], side, tuple(proposals[s["proposal"]]))
                if geometry not in cache:
                    cache[geometry] = components(opposite, proposals[s["proposal"]])
                require(s["coordinates"] in cache[geometry], "Not a connected component")
                algebra = ("algebra",) + key
                if algebra not in cache:
                    space = restriction.Space(opposite, duals, [[q] for q in s["coordinates"]])
                    columns = compress_duals(space.basis, duals)[1] if space.dimension <= 64 else None
                    cache[algebra] = (space.dimension, space.logical_rank, columns)
                dim, rank, columns = cache[algebra]
                require((s["dimension"], s["logical_rank"]) == (dim, rank), "Dimension/rank mismatch")
                es = [e for e in events if e["stage"].startswith(s["label"] + ":")]
                if "stats" not in s:
                    require(not es and s["status"] in ("no_logicals", "dimension_cap"), "Skipped space exported")
                    continue
                require(s["dual_columns"] == columns and len(columns) == rank, "Lost logical information")
                stats = s["stats"]
                weights = [e["weight"] for e in es]
                require(weights == sorted(set(weights), reverse=True), "Non-improving native exports")
                require(
                    stats["exports"] == len(es) and stats["best"] == min(weights, default=None), "Export accounting"
                )
                require(stats["collisions"] == 0, "Unexpected collisions")
                require(stats["single_scores"] == stats["reductions"] * dim, "Singles count")
                require(stats["pair_scores"] == stats["reductions"] * dim * (dim - 1) // 2, "Pairs count")
                if s["exact"]:
                    require(dim <= 20 and stats["reductions"] == 0, "Wrong exact policy")
                    if stats["enumeration_complete"]:
                        require(s["status"] == "exhausted", "Missing exhaustion")
                        require(stats["enumerated"] == 2**dim - 1, "Incomplete Gray traversal")
                        require(stats["enumerated_nontrivial"] == 2**dim - 2 ** (dim - rank), "Logical traversal count")
                sessions.append(dict(case=r["case"], method=r["method"], seed=r["seed"], side=side, **s))
            for e in events:
                if e["stage"] in INITIAL or e["stage"] == "guided":
                    continue
                label, kind = e["stage"].split(":")
                s = spaces[label]
                require(set(e["support"]) <= set(s["coordinates"]), "Export outside component")
                require(kind in (("enumeration",) if s["exact"] else ("single", "pair")), "Wrong search stage")
    standard_audit(directory)
    cells = []
    lines = ["# Component comparison", "", "| Case | Method | Target | Seed weights |", "|---|---|---:|---|"]
    for case in env["cases"]:
        for method in env["methods"]:
            rs = sorted([r for r in records if r["case"] == case and r["method"] == method], key=lambda r: r["seed"])
            cell = dict(case=case, method=method, target=rs[0]["target"], weights=[best(r) for r in rs])
            cells.append(cell)
            lines.append(f"| {case} | {method} | {cell['target']} | {cell['weights']} |")
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(directory / "summary.json", dict(cells=cells))
    atomic_json(
        directory / "audit-components.json",
        dict(
            status="passed",
            configurations=len(records),
            initialization=initial,
            saved_documents=complete["saved_candidates"],
            sessions=sessions,
        ),
    )
    print("Audited", len(records), "configurations and", complete["saved_candidates"], "saved witnesses")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    main(p.parse_args().directory)
