"""Audit and summarize the frozen full/block x pairs/Stern comparison."""

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
from strategy_prototypes.block_collision.adapter import CONFIG


def audit_worker(record, side, n):
    if record["method"] == "guided":
        control_worker(record, side)
        return []
    worker = record["workers"][side][0]
    c, events = worker["counters"], worker["events"]
    require(c["parameters"] == CONFIG and c["method"] == record["method"], "Wrong policy")
    require([e["seconds"] for e in events] == sorted(e["seconds"] for e in events), "Unordered exports")
    stages = [e["stage"] in INITIAL for e in events]
    require(stages == sorted(stages, reverse=True), "Initialization after search")
    spaces = {s["label"]: s for s in c["spaces"]}
    require(len(spaces) == len(c["spaces"]), "Duplicate spaces")
    for e in events:
        require(0 <= e["seconds"] <= worker["search_seconds"], "Event outside worker")
        if e["stage"] not in INITIAL:
            label, kind = e["stage"].split(":")
            require(label in spaces and kind in {"single", "pair", "collision4", "enumeration"}, "Unknown stage")
            require(set(e["support"]) <= set(spaces[label]["coordinates"]), "Witness outside its block")
            require(kind != "collision4" or "stern" in record["method"], "Collision in pair control")
            require(kind != "enumeration" or record["method"] == "block-exact", "Unexpected exhaustive search")
    summaries = []
    for label, space in spaces.items():
        expected = (
            list(range(n))
            if label == "full"
            else list(range(int(label[-1]) * (n // 2), (int(label[-1]) + 1) * (n // 2)))
        )
        require(space["coordinates"] == expected, "Wrong layout")
        es = [e for e in events if e["stage"].startswith(label + ":")]
        if "stats" not in space:
            require(not es and space["status"] in {"dimension_cap", "no_logicals"}, "Missing session stats")
            continue
        s = space["stats"]
        require(s["dimension"] == space["dimension"], "Dimension mismatch")
        require(s["exports"] == len(es), "Export count mismatch")
        require(s["best"] == min((e["weight"] for e in es), default=None), "Best mismatch")
        weights = [e["weight"] for e in es]
        require(weights == sorted(set(weights), reverse=True), "Non-improving native export")
        d = s["dimension"]
        require(s["single_scores"] == s["reductions"] * d, "Single count mismatch")
        require(s["pair_scores"] == s["reductions"] * d * (d - 1) // 2, "Pair count mismatch")
        require("stern" in record["method"] or s["collisions"] == 0, "Collisions in control")
        if record["method"] == "block-exact":
            require(s["reductions"] == 0 and d <= CONFIG["enumeration_cap"], "Invalid exact session")
            if s["enumeration_complete"]:
                require(space["status"] == "exhausted", "Completion status mismatch")
                require(s["enumerated"] == (1 << d) - 1, "Incomplete Gray traversal")
                require(
                    s["enumerated_nontrivial"] == (1 << d) - (1 << (d - space["logical_rank"])),
                    "Logical count mismatch",
                )
        summaries.append(
            dict(case=record["case"], method=record["method"], seed=record["seed"], side=side, label=label, **s)
        )
    return summaries


def main(directory):
    directory = directory.resolve()
    require((directory / "completed.json").exists(), "Incomplete grid")
    env, records = checked_records(directory)
    complete = json.loads((directory / "completed.json").read_text())
    require(complete["configurations"] == len(records), "Completion count mismatch")
    require(complete["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Save count mismatch")
    archived = {}
    for name in (
        "sources.tar.gz",
        "candidate-sources.tar.gz",
        "dispatch-sources.tar.gz",
        "allocation-sources.tar.gz",
        "large-structure-sources.tar.gz",
        "block-collision-sources.tar.gz",
    ):
        archived.update(archive_hashes(directory / name))
    require(archived == env["source_hashes"], "Source archive mismatch")
    require(env["threads"] == 1 and all(p["num_threads"] == 1 for p in env["threadpools"]), "Thread mismatch")
    initial = check_initialization(records)
    manifest = json.loads((directory / "manifest.json").read_text())
    ns = {}
    for case in manifest["cases"]:
        with np.load(directory / "matrices" / case["file"], allow_pickle=False) as data:
            ns[case["id"]] = data["hx"].shape[1]
    sessions = [s for r in records for side in ("X", "Z") for s in audit_worker(r, side, ns[r["case"]])]
    standard_audit(directory)
    standard = json.loads((directory / "audit.json").read_text())
    cells = []
    lines = [
        "# Block/collision comparison",
        "",
        f"{len(records)} configurations, {env['seconds_per_code']:g} seconds/code, {env['seeds']} seeds; one CPU.",
        "",
        "Targets are witnessed upper bounds, including the prior affine-90 result. Lower weight is better. "
        "All times use the harness clock; half the code budget goes to each sector.",
        "",
        "| Code | Method | Target | Final weights | Target hits | Collision-caused improvements |",
        "|---|---|---:|---|---:|---:|",
    ]
    for case in env["cases"]:
        for method in env["methods"]:
            rs = sorted((r for r in records if r["case"] == case and r["method"] == method), key=lambda r: r["seed"])
            collision_improvements = sum(
                e["stage"].endswith(":collision4") and e["seconds"] <= r["budget_seconds"] / 2
                for r in rs
                for ws in r["workers"].values()
                for e in ws[0]["events"]
            )
            cell = dict(
                case=case,
                method=method,
                target=rs[0]["target"],
                weights=[best(r) for r in rs],
                hits=sum(best(r) <= r["target"] for r in rs),
                collision_improvements=collision_improvements,
            )
            cells.append(cell)
            lines.append(
                f"| {case} | {method} | {cell['target']} | {cell['weights']} | "
                f"{cell['hits']}/{len(rs)} | {collision_improvements} |"
            )
    lines += [
        "",
        "Collision improvement counts are per-session improvements, "
        "not necessarily improvements over common initialization. "
        "Exact completion applies only to individual enumerated blocks; larger blocks are skipped.",
        "",
        f"All {standard['saved_witness_documents_checked']:,} saved documents audited. Source/binary/archive, "
        "matrices, initialization, counters, block membership, native improvement order "
        "and exact traversal counts checked. "
        "Trusted algebra validates witnesses. No full candidate gate or full-code distance certificate.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(directory / "summary.json", dict(cells=cells))
    atomic_json(
        directory / "audit-block-collision.json",
        dict(
            status="passed",
            configurations=len(records),
            saved_witness_documents_checked=standard["saved_witness_documents_checked"],
            initialization=initial,
            source_binary_archive_hashes_match=True,
            sessions=sessions,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
