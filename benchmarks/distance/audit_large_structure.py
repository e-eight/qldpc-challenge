"""Audit the bounded large-code probe and report target recovery, including misses."""

import argparse
import json
import statistics
from pathlib import Path

from run import np

# isort: split
from audit_allocation import audit_worker as control_worker
from audit_candidates import INITIAL, archive_hashes, check_initialization, check_transforms, require
from audit_strategies import main as standard_audit
from common import atomic_json
from report_candidates import best
from report_strategies import checked_records
from strategy_prototypes.large_structure.adapter import CONFIG


def audit_worker(record, side):
    if record["method"] != "structure":
        control_worker(record, side)
        return 0
    worker = record["workers"][side][0]
    c = worker["counters"]
    events = worker["events"]
    require(c["parameters"] == CONFIG, "Policy mismatch")
    times = [e["seconds"] for e in events]
    require(times == sorted(times) and all(0 <= t <= worker["search_seconds"] for t in times), "Event time mismatch")
    spaces = {s["label"]: s for s in c["spaces"]}
    require(len(spaces) == len(c["spaces"]), "Duplicate space label")
    stage_order = [0 if e["stage"] in INITIAL else 2 if e["stage"] == "guided" else 1 for e in events]
    require(stage_order == sorted(stage_order), "Stage order mismatch")
    require(all(e["stage"] in INITIAL | {"guided"} | set(spaces) for e in events), "Unexpected stage")
    exports = 0
    for label, s in spaces.items():
        es = [e for e in events if e["stage"] == label]
        require(s["exports"] == len(es), "Export count mismatch")
        require(s["best"] == min((e["weight"] for e in es), default=None), "Best weight mismatch")
        require(s["batches"] >= len(es), "More exports than batches")
        groups = [set(g) for g in s["groups"]]
        flat = [q for g in s["groups"] for q in g]
        require(len(flat) == len(set(flat)), "Overlapping encoder supports")
        for e in es:
            support = set(e["support"])
            require(support <= set(flat), "Witness outside proposed space")
            require(all(not support & g or g <= support for g in groups), "Witness violates orbit constancy")
        if es:
            require(s["logical_rank"] > 0, "Export from trivial space")
        exports += len(es)
    require(c["active_spaces"] == sum(s["logical_rank"] > 0 for s in c["spaces"]), "Active space mismatch")
    return exports


def main(directory):
    directory = directory.resolve()
    require((directory / "completed.json").exists(), "Incomplete grid")
    env, records = checked_records(directory)
    complete = json.loads((directory / "completed.json").read_text())
    require(complete["configurations"] == len(records), "Completion mismatch")
    require(complete["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Save total mismatch")
    archives = {}
    for name in (
        "sources.tar.gz",
        "candidate-sources.tar.gz",
        "dispatch-sources.tar.gz",
        "allocation-sources.tar.gz",
        "large-structure-sources.tar.gz",
    ):
        for key, value in archive_hashes(directory / name).items():
            require(env["source_hashes"].get(key) == value, "Archive differs from measured source")
            archives[key] = value
    require(archives == env["source_hashes"], "Incomplete archive")
    require(env["threads"] == 1 and all(p["num_threads"] == 1 for p in env["threadpools"]), "Thread mismatch")
    transforms = check_transforms(directory, directory.parent / "corpus")
    initialization = check_initialization(records)
    exports = sum(audit_worker(r, s) for r in records for s in ("X", "Z"))
    manifest = json.loads((directory / "manifest.json").read_text())
    for case in manifest["cases"]:
        with np.load(directory / "matrices" / case["file"], allow_pickle=False) as data:
            n = data["hx"].shape[1]
        for r in records:
            if r["case"] != case["id"] or r["method"] != "structure":
                continue
            for ws in r["workers"].values():
                require(ws[0]["counters"]["spec"] == case["structure_spec"], "Construction spec mismatch")
                for space in ws[0]["counters"]["spaces"]:
                    require(all(0 <= q < n for g in space["groups"] for q in g), "Invalid coordinate")
    standard_audit(directory)
    standard = json.loads((directory / "audit.json").read_text())
    require(
        archive_hashes(directory / "supplemental-sources.tar.gz") == standard["supplemental_source_hashes"],
        "Supplemental archive mismatch",
    )
    cells = []
    lines = [
        "# Large-code target recovery",
        "",
        f"{len(records)} configurations, {env['seconds_per_code']:g} seconds/code, {env['seeds']} seeds, one CPU.",
        "",
        "Targets are repository witnessed upper bounds, not exact-distance certificates. "
        "First-hit times below are per-sector elapsed time with half the code budget. None denotes a censored miss.",
        "",
        "| Code | Method | Final weights | Target hits | First-hit seconds/sector | Structured-only weights |",
        "|---|---|---|---:|---|---|",
    ]
    for case in env["cases"]:
        for method in env["methods"]:
            rs = sorted((r for r in records if r["case"] == case and r["method"] == method), key=lambda r: r["seed"])
            hits = []
            candidates = []
            for r in rs:
                es = [
                    e
                    for ws in r["workers"].values()
                    for e in ws[0]["events"]
                    if e["seconds"] <= r["budget_seconds"] / 2
                ]
                hits.append(min((e["seconds"] for e in es if e["weight"] <= r["target"]), default=None))
                candidates.append(
                    min(
                        (e["weight"] for e in es if e["stage"].startswith(("seed_blocks_", "strip_", "orbits_"))),
                        default=None,
                    )
                )
            cell = dict(
                case=case,
                method=method,
                target=rs[0]["target"],
                weights=[best(r) for r in rs],
                hits=sum(t is not None for t in hits),
                first_hit_seconds=hits,
                structured_only=candidates,
            )
            cell["median"] = statistics.median(cell["weights"])
            cells.append(cell)
            lines.append(
                f"| {case} | {method} | {cell['weights']} | {cell['hits']}/{len(rs)} | "
                f"{[round(t, 4) if t is not None else None for t in hits]} | {candidates} |"
            )
    lines += [
        "",
        f"All {standard['saved_witness_documents_checked']:,} saved witness documents passed the audit. "
        "Source/binary/archive, common initialization, actual matrices, construction specs, "
        "restricted/orbit supports and exported counters checked. No exact-distance or full candidate-gate claim.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(directory / "summary.json", dict(cells=cells))
    atomic_json(
        directory / "audit-large-structure.json",
        dict(
            status="passed",
            configurations=len(records),
            saved_witness_documents_checked=standard["saved_witness_documents_checked"],
            structured_exports_checked=exports,
            identical_complete_initialization=initialization,
            transformations=transforms,
            source_binary_archive_hashes_match=True,
            space_and_export_checks_match=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
