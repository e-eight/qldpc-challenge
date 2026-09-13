"""Audit and report the completed exploratory inferred-block routing grid."""

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_candidates import archive_hashes, check_initialization, check_transforms, require  # noqa: E402
from audit_strategies import main as audit_standard  # noqa: E402
from common import ROOT, atomic_json, sha256  # noqa: E402
from report_strategies import checked_records  # noqa: E402

INITIAL = {"initialization", "initial_seed"}


def best(record, stages=None, timely=True):
    return min(
        (
            event["weight"]
            for workers in record["workers"].values()
            for worker in workers
            for event in worker["events"]
            if (not timely or event["seconds"] <= record["budget_seconds"] / 2)
            and (stages is None or event["stage"] in stages)
        ),
        default=None,
    )


def show(value):
    return "none" if value is None else str(value)


def measured_archives(directory, expected):
    members = {}
    for filename in ("sources.tar.gz", "candidate-sources.tar.gz", "route-sources.tar.gz"):
        for name, digest in archive_hashes(directory / filename).items():
            require(name in expected and expected[name] == digest, f"Measured archive bytes mismatch: {name}")
            require(name not in members or members[name] == digest, f"Conflicting source archive: {name}")
            members[name] = digest
    require(set(members) == set(expected), "Measured source missing from archives")
    return len(members)


def check_route_exports(record):
    exported, late = 0, 0
    for side in ("X", "Z"):
        require(len(record["workers"][side]) == 1, "Expected one worker per sector")
        worker = record["workers"][side][0]
        labels = {event["stage"] for event in worker["events"]}
        candidate_stage = "routed_single_block" if record["method"] == "routed" else "circulant"
        require(labels <= INITIAL | {"guided", candidate_stage}, "Unexpected route diagnostic stage")
        if "guided" in labels:
            require("fallback" in worker["counters"], "Guided output lacks fallback accounting")
        if record["method"] != "routed":
            continue
        candidate = worker["counters"]["candidate"]
        events = [event for event in worker["events"] if event["stage"] == candidate_stage]
        require(candidate["exported"] == len(events), "Routed export counter mismatch")
        require(candidate["best"] == min((e["weight"] for e in events), default=None), "Routed best mismatch")
        routes = candidate["routes"]
        for key in ("exported", "batches", "scored_bases"):
            require(sum(route[key] for route in routes) == candidate[key], f"Routed {key} total mismatch")
        require(
            min((route["best"] for route in routes if route["best"] is not None), default=None) == candidate["best"],
            "Routed per-route best mismatch",
        )
        if events:
            require(candidate["applicable"], "Routed output without applicable native session")
            require(any(route["native_applicable"] for route in routes), "Routed output without native kernel space")
        exported += len(events)
        late += sum(event["seconds"] > record["budget_seconds"] / 2 for event in events)
    return exported, late


def main(directory, corpus=None):
    directory = Path(directory)
    corpus = Path(corpus) if corpus else directory.parent / "corpus"
    require((directory / "completed.json").is_file(), "Refusing to audit an active grid")
    env, records = checked_records(directory)
    require(env["methods"] == ["circulant", "routed"] and env["seeds"] == 1, "Unexpected diagnostic grid")
    require(env["seconds_per_code"] == 2 and env["seed_start"] == 1240, "Unexpected exploratory budget or seed")
    require(env["threads"] == 1 and all(p["num_threads"] == 1 for p in env["threadpools"]), "Thread limit mismatch")
    require(
        env["override"]
        == {
            "recorded_method": "routed",
            "wrapper_method": "matrix-structure",
            "adapter_mapping": {"matrix-structure": "structure_route_v2"},
        },
        "Unexpected search-wrapper override",
    )
    completed = json.loads((directory / "completed.json").read_text())
    require(completed["configurations"] == len(records), "Completed grid count mismatch")
    require(
        completed["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Completed save count mismatch"
    )
    archives = measured_archives(directory, env["source_hashes"])
    transforms = check_transforms(directory, corpus)
    initialization = check_initialization(records)
    counts = [check_route_exports(record) for record in records]
    audit_standard(directory)
    standard = json.loads((directory / "audit.json").read_text())
    require(
        archive_hashes(directory / "supplemental-sources.tar.gz") == standard["supplemental_source_hashes"],
        "Supplemental archive hash mismatch",
    )
    audit_files = [Path(__file__).resolve(), Path(__file__).resolve().parents[2] / "audit_candidates.py"]
    with tarfile.open(directory / "route-audit-sources.tar.gz", "w:gz") as archive:
        for path in audit_files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))

    cells = []
    lines = [
        "# Exploratory routing of inferred coordinate blocks",
        "",
        "Separate follow-up: nine inputs, two methods, one seed (1240), 2 seconds/code split evenly "
        "between X and Z. One search thread, serial timed runs. This is a mechanism diagnostic, "
        "not a reliable multi-seed ranking or an added method in the frozen main comparison.",
        "",
        "Both methods include identical common initialization and use a fresh guided fallback if "
        "their specialized pass is inapplicable or returns early. Recovery, reordering, native "
        "preparation, search, mapping and callback delivery count against the budget. The routed "
        "method bypasses only the coordinate-layout recognizer; actual restricted kernels and "
        "logical tests remain in the existing native implementation.",
        "",
        "Lower weights are better. None means no timely output from that stage, not a distance bound.",
        "",
        "| Input | Common init | Circulant portfolio | Routed portfolio | Circulant only | Routed only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in env["cases"]:
        original = next(r for r in records if r["case"] == case and r["method"] == "circulant")
        routed = next(r for r in records if r["case"] == case and r["method"] == "routed")
        cell = dict(
            case=case,
            initial=best(original, INITIAL),
            circulant_portfolio=best(original),
            routed_portfolio=best(routed),
            circulant_only=best(original, {"circulant"}),
            routed_only=best(routed, {"routed_single_block"}),
            circulant_fallback=best(original, {"guided"}),
            routed_fallback=best(routed, {"guided"}),
        )
        cell["routed_improves_initial"] = cell["routed_only"] is not None and cell["routed_only"] < cell["initial"]
        cells.append(cell)
        lines.append(
            "| "
            + " | ".join(
                show(cell[key])
                for key in (
                    "case",
                    "initial",
                    "circulant_portfolio",
                    "routed_portfolio",
                    "circulant_only",
                    "routed_only",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Fallback attribution",
        "",
        "| Input | Circulant guided fallback | Routed guided fallback | Routed beats common init |",
        "|---|---:|---:|---|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['case']} | {show(cell['circulant_fallback'])} | {show(cell['routed_fallback'])} | "
            f"{cell['routed_improves_initial']} |"
        )
    lines += [
        "",
        "## Per-code resource accounting",
        "",
        "Times sum both sector workers. Native batches target 50 ms; late exports are retained "
        "but receive no deadline credit.",
        "",
        "| Input | Method | CPU s | Elapsed s | Candidate setup s | Guided fallback s | Late events |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    timing = []
    for record in records:
        workers = [worker for side in record["workers"].values() for worker in side]
        row = dict(
            case=record["case"],
            method=record["method"],
            cpu_seconds=sum(w["cpu_seconds"] for w in workers),
            elapsed_seconds=sum(w["search_seconds"] for w in workers),
            candidate_setup_seconds=sum(w["counters"].get("candidate", {}).get("setup_seconds", 0) for w in workers),
            fallback_seconds=sum(w["counters"].get("fallback_seconds", 0) for w in workers),
            late_events=sum(side["late_events"] for side in record["sides"].values()),
        )
        timing.append(row)
        lines.append(
            f"| {row['case']} | {row['method']} | {row['cpu_seconds']:.4f} | {row['elapsed_seconds']:.4f} | "
            f"{row['candidate_setup_seconds']:.4f} | {row['fallback_seconds']:.4f} | {row['late_events']} |"
        )
    limits = [
        "One seed and a 2-second budget support a bounded mechanism diagnostic only.",
        "Recovery uses check-row ordering; two equal permutation cycles are proposals, not automorphism proofs.",
        "Parent event timestamps determine deadline credit, including the final native batch return.",
        "All returned supports were mapped to original coordinates and trusted-validated by the runner.",
        "Audit checks preservation and counter identity; it does not reconstruct hidden native batch intermediates.",
        "No full candidate gate or independently replayed exact-distance certificate is claimed.",
    ]
    lines += ["", "## Evidence and limits", ""] + [f"- {limit}" for limit in limits]
    lines += [
        "",
        f"Standard persistence audit checked {standard['saved_witness_documents_checked']:,} saved witness documents. "
        "All measured source/binary hashes and all three measured source archive contents match. "
        "Original and transformed matrices, reference remappings, and complete common initialization were checked.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(directory / "summary.json", dict(cells=cells, timing=timing, limits=limits))
    atomic_json(
        directory / "audit-route.json",
        dict(
            status="passed",
            configurations=len(records),
            saved_witness_documents_checked=standard["saved_witness_documents_checked"],
            routed_export_events_checked=sum(count for count, _ in counts),
            routed_late_exports=sum(count for _, count in counts),
            measured_archived_sources_checked=archives,
            transformations=transforms,
            identical_complete_initialization=initialization,
            source_and_binary_hashes_match=True,
            route_export_counters_match=True,
            audit_source_hashes={str(path.relative_to(ROOT)): sha256(path) for path in audit_files},
            report_sha256=hashlib.sha256((directory / "REPORT.md").read_bytes()).hexdigest(),
            limits=limits,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--cpu", type=int, default=7)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    main(args.directory, args.corpus)
