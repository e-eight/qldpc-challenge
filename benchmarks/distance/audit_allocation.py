"""Audit allocation decisions, resumable session exports, and persisted grids."""

import argparse
import json
import statistics
from pathlib import Path

from run import np

# isort: split
from allocation_search import CONFIG, preferred
from audit_candidates import INITIAL, archive_hashes, check_initialization, check_transforms, require
from audit_dispatch import audit_worker as control_worker
from audit_strategies import main as standard_audit
from common import atomic_json
from report_candidates import best
from report_strategies import checked_records


def audit_worker(record, side):
    if record["method"] != "race":
        return control_worker(record, side)
    worker = record["workers"][side][0]
    c, events = worker["counters"], worker["events"]
    require(c["parameters"] == CONFIG, "Allocation policy mismatch")
    times = [e["seconds"] for e in events]
    require(times == sorted(times) and all(0 <= t <= worker["search_seconds"] for t in times), "Event time mismatch")
    require(all(e["stage"] in INITIAL | {"guided", "routed_single_block"} for e in events), "Unexpected stage")
    search_started = False
    for e in events:
        if e["stage"] not in INITIAL:
            search_started = True
        else:
            require(not search_started, "Initialization after search")
    expected = min(CONFIG["detector_seconds_cap"], CONFIG["detector_budget_fraction"] * record["budget_seconds"] / 2)
    require(0 <= c["detection_allowance_seconds"] <= expected + 1e-8, "Detector allowance mismatch")
    routed = [e for e in events if e["stage"] == "routed_single_block"]
    if "order" not in c:
        require(c["route_decision"] == "no_partition" and not routed and "sessions" not in c, "Invalid fallback")
        require("fallback" in c, "Missing fallback")
        return 0
    order = c["order"]
    require(sorted(order) == list(range(len(order))) and len(order) % 2 == 0, "Nonbijective route order")
    require(
        any(d["status"] == "accepted" and d["confidence"] >= CONFIG["minimum_confidence"] for d in c["detection"]),
        "No accepted detector",
    )
    left, right = set(order[: len(order) // 2]), set(order[len(order) // 2 :])
    require(all(set(e["support"]) <= left or set(e["support"]) <= right for e in routed), "Route left block space")
    require(c["route_decision"] == "competing_sessions", "Incorrect competition label")
    slices = c["slices"]
    require(
        [s["engine"] for s in slices if s["phase"] == "pilot"] == ["guided", "route"], "Missing or reordered pilots"
    )
    for a, b in zip(slices, slices[1:]):
        require(a["end_seconds"] <= b["start_seconds"], "Overlapping slices")
    for s in slices:
        require(0 <= s["start_seconds"] <= s["end_seconds"] <= worker["search_seconds"], "Slice time mismatch")
        require(s["deadline_seconds"] <= record["budget_seconds"] / 2 + 1e-5, "Slice exceeds total budget")
    for name, label in [("route", "routed_single_block"), ("guided", "guided")]:
        es = [e for e in events if e["stage"] == label]
        session = c["sessions"].get(name, {})
        require(session.get("exported", 0) == len(es), "Session export count mismatch")
        require(session.get("best") == min((e["weight"] for e in es), default=None), "Session best mismatch")
        ss = [s for s in slices if s["engine"] == name]
        require([s["batches"] for s in ss] == sorted(s["batches"] for s in ss), "Session restarted")
        require(not ss or ss[-1]["batches"] == session["batches"], "Batch accounting mismatch")
        for e in es:
            require(
                any(s["start_seconds"] <= e["seconds"] <= s["end_seconds"] + 1e-5 for s in ss),
                "Export outside engine slice",
            )
    for d in c["decisions"]:
        require(
            d["preferred"] == preferred(d["route_best"], d["guided_best"], c["initial_best"]), "Wrong preferred engine"
        )
        require(0 < d["epoch_seconds"] <= CONFIG["allocation_epoch_seconds"], "Invalid epoch")
        for name, label in [("route", "routed_single_block"), ("guided", "guided")]:
            weights = [e["weight"] for e in events if e["stage"] == label and e["seconds"] <= d["seconds"]]
            require(d[name + "_best"] == min(weights, default=None), "Decision used wrong evidence")
        allocated = [
            s
            for s in slices
            if s["start_seconds"] >= d["seconds"] and s["start_seconds"] < d["seconds"] + d["epoch_seconds"]
        ]
        require(
            allocated and allocated[0]["phase"] == "explore" and allocated[0]["engine"] != d["preferred"],
            "Missing exploration",
        )
        expected_end = d["seconds"] + CONFIG["exploration_fraction"] * d["epoch_seconds"]
        require(abs(allocated[0]["deadline_seconds"] - expected_end) < 1e-7, "Exploration allocation mismatch")
        for s in allocated[1:]:
            require(s["phase"] == "exploit" and s["engine"] == d["preferred"], "Incorrect exploitation")
            require(
                abs(s["deadline_seconds"] - d["seconds"] - d["epoch_seconds"]) < 1e-7,
                "Exploitation allocation mismatch",
            )
    return len(routed)


def main(directory):
    directory = directory.resolve()
    require((directory / "completed.json").exists(), "Incomplete grid")
    env, records = checked_records(directory)
    completed = json.loads((directory / "completed.json").read_text())
    require(completed["configurations"] == len(records), "Completion mismatch")
    require(completed["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Save mismatch")
    archived = {}
    for name in ("sources.tar.gz", "candidate-sources.tar.gz", "dispatch-sources.tar.gz", "allocation-sources.tar.gz"):
        for key, value in archive_hashes(directory / name).items():
            require(env["source_hashes"].get(key) == value, "Archive mismatch")
            archived[key] = value
    require(archived == env["source_hashes"], "Incomplete source archives")
    require(env["threads"] == 1 and all(p["num_threads"] == 1 for p in env["threadpools"]), "Thread mismatch")
    transforms = check_transforms(directory, directory.parent / "corpus")
    initial = check_initialization(records)
    exports = sum(audit_worker(r, side) for r in records for side in ("X", "Z"))
    manifest = json.loads((directory / "manifest.json").read_text())
    sizes = {}
    for case in manifest["cases"]:
        with np.load(directory / "matrices" / case["file"], allow_pickle=False) as data:
            sizes[case["id"]] = data["hx"].shape[1]
    for r in records:
        for ws in r["workers"].values():
            if "order" in ws[0]["counters"]:
                require(len(ws[0]["counters"]["order"]) == sizes[r["case"]], "Partition dimension mismatch")
    standard_audit(directory)
    standard = json.loads((directory / "audit.json").read_text())
    require(
        archive_hashes(directory / "supplemental-sources.tar.gz") == standard["supplemental_source_hashes"],
        "Supplemental archive mismatch",
    )
    cells, paired, allocation = [], [], []
    lines = [
        "# Competing search allocation",
        "",
        f"{len(records)} configurations; {env['seconds_per_code']:g} seconds/code; {env['seeds']} seeds; one CPU.",
        "",
        "Lower witnessed upper bounds are better. "
        "Preparation, initialization, detection, setup and exports are timed. "
        "Late outputs are saved without deadline credit.",
        "",
        "| Input | " + " | ".join(env["methods"]) + " |",
        "|---|" + "---:|" * len(env["methods"]),
    ]
    for case in env["cases"]:
        row = []
        for method in env["methods"]:
            rs = sorted((r for r in records if r["case"] == case and r["method"] == method), key=lambda r: r["seed"])
            weights = [best(r) for r in rs]
            cell = dict(
                case=case,
                method=method,
                weights=weights,
                median=statistics.median(weights),
                routed_only=[best(r, True) for r in rs],
            )
            cells.append(cell)
            row.append(str(cell["median"]))
        lines.append(f"| {case} | " + " | ".join(row) + " |")
    lines += ["", "| Control | Race lower | Equal | Higher |", "|---|---:|---:|---:|"]
    for method in env["methods"]:
        if method == "race":
            continue
        counts = dict(lower=0, equal=0, higher=0)
        for r in records:
            if r["method"] != "race":
                continue
            other = next(v for v in records if (v["case"], v["seed"], v["method"]) == (r["case"], r["seed"], method))
            a, b = best(r), best(other)
            counts["lower" if a < b else "equal" if a == b else "higher"] += 1
        paired.append(dict(control=method, **counts))
        lines.append(f"| {method} | {counts['lower']} | {counts['equal']} | {counts['higher']} |")
    for r in records:
        if r["method"] != "race":
            continue
        for side, ws in r["workers"].items():
            c = ws[0]["counters"]
            ds = c.get("decisions", [])
            allocation.append(
                dict(
                    case=r["case"],
                    seed=r["seed"],
                    side=side,
                    decision=c["route_decision"],
                    preferred_counts={name: sum(d["preferred"] == name for d in ds) for name in ("guided", "route")},
                    switches=sum(a["preferred"] != b["preferred"] for a, b in zip(ds, ds[1:])),
                    active_seconds={
                        name: sum(
                            s["end_seconds"] - s["start_seconds"] for s in c.get("slices", []) if s["engine"] == name
                        )
                        for name in ("guided", "route")
                    },
                )
            )
    lines += [
        "",
        f"All {standard['saved_witness_documents_checked']:,} saved witness documents audited. "
        "Initialization, source/binary/archive hashes, transforms, exports and allocation decisions checked. "
        "No exact-distance or full candidate-gate claim.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(directory / "summary.json", dict(cells=cells, paired=paired, allocation=allocation))
    atomic_json(
        directory / "audit-allocation.json",
        dict(
            status="passed",
            configurations=len(records),
            saved_witness_documents_checked=standard["saved_witness_documents_checked"],
            routed_exports_checked=exports,
            identical_complete_initialization=initial,
            transformations=transforms,
            source_binary_archive_hashes_match=True,
            allocation_and_exports_match=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
