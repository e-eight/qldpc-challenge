"""Audit complete dispatch grids and report quality, attribution, and overhead."""

import argparse
import json
import statistics
from pathlib import Path

from run import np

# isort: split
from audit_candidates import archive_hashes, check_exports, check_initialization, check_transforms, require
from audit_strategies import main as standard_audit
from common import atomic_json
from dispatch_search import CONFIG
from report_candidates import best
from report_strategies import checked_records


def audit_worker(record, side):
    if record["method"] != "dispatch":
        return check_exports(record, side)
    worker = record["workers"][side][0]
    counters, events = worker["counters"], worker["events"]
    require(counters["parameters"] == CONFIG, "Dispatch policy mismatch")
    require(
        all(e["stage"] in {"initialization", "initial_seed", "routed_single_block", "guided"} for e in events),
        "Unexpected dispatch stage",
    )
    labels = [e["stage"] for e in events]
    stage_order = {"initialization": 0, "initial_seed": 0, "routed_single_block": 1, "guided": 2}
    require([stage_order[s] for s in labels] == sorted(stage_order[s] for s in labels), "Stage order mismatch")
    times = [e["seconds"] for e in events]
    require(times == sorted(times) and all(0 <= t <= worker["search_seconds"] for t in times), "Event time mismatch")
    route_events = [e for e in events if e["stage"] == "routed_single_block"]
    route = counters.get("route", {})
    require(route.get("exported", 0) == len(route_events), "Route export count mismatch")
    require(route.get("best") == min((e["weight"] for e in route_events), default=None), "Route best mismatch")
    require(route.get("batches", 0) >= len(route_events), "More exports than batches")
    if "order" in counters:
        order = counters["order"]
        require(sorted(order) == list(range(len(order))), "Nonbijective route order")
        require(len(order) % 2 == 0, "Odd route partition")
        require(
            any(
                d["status"] == "accepted" and d["confidence"] >= CONFIG["minimum_confidence"]
                for d in counters["detection"]
            ),
            "Route without accepted detector",
        )
        left, right = set(order[: len(order) // 2]), set(order[len(order) // 2 :])
        for event in route_events:
            require(set(event["support"]) <= left or set(event["support"]) <= right, "Route left its block space")
    else:
        require(not route_events and not route, "Route without partition")
    if counters["route_decision"] == "continue_improving_route":
        require(
            counters["pilot_best"] is not None and counters["pilot_best"] < counters["initial_best"],
            "Unhelpful route was continued",
        )
    elif counters["route_decision"] == "pilot_did_not_improve":
        require(
            counters["pilot_best"] is None or counters["pilot_best"] >= counters["initial_best"],
            "Helpful pilot was mislabeled",
        )
    if "guided" in labels:
        require("fallback" in counters and counters["fallback_seconds"] > 0, "Missing fallback accounting")
    expected = min(CONFIG["detector_seconds_cap"], CONFIG["detector_budget_fraction"] * record["budget_seconds"] / 2)
    require(0 <= counters["detection_allowance_seconds"] <= expected + 1e-8, "Detector allowance exceeded policy")
    return len(route_events)


def main(directory):
    directory = directory.resolve()
    require((directory / "completed.json").exists(), "Refusing incomplete grid")
    env, records = checked_records(directory)
    completed = json.loads((directory / "completed.json").read_text())
    require(completed["configurations"] == len(records), "Completion count mismatch")
    require(completed["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Save total mismatch")
    archived = {}
    for name in ("sources.tar.gz", "candidate-sources.tar.gz", "dispatch-sources.tar.gz"):
        for key, value in archive_hashes(directory / name).items():
            require(env["source_hashes"].get(key) == value, "Measured archive differs from source hash")
            archived[key] = value
    require(archived == env["source_hashes"], "Source archive incomplete")
    require(env["threads"] == 1 and all(p["num_threads"] == 1 for p in env["threadpools"]), "Thread limit mismatch")
    transforms = check_transforms(directory, directory.parent / "corpus")
    initial = check_initialization(records)
    exports = sum(audit_worker(r, side) for r in records for side in ("X", "Z"))
    standard_audit(directory)
    standard = json.loads((directory / "audit.json").read_text())
    require(
        archive_hashes(directory / "supplemental-sources.tar.gz") == standard["supplemental_source_hashes"],
        "Supplemental archive mismatch",
    )
    for r in records:
        if r["method"] == "dispatch":
            case = next(
                c for c in json.loads((directory / "manifest.json").read_text())["cases"] if c["id"] == r["case"]
            )
            with np.load(directory / "matrices" / case["file"], allow_pickle=False) as data:
                n = data["hx"].shape[1]
            for ws in r["workers"].values():
                if "order" in ws[0]["counters"]:
                    require(len(ws[0]["counters"]["order"]) == n, "Route dimension mismatch")
    cells, timing = [], []
    lines = [
        "# Bounded dispatcher comparison",
        "",
        f"{len(records)} configurations; {env['seconds_per_code']:g} seconds/code; {env['seeds']} seeds; one CPU.",
        "",
        "All methods share identical timed preparation/initialization. Lower witnessed upper bounds are better. "
        "Search receives only matrices, never code names or reference witnesses/targets. "
        "Late outputs are saved without deadline credit.",
        "",
        "| Input | Target | " + " | ".join(env["methods"]) + " |",
        "|---|---:|" + "---:|" * len(env["methods"]),
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
                target=rs[0]["target"],
            )
            cells.append(cell)
            row.append(f"{cell['median']:g}")
        lines.append(f"| {case} | {rs[0]['target']} | " + " | ".join(row) + " |")
    paired = []
    lines += ["", "## Paired dispatcher outcomes", "", "| Control | Lower | Equal | Higher |", "|---|---:|---:|---:|"]
    for method in env["methods"]:
        if method == "dispatch":
            continue
        counts = dict(lower=0, equal=0, higher=0)
        for r in records:
            if r["method"] != "dispatch":
                continue
            other = next(
                v for v in records if v["case"] == r["case"] and v["seed"] == r["seed"] and v["method"] == method
            )
            a, b = best(r), best(other)
            counts["lower" if a < b else "equal" if a == b else "higher"] += 1
        paired.append(dict(control=method, **counts))
        lines.append(f"| {method} | {counts['lower']} | {counts['equal']} | {counts['higher']} |")
    lines += [
        "",
        "## Dispatcher decisions and costs",
        "",
        "Times below are per-sector medians; max detection includes Python conversion and both proposals. "
        "Detection and native search use cooperative deadlines, not hard real-time cancellation.",
        "",
        "| Input | Decisions (sectors) | Median/max detection ms | Max detector overrun ms | "
        "Median route s | Median guided s | Routed-only weights |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for case in env["cases"]:
        rs = [r for r in records if r["case"] == case and r["method"] == "dispatch"]
        ws = [w for r in rs for workers in r["workers"].values() for w in workers]
        cs = [w["counters"] for w in ws]
        decisions = {
            label: sum(c["route_decision"] == label for c in cs) for label in sorted({c["route_decision"] for c in cs})
        }
        row = dict(
            case=case,
            decisions=decisions,
            median_detection_seconds=statistics.median(c["detection_seconds"] for c in cs),
            max_detection_seconds=max(c["detection_seconds"] for c in cs),
            max_detection_overrun_seconds=max(
                max(0, c["detection_seconds"] - c["detection_allowance_seconds"]) for c in cs
            ),
            max_detector_workspace_bytes=max((d["workspace_bytes"] for c in cs for d in c["detection"]), default=0),
            median_route_seconds=statistics.median(c.get("route_seconds", 0) for c in cs),
            median_guided_seconds=statistics.median(c.get("fallback_seconds", 0) for c in cs),
            max_route_overrun_seconds=max(c.get("route_overrun_seconds", 0) for c in cs),
            cpu_seconds=sum(w["cpu_seconds"] for w in ws),
            elapsed_seconds=sum(w["search_seconds"] for w in ws),
            routed_only=[best(r, True) for r in sorted(rs, key=lambda r: r["seed"])],
        )
        timing.append(row)
        lines.append(
            f"| {case} | {decisions} | {1000 * row['median_detection_seconds']:.3f} / "
            f"{1000 * row['max_detection_seconds']:.3f} | {1000 * row['max_detection_overrun_seconds']:.3f} | "
            f"{row['median_route_seconds']:.3f} | {row['median_guided_seconds']:.3f} | {row['routed_only']} |"
        )
    lines += [
        "",
        f"All {standard['saved_witness_documents_checked']:,} saved documents passed the persistence audit. "
        "Common initialization, complete grid, source/binary/archive hashes, actual transformations, route "
        "support partitions and decision/export counters were checked. Trusted witness algebra was checked by "
        "the runner. No exact-distance claim or full submission gate.",
        "",
        "This detector still uses supplied check-row ordering. The new qubit/row permutations are mechanism "
        "controls, not held-out code families. A helpful pilot only means improvement over initialization; "
        "it does not guarantee the restricted search will beat guided at the final deadline.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(directory / "summary.json", dict(cells=cells, paired=paired, timing=timing))
    atomic_json(
        directory / "audit-dispatch.json",
        dict(
            status="passed",
            configurations=len(records),
            saved_witness_documents_checked=standard["saved_witness_documents_checked"],
            routed_exports_checked=exports,
            identical_complete_initialization=initial,
            transformations=transforms,
            source_binary_archive_hashes_match=True,
            route_decisions_and_exports_match=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
