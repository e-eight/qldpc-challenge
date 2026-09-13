"""Aggregate completed dispatcher grids without rerunning search."""

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    grids = {}
    for name, count, budget, seeds in (("screen-2s", 156, 2, 3), ("original-10s", 56, 10, 2)):
        directory = BASE / name
        env = json.loads((directory / "environment.json").read_text())
        audit = json.loads((directory / "audit-dispatch.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        records = json.loads((directory / "results.json").read_text())
        assert audit["status"] == "passed" and audit["configurations"] == count == len(records)
        assert env["seconds_per_code"] == budget and env["seeds"] == seeds
        grids[name] = dict(environment=env, audit=audit, summary=summary, records=records)
    first, second = grids.values()
    assert first["environment"]["source_hashes"] == second["environment"]["source_hashes"]
    assert first["environment"]["binary_hashes"] == second["environment"]["binary_hashes"]
    signatures = {
        (c["case"], c["side"]): c["signature_sha256"] for c in first["audit"]["identical_complete_initialization"]
    }
    for c in second["audit"]["identical_complete_initialization"]:
        assert signatures[c["case"], c["side"]] == c["signature_sha256"]
    records = [r for g in grids.values() for r in g["records"]]
    workers = [w for r in records for ws in r["workers"].values() for w in ws]
    inputs = [
        BASE / name / file
        for name in grids
        for file in ("environment.json", "audit-dispatch.json", "summary.json", "results.json")
    ]
    inputs.append(Path(__file__))
    result = dict(
        configurations=len(records),
        saved_documents=sum(r["saved_candidates"] for r in records),
        identical_measured_sources_and_binaries=True,
        identical_initialization_across_budgets=True,
        timed_cpu_seconds=sum(w["cpu_seconds"] for w in workers),
        timed_elapsed_seconds=sum(w["search_seconds"] for w in workers),
        late_exports=sum(s["late_events"] for r in records for s in r["sides"].values()),
        input_hashes={str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        grids={name: g["summary"] for name, g in grids.items()},
    )
    (BASE / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    paired, unsupported, dispatch_counters = {}, [], []
    for name, grid in grids.items():
        for row in grid["summary"]["paired"]:
            counts = paired.setdefault(row["control"], dict(lower=0, equal=0, higher=0))
            for key in counts:
                counts[key] += row[key]
        for record in grid["records"]:
            if record["method"] != "dispatch":
                continue
            cs = [w["counters"] for ws in record["workers"].values() for w in ws]
            dispatch_counters.extend(cs)
            if all(c["route_decision"] == "no_partition" for c in cs):
                guide = next(
                    r
                    for r in grid["records"]
                    if r["case"] == record["case"] and r["seed"] == record["seed"] and r["method"] == "guided"
                )
                unsupported.append(
                    dict(
                        grid=name,
                        case=record["case"],
                        seed=record["seed"],
                        dispatch=min(v["best_in_budget"] for v in record["sides"].values()),
                        guided=min(v["best_in_budget"] for v in guide["sides"].values()),
                    )
                )
    decisions = dict(
        paired=paired,
        unsupported=unsupported,
        max_detection_seconds=max(c["detection_seconds"] for c in dispatch_counters),
        max_detection_overrun_seconds=max(
            max(0, c["detection_seconds"] - c["detection_allowance_seconds"]) for c in dispatch_counters
        ),
        max_detector_workspace_bytes=max(d["workspace_bytes"] for c in dispatch_counters for d in c["detection"]),
        input_hashes=result["input_hashes"],
    )
    (BASE / "decision-summary.json").write_text(json.dumps(decisions, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "configurations",
                    "saved_documents",
                    "timed_cpu_seconds",
                    "timed_elapsed_seconds",
                    "late_exports",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
