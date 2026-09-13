"""Report complete candidate grids without attributing fallback gains to proposals."""

import argparse
import statistics
from pathlib import Path

from common import atomic_json
from report_strategies import checked_records

INITIAL = {"initialization", "initial_seed"}
CONTROLS = {"incremental", "guided", "circulant"}


def events(record, candidate_only=False):
    return [
        e
        for ws in record["workers"].values()
        for w in ws
        for e in w["events"]
        if e["seconds"] <= record["budget_seconds"] / 2 and (not candidate_only or e["stage"] not in INITIAL | CONTROLS)
    ]


def best(record, candidate_only=False):
    return min((e["weight"] for e in events(record, candidate_only)), default=None)


def main(directory):
    env, records = checked_records(directory)
    cells, paired = [], []
    lines = [
        "# Candidate-generator comparison",
        "",
        f"{len(records)} configurations; {env['seeds']} seeds; {env['seconds_per_code']:g} seconds/code; "
        "one worker, equal X/Z budgets, serial runs.",
        "",
        "Preparation, common logical initialization, candidate setup/search, guided fallback and witness "
        "delivery are all charged. Finite or inapplicable candidate passes give their remaining budget "
        "to a fresh guided session. Reference witnesses/targets never enter search. No target stopping.",
        "",
        "## Median delivered upper bound",
        "",
        "| Code | Target | " + " | ".join(env["methods"]) + " |",
        "|---|---:|" + "---:|" * len(env["methods"]),
    ]
    for case in env["cases"]:
        row = []
        for method in env["methods"]:
            rs = sorted([r for r in records if r["case"] == case and r["method"] == method], key=lambda r: r["seed"])
            values = [best(r) for r in rs]
            if any(v is None for v in values):
                raise ValueError("Missing timely common initialization")
            candidate = [best(r, True) for r in rs]
            initial = [min(w["counters"]["initial_best"] for ws in r["workers"].values() for w in ws) for r in rs]
            cell = dict(
                case=case,
                method=method,
                weights=values,
                median=statistics.median(values),
                initial=initial,
                candidate_weights=candidate,
                candidate_improved_initial=sum(v is not None and v < b for v, b in zip(candidate, initial)),
                target_hits=sum(v <= r["target"] for v, r in zip(values, rs)),
            )
            cells.append(cell)
            row.append(str(cell["median"]))
        lines.append(f"| {case} | {rs[0]['target']} | " + " | ".join(row) + " |")
    lines += ["", "## Paired against guided", "", "| Method | Lower | Equal | Higher |", "|---|---:|---:|---:|"]
    for method in env["methods"]:
        if method == "guided":
            continue
        counts = dict(lower=0, equal=0, higher=0)
        for r in records:
            if r["method"] != method:
                continue
            other = next(
                v for v in records if v["case"] == r["case"] and v["seed"] == r["seed"] and v["method"] == "guided"
            )
            a, b = best(r), best(other)
            counts["lower" if a < b else "equal" if a == b else "higher"] += 1
        paired.append(dict(method=method, **counts))
        lines.append(f"| {method} | {counts['lower']} | {counts['equal']} | {counts['higher']} |")
    lines += [
        "",
        "## Candidate-only outputs across seeds",
        "",
        "These values exclude common initialization and guided fallback. None means no timely candidate "
        "export, not a distance bound. Transfers and different reduced-space branches are separated below.",
        "",
        "| Code | Method | Candidate weights | Improved common code-level initialization / seeds |",
        "|---|---|---|---:|",
    ]
    for c in cells:
        if c["method"] in CONTROLS:
            continue
        lines.append(
            f"| {c['case']} | {c['method']} | {c['candidate_weights']} | "
            f"{c['candidate_improved_initial']}/{env['seeds']} |"
        )
    stages = []
    lines += [
        "",
        "## Per-stage best delivered weight",
        "",
        "| Code | Method | Stage | Best | Runs with output |",
        "|---|---|---|---:|---:|",
    ]
    for case in env["cases"]:
        for method in env["methods"]:
            rs = [r for r in records if r["case"] == case and r["method"] == method]
            labels = sorted({e["stage"] for r in rs for e in events(r)})
            for label in labels:
                weights = [min((e["weight"] for e in events(r) if e["stage"] == label), default=None) for r in rs]
                valid = [v for v in weights if v is not None]
                stages.append(dict(case=case, method=method, stage=label, weights=weights))
                lines.append(f"| {case} | {method} | {label} | {min(valid)} | {len(valid)}/{env['seeds']} |")
    timing = []
    lines += [
        "",
        "## Resources and delivery",
        "",
        "| Method | CPU / elapsed | Late exports | Median fallback seconds/side |",
        "|---|---:|---:|---:|",
    ]
    for method in env["methods"]:
        rs = [r for r in records if r["method"] == method]
        workers = [w for r in rs for ws in r["workers"].values() for w in ws]
        elapsed = sum(w["search_seconds"] for w in workers)
        cpu = sum(w["cpu_seconds"] for w in workers)
        late = sum(s["late_events"] for r in rs for s in r["sides"].values())
        fallback = statistics.median(w["counters"].get("fallback_seconds", 0) for w in workers)
        timing.append(
            dict(
                method=method,
                cpu_seconds=cpu,
                elapsed_seconds=elapsed,
                late_exports=late,
                median_fallback_seconds=fallback,
            )
        )
        lines.append(f"| {method} | {cpu / elapsed:.4f} | {late} | {fallback:.4f} |")
    saved = sum(r["saved_candidates"] for r in records)
    lines += [
        "",
        "All bounds have explicit witnesses. Raw late outputs are retained without deadline credit. "
        "The legacy circulant API exports one winner per approximately 50-ms batch; native guided and "
        "incremental observers target 2-ms batches. This limits comparable time-to-hit precision.",
        "",
        "The structure detector uses check-row order as a clue; qubit relabeling and combined row/column "
        "shuffle controls are separate diagnostics. Proposed mappings or applicability flags alone do not "
        "establish a useful symmetry. Only actual validated witnesses establish bound improvements.",
        "",
        "Reduced-space outputs distinguish the coupled polynomial branch from the known single-block "
        "approach. Its pruned combinations are incomplete, including for pairs. Decoder trials use a "
        "fixed toy-motivated channel jitter; this is not an exact reproduction of the paper's settings.",
        "",
        f"All {saved:,} distinct-per-side/run supports were checked and saved through the research kit. "
        "This is an experimental upper-bound comparison, not an exact-distance certificate or full candidate gate.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        directory / "summary.json",
        dict(cells=cells, paired=paired, stages=stages, timing=timing, saved_candidates=saved),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
