"""Report initialized experiments from checked event logs, including trajectories."""

import argparse
import statistics
from pathlib import Path

from common import atomic_json
from report_strategies import checked_records


def weight_at(record, seconds):
    return min(
        (
            e["weight"]
            for ws in record["workers"].values()
            for w in ws
            for e in w["events"]
            if e["seconds"] <= seconds / 2
        ),
        default=None,
    )


def main(directory):
    env, records = checked_records(directory)
    lines = [
        "# Initialized strategy experiment",
        "",
        f"{len(records)} configurations, {env['seeds']} seeds, one worker; "
        f"{env['seconds_per_code']:g} seconds/code split equally between X and Z.",
        "",
        "Preparation, logical combinations, session setup, search, refinement and witness delivery "
        "are charged to the side budget. Imports, matrix loading and independent validation are excluded. "
        "Reference supports and targets are never passed to search. Checkpoints are prefixes of the "
        "same continuing run, not separately initialized trials.",
        "",
    ]
    cells = []
    for budget in env["checkpoints"]:
        lines += [
            f"## Median best weight at {budget:g} seconds/code",
            "",
            "| Case | Target | " + " | ".join(env["methods"]) + " |",
            "|---|---:|" + "---:|" * len(env["methods"]),
        ]
        for case in env["cases"]:
            row = []
            for method in env["methods"]:
                subset = [r for r in records if r["case"] == case and r["method"] == method]
                values = [weight_at(r, budget) for r in subset]
                median = statistics.median(values) if all(v is not None for v in values) else None
                hits = sum(v is not None and v <= r["target"] for v, r in zip(values, subset))
                cells.append(dict(case=case, method=method, seconds=budget, weights=values, median=median, hits=hits))
                row.append(str(median) if median is not None else "missing witness")
            lines.append(f"| {case} | {subset[0]['target']} | " + " | ".join(row) + " |")
        lines += [""]
    paired = []
    if "guided" in env["methods"]:
        for method in env["methods"]:
            if method == "guided":
                continue
            for budget in env["checkpoints"]:
                lower = equal = higher = 0
                for r in records:
                    if r["method"] != method:
                        continue
                    other = next(
                        s
                        for s in records
                        if s["method"] == "guided" and s["case"] == r["case"] and s["seed"] == r["seed"]
                    )
                    a, b = weight_at(r, budget), weight_at(other, budget)
                    if a is None or b is None:
                        continue
                    lower += a < b
                    equal += a == b
                    higher += a > b
                paired.append(dict(method=method, seconds=budget, lower=lower, equal=equal, higher=higher))
    lines += [
        "## Paired outcomes against guided",
        "",
        "| Method | Seconds/code | Lower | Equal | Higher |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in paired:
        lines.append(f"| {p['method']} | {p['seconds']} | {p['lower']} | {p['equal']} | {p['higher']} |")
    lines += [
        "",
        "## Refinement contribution",
        "",
        "| Case | Calls | Local improvements | Global improvements | Refinement seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    refinement = []
    for case in env["cases"]:
        counters = [
            w["counters"]
            for r in records
            if r["case"] == case and r["method"] == "guided-refine"
            for ws in r["workers"].values()
            for w in ws
        ]
        if not counters:
            continue
        stats = {
            key: sum(c.get(key, 0) for c in counters)
            for key in ("descent_calls", "descent_improvements", "global_descent_improvements", "descent_seconds")
        }
        refinement.append(dict(case=case, **stats))
        lines.append(
            f"| {case} | {stats['descent_calls']} | {stats['descent_improvements']} | "
            f"{stats['global_descent_improvements']} | {stats['descent_seconds']:.4f} |"
        )
    saved = sum(r["saved_candidates"] for r in records)
    late = sum(s["late_events"] for r in records for s in r["sides"].values())
    lines += [
        "",
        "## Interpretation and evidence",
        "",
        "These are witnessed upper bounds, not exact distances. Few seeds support screening, not "
        "reliable miss probabilities. Late observations are retained but receive no deadline credit. "
        "Global-refinement counts may include late or later superseded improvements.",
        "",
        "The refinement pool contains up to eight classes from initialization and globally improving "
        "native search events. It does not access every elite basis or restart guided search. "
        "Refinement does not feed a modified basis back into the native population. "
        "The common initial bound is retained by the wrapper; native sessions retain their original "
        "internal scoring policy. The Python policy is an experimental composition of unchanged "
        "native kernels, not a fused or fully optimized C++ implementation.",
        "",
        "Each side prepares both matrix directions independently; guided additionally prepares its "
        "own native session representation. This duplicated setup is measured. Python imports/JIT "
        "and filesystem validation costs remain outside the search comparison.",
        "",
        f"{saved:,} distinct-per-side/run witnesses saved through the research kit; {late} late events. "
        "Original event logs, matrix copies, source archives and binary hashes accompany this report. "
        "The trusted verifier and leaderboard are unchanged; no full candidate gate was run.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        directory / "summary.json",
        {"cells": cells, "paired": paired, "refinement": refinement, "saved_candidates": saved, "late_events": late},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
