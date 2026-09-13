"""Compare delivered witnesses under matched external/native deadlines."""

import argparse
import statistics
from pathlib import Path

from common import atomic_json
from report_strategies import checked_records


def weight(record, side=None, stage=None):
    workers = record["workers"].values() if side is None else [record["workers"][side]]
    return min(
        (
            e["weight"]
            for ws in workers
            for w in ws
            for e in w["events"]
            if e["seconds"] <= record["budget_seconds"] / 2 and (stage is None or e["stage"] == stage)
        ),
        default=None,
    )


def main(directory):
    env, records = checked_records(directory)
    lines = [
        "# Refreshed external-tool comparison",
        "",
        f"{len(records)} configurations; {env['seeds']} fresh seeds; one worker pinned CPU{env['cpu']}; "
        f"{env['seconds_per_code']:g} seconds/code, split equally between X and Z.",
        "",
        "Preparation, identical logical initialization, session setup, search and observed witness "
        "delivery are charged to the budget. Imports, synthetic-toy JIT warmup, matrix loading and "
        "independent validation are excluded. No method receives a reference target or stops on "
        "reaching one. Separate runs measure each budget; these are not trajectory prefixes.",
        "",
        "## Median best delivered weight (lower is better)",
        "",
        "| Code | Target | " + " | ".join(env["methods"]) + " |",
        "|---|---:|" + "---:|" * len(env["methods"]),
    ]
    cells = []
    for case in env["cases"]:
        row = []
        for method in env["methods"]:
            subset = sorted(
                [r for r in records if r["case"] == case and r["method"] == method], key=lambda r: r["seed"]
            )
            values = [weight(r) for r in subset]
            if any(v is None for v in values):
                raise ValueError("Missing timely initialization witness")
            median = statistics.median(values)
            hits = sum(v <= r["target"] for v, r in zip(values, subset))
            init = [min(w["counters"]["initial_best"] for ws in r["workers"].values() for w in ws) for r in subset]
            improved = sum(v < initial for v, initial in zip(values, init))
            cells.append(
                dict(
                    case=case,
                    method=method,
                    weights=values,
                    median=median,
                    hits=hits,
                    initial_weights=init,
                    improved_initialization=improved,
                    sides={s: [weight(r, s) for r in subset] for s in ("X", "Z")},
                )
            )
            row.append(f"{median:g}")
        lines.append(f"| {case} | {subset[0]['target']} | " + " | ".join(row) + " |")
    lines += [
        "",
        "## Target hits / seeds",
        "",
        "| Code | " + " | ".join(env["methods"]) + " |",
        "|---|" + "---:|" * len(env["methods"]),
    ]
    for case in env["cases"]:
        row = [f"{c['hits']}/{env['seeds']}" for c in cells if c["case"] == case]
        lines.append(f"| {case} | " + " | ".join(row) + " |")
    paired = []
    lines += [
        "",
        "## Paired outcomes against guided",
        "",
        "| Method | Lower | Equal | Higher |",
        "|---|---:|---:|---:|",
    ]
    for method in env["methods"]:
        if method == "guided":
            continue
        counts = dict(lower=0, equal=0, higher=0)
        for r in records:
            if r["method"] != method:
                continue
            other = next(
                s for s in records if s["method"] == "guided" and s["case"] == r["case"] and s["seed"] == r["seed"]
            )
            a, b = weight(r), weight(other)
            counts["lower"] += a < b
            counts["equal"] += a == b
            counts["higher"] += a > b
        paired.append(dict(method=method, **counts))
        lines.append(f"| {method} | {counts['lower']} | {counts['equal']} | {counts['higher']} |")
    lines += [
        "",
        "## Per-sector weights across seeds",
        "",
        "| Code | Method | X | Z | Improved common code-level initialization / seeds |",
        "|---|---|---|---|---:|",
    ]
    for c in cells:
        sides = {s: ", ".join(map(str, c["sides"][s])) for s in ("X", "Z")}
        lines.append(
            f"| {c['case']} | {c['method']} | {sides['X']} | {sides['Z']} | "
            f"{c['improved_initialization']}/{env['seeds']} |"
        )
    timing = []
    lines += [
        "",
        "## Delivery and resource accounting",
        "",
        "| Method | Late events | Mean CPU / elapsed | Median initialization ms/side |",
        "|---|---:|---:|---:|",
    ]
    for method in env["methods"]:
        subset = [r for r in records if r["method"] == method]
        ws = [w for r in subset for workers in r["workers"].values() for w in workers]
        elapsed = sum(w["search_seconds"] for w in ws)
        cpu = sum(w["cpu_seconds"] + w["counters"].get("subprocess_cpu_seconds", 0) for w in ws)
        late = sum(s["late_events"] for r in subset for s in r["sides"].values())
        init_ms = 1000 * statistics.median(
            w["counters"]["preparation_seconds"] + w["counters"]["initialization_seconds"] for w in ws
        )
        timing.append(
            dict(
                method=method,
                late_events=late,
                cpu_seconds=cpu,
                elapsed_seconds=elapsed,
                median_initialization_ms=init_ms,
            )
        )
        lines.append(f"| {method} | {late} | {cpu / elapsed:.3f} | {init_ms:.3f} |")
    saved = sum(r["saved_candidates"] for r in records)
    lines += [
        "",
        "## Scope and limitations",
        "",
        "Common initialization is retained by the wrapper, not injected into each engine's internal "
        "fitness/incumbent. Guided has an additional native preparation step; that cost is measured. "
        "QDistEvol runs its pinned upstream algorithm with population 100 and ten offspring per parent. "
        "The observer returns each scored result unchanged; it retains a first minimum-weight "
        "representative on each strict improvement and interrupts only at the deadline.",
        "",
        "The unchanged M4RI CLI exports at process exit. Its search timeout reserves 50 ms for delivery; "
        "all exported supports are retained, but late ones get no budget credit. We do not infer its "
        "internal time-to-target. Native bases, M4RI iterations and QDistEvol candidates are different "
        "units of work; none is used as a cross-method speedup measure.",
        "",
        "Three seeds are a screening comparison, not a reliable miss-probability estimate. Paired "
        "counts describe this fixed corpus, and do not imply shared random trials across engines. "
        "All weights are witnessed upper bounds, not exact distances. Sector symmetries and family "
        "specific proposals are intentionally absent from every method in this general-search study.",
        "",
        f"All {saved:,} distinct-per-side/run supports were independently checked and saved through the "
        "research kit. Raw event logs, commands, outputs, matrices, source archives and executable "
        "hashes accompany the results. The trusted verifier and leaderboard were unchanged; no "
        "full candidate gate or publication workflow was run.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(directory / "summary.json", dict(cells=cells, paired=paired, timing=timing, saved_candidates=saved))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
