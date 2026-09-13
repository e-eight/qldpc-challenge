"""Export complete, validated strategy comparisons from their original records."""

import argparse
import itertools
import json
import statistics
from pathlib import Path

from common import atomic_json, code_target
from study_strategies import summarize


def best(record):
    weights = [side["best_in_budget"] for side in record["sides"].values()]
    return min((w for w in weights if w is not None), default=float("inf"))


def applicable(record):
    return any(worker["status"] != "not_applicable" for workers in record["workers"].values() for worker in workers)


def show(value):
    return "none" if value == float("inf") else f"{value:g}"


def checked_records(directory):
    environment = json.loads((directory / "environment.json").read_text())
    cases = {case["id"]: case for case in json.loads((directory / "manifest.json").read_text())["cases"]}
    records = json.loads((directory / "results.json").read_text())
    expected = set(
        itertools.product(
            environment["cases"],
            environment["methods"],
            range(environment["seed_start"], environment["seed_start"] + environment["seeds"]),
        )
    )
    keys = [(r["case"], r["method"], r["seed"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Duplicate or incomplete strategy grid")
    for record in records:
        raw_dir = directory / record["case"] / f"{record['method']}-s{record['seed']}"
        if json.loads((raw_dir / "result.json").read_text()) != record:
            raise ValueError("Aggregate record differs from original")
        if record["validation_status"] != "passed":
            raise ValueError("Unvalidated record")
        if record["budget_seconds"] != environment["seconds_per_code"]:
            raise ValueError("Budget mismatch")
        target = code_target(cases[record["case"]])
        if record["target"] != target:
            raise ValueError("Target mismatch")
        count = 0
        for side in ("X", "Z"):
            events = [json.loads(line) for line in (raw_dir / f"{side}.jsonl").read_text().splitlines()]
            if events != record["workers"][side][0]["events"]:
                raise ValueError("Event log differs from aggregate")
            if record["sides"][side] != summarize(events, record["budget_seconds"] / 2, target):
                raise ValueError("Deadline summary mismatch")
            count += len({tuple(e["support"]) for e in events})
        if count != record["saved_candidates"]:
            raise ValueError("Witness persistence count mismatch")
    return environment, records


def main(directory):
    env, records = checked_records(directory)
    methods, cases, seeds = env["methods"], env["cases"], env["seeds"]
    lines = [
        "# Strategy prototype comparison",
        "",
        f"{len(records)} configurations; {seeds} seeds per method/case; one worker on CPU {env['cpu']}; "
        f"{env['seconds_per_code']:g} seconds per code, split equally between X and Z.",
        "",
        "Matrices are prepared outside the clock; search state starts cold. Reference supports are never "
        "search inputs. Methods are shuffled and run serially. Searches continue after reaching a target. "
        "Only delivered, within-budget witnesses receive credit; all late evidence is retained.",
        "",
        "## Median best weight (lower is better)",
        "",
        "| Case | Target | " + " | ".join(methods) + " |",
        "|---|---:|" + "---:|" * len(methods),
    ]
    indexed = {(r["case"], r["method"], r["seed"]): r for r in records}
    cells = []
    for case in cases:
        row = []
        for method in methods:
            subset = [r for r in records if r["case"] == case and r["method"] == method]
            active = [r for r in subset if applicable(r)]
            median = statistics.median(best(r) for r in active) if active else None
            hits = sum(any(side["target_hit"] for side in r["sides"].values()) for r in active)
            cell = dict(
                case=case,
                method=method,
                applicable=len(active),
                hits=hits,
                median_best=None if median in (None, float("inf")) else median,
                no_witness=sum(best(r) == float("inf") for r in active),
            )
            cells.append(cell)
            row.append(show(median) if median is not None else "N/A")
        target = next(r["target"] for r in records if r["case"] == case)
        lines.append(f"| {case} | {target} | " + " | ".join(row) + " |")
    lines.extend(
        ["", "## Frozen-target hits", "", "| Case | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    )
    for case in cases:
        row = [f"{c['hits']}/{c['applicable']}" if c["applicable"] else "N/A" for c in cells if c["case"] == case]
        lines.append(f"| {case} | " + " | ".join(row) + " |")
    paired = {}
    if "incremental" in methods:
        lines.extend(
            [
                "",
                "## Paired outcomes against incremental RIS",
                "",
                "| Method | Lower weight | Equal | Higher weight |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in methods:
            if method == "incremental":
                continue
            lower = equal = higher = 0
            for r in records:
                if r["method"] != method or not applicable(r):
                    continue
                control = indexed[(r["case"], "incremental", r["seed"])]
                a, b = best(r), best(control)
                lower += a < b
                equal += a == b
                higher += a > b
            paired[method] = {"lower": lower, "equal": equal, "higher": higher}
            lines.append(f"| {method} | {lower} | {equal} | {higher} |")
    observations = sum(len(w["events"]) for r in records for ws in r["workers"].values() for w in ws)
    saved = sum(r["saved_candidates"] for r in records)
    late = sum(s["late_events"] for r in records for s in r["sides"].values())
    lines.extend(
        [
            "",
            "These are descriptive matched comparisons. Five seeds (or the displayed seed count) do not "
            "establish universal reliability. The circulant pass has a restricted applicability denominator. "
            "No witness is scored as a success merely because it was returned after the deadline.",
            "",
            "## Evidence and interpretation",
            "",
            f"All {observations:,} emitted observations are retained; {saved:,} distinct-per-side/run supports "
            f"were independently validated and packaged through the research kit. {late:,} observations arrived late.",
            "",
            "The original per-run event logs and result records, matrices, source archive, and binary hashes "
            "accompany this report. Target values remain upper-bound/reference challenges, not exact-distance claims. "
            "The full candidate gate was not run and no leaderboard entries were changed.",
            "",
            "Connected-region search samples kernel dependencies under local column ordering; it does not "
            "exhaustively enumerate connected supports. Guided search is a small elite-basis experiment, not "
            "a faithful port or benchmark of QDistEvol. Descent includes its own cold RIS seeding within the budget; "
            "its counters distinguish local refinement from seed-search time.",
            "",
            "Workspace allocation differs between interfaces: structure/descent reserve scratch during preparation; "
            "fresh/incremental/guided reserve session storage during timed search. Batching also differs (roughly "
            "2 ms guided, 5 ms controls, 128-basis seeds plus up to 5 ms descent, per-column structure callbacks). "
            "These are comparisons of the implemented search interfaces, not isolated kernel timings.",
            "",
        ]
    )
    if "descent" in methods:
        lines.extend(
            [
                "## Contribution of stabilizer descent",
                "",
                "| Case | Local improvements | Global incumbent improvements | Fraction of time in descent |",
                "|---|---:|---:|---:|",
            ]
        )
        for case in cases:
            runs = [r for r in records if r["case"] == case and r["method"] == "descent"]
            counters = [w["counters"] for r in runs for ws in r["workers"].values() for w in ws]
            local = sum(c["descent_improvements"] for c in counters)
            useful = sum(c["global_descent_improvements"] for c in counters)
            fraction = sum(c["descent_seconds"] for c in counters) / sum(c["elapsed_seconds"] for c in counters)
            lines.append(f"| {case} | {local} | {useful} | {fraction:.1%} |")
        lines.extend(
            [
                "",
                "A zero global count means local moves never improved that run's incumbent. "
                "Positive counts can include late improvements or improvements later superseded by RIS. "
                "The hybrid also restarts RIS sessions after 128 bases, so its comparison with persistent "
                "incremental RIS is not an isolated ablation of stabilizer moves.",
                "",
            ]
        )
    preparation_path = directory.parent / "prepared-basis" / "results.json"
    if preparation_path.exists():
        preparation = json.loads(preparation_path.read_text())
        if any(r["validation_status"] != "passed" for r in preparation):
            raise ValueError("Unvalidated preparation diagnostic")
        lines.extend(
            [
                "",
                "## Prepared witnesses change the practical baseline",
                "",
                "A post-screen diagnostic scanned the canonical logical bases built during setup, without reference "
                "supports or random search. These are independently validated and saved witnesses. Where this simple "
                "initialization already meets a target, warm-search recovery does not demonstrate an end-to-end gain.",
                "",
                "| Case | Best prepared witness | Frozen target |",
                "|---|---:|---:|",
            ]
        )
        for record in preparation:
            if record["case"] in cases:
                lines.append(f"| {record['case']} | {record['best']} | {record['target']} |")
        lines.extend(
            [
                "",
                "See ../prepared-basis/results.json and ../logical-initialization/results.json. "
                "Future runs should initialize all strategies from these witnesses and refresh their targets.",
                "",
            ]
        )
    (directory / "REPORT.md").write_text("\n".join(lines))
    atomic_json(
        directory / "summary.json",
        {
            "configurations": len(records),
            "seeds": seeds,
            "cells": cells,
            "paired_vs_incremental": paired,
            "observations": observations,
            "saved_candidates": saved,
            "late_observations": late,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
