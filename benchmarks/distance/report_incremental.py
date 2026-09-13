"""Summarize completed studies by seed-level quality, preserving missed targets."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from common import atomic_json, code_target
from study_incremental import CASES, METHODS


def best(run):
    values = [s["best_in_budget"] for s in run["sides"].values() if s["best_in_budget"] is not None]
    return min(values, default=float("inf"))


def main(root):
    first = json.loads((root / "t1-1s" / "environment.json").read_text())
    if first.get("cpp_batch_seed_policy") != "blake2b(study_seed,batch_index)":
        raise ValueError("The legacy C++ control needs its seed-policy correction before the final comparison")
    seeds, seed_start = first["seeds"], first["seed_start"]
    lines = [
        "# Incremental RIS comparison",
        "",
        f"Four methods use the same frozen matrices and targets, {seeds} held-out seeds per code, "
        "and 1 or 10 seconds per code split equally between X and Z. Runs use one or four "
        "workers on fixed guest CPUs. Method order is randomized within each case and seed. "
        f"The two-seed pilot used separate seeds 200–201; evaluation uses {seed_start}–{seed_start + seeds - 1}.",
        "",
        "Each side stops on reaching the fixed code target or exhausting its half-budget; unused time "
        "is not transferred between sides. Best-weight results therefore describe target-stopping search.",
        "",
        f"Incremental RIS uses six-pivot fresh reductions every {first['incremental_restart_interval']} scored bases, "
        f"with {first['incremental_exchange_proposals']} "
        "uniform row/nonpivot exchange proposals before each intervening scan. These settings "
        "were fixed before the held-out runs. Correlated bases are not independent RIS trials.",
        "",
        "All supports, including late arrivals, are retained in each study's results.jsonl and "
        "checked against the repository GF(2) routines before export. They were also packaged "
        "and saved through the research kit. These are distance upper bounds, not exact-distance "
        "certificates or newly validated leaderboard submissions.",
        "",
        "Native preparation is outside the search budget. dist-m4ri is the unmodified pinned CLI, "
        "built with the same host optimization flags. Its timeout is 50 ms shorter per side "
        "to allow startup and exit-time witness export; actual exit is still checked against "
        "the full deadline. Thus this measures usable witness delivery with current interfaces, "
        "not pure kernel efficiency or exact internal discovery latency.",
        "",
        "Each table entry is **target hits / seeds; median best weight**. A run succeeds when "
        "either side delivers a witness at or below the fixed code target. Missing witnesses "
        "count as misses. Per-code 95% Wilson intervals are in each SUCCESS_RATES.md. "
        "This seed count provides a coarse reliability estimate.",
        "",
    ]
    if first.get("cpp_control_repair"):
        lines.insert(
            4,
            "The one-worker C++ controls were rerun separately after correcting overlapping batch seeds. "
            "Their measurements were not interleaved with the other methods in time. "
            "See [the seed-policy correction](SEED_POLICY.md) for provenance and limits.\n",
        )
    if (root / "FINDINGS.md").exists():
        lines[2:2] = ["[Findings and next experiment](FINDINGS.md) · [Validation](VALIDATION.md)", ""]
    all_runs = []
    work_lines = [
        "# Scored bases and workspace",
        "",
        "Rates use completed work divided by observed search time, including batch overshoot. "
        "They describe throughput, not in-budget success. Incremental samples are correlated; "
        "fresh reductions are counted separately. Values are medians across seeds, combining "
        "both sides of each run. Workspace is per session across its workers, excluding "
        "prepared bases, thread stacks, Python objects, and allocator overhead.",
        "",
        "| Workers | Budget | Code | Six-pivot bases/s | Incremental bases/s | Ratio | "
        "Incremental fresh reductions/s | Accepted proposals | Incremental workspace KiB |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for workers in (1, 4):
        for seconds in (1, 10):
            label = f"t{workers}-{seconds}s"
            directory = root / label
            environment = json.loads((directory / "environment.json").read_text())
            if environment.get("cpp_batch_seed_policy") != first["cpp_batch_seed_policy"]:
                raise ValueError(f"C++ seed policy mismatch: {label}")
            if environment["workers"] != workers or environment["seconds_per_side"] != seconds / 2:
                raise ValueError(f"Study worker/budget mismatch: {label}")
            for field in (
                "seeds",
                "seed_start",
                "corpus_manifest_sha256",
                "ris_binary_sha256",
                "native_binary_sha256",
                "m4ri_binary_sha256",
                "incremental_restart_interval",
                "incremental_exchange_proposals",
                "stop_at_target",
            ):
                if environment[field] != first[field]:
                    raise ValueError(f"Study configuration mismatch for {field}: {label}")
            corpus = json.loads((directory / "corpus.json").read_text())
            cases = {c["id"]: c for c in corpus["cases"]}
            runs = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
            expected = {
                (c, m, s)
                for c in CASES
                for m in METHODS
                for s in range(environment["seed_start"], environment["seed_start"] + environment["seeds"])
            }
            observed = {(r["case"], r["method"], r["seed"]) for r in runs}
            if (
                observed != expected
                or len(runs) != len(expected)
                or any(r["validation_status"] != "passed" for r in runs)
            ):
                raise ValueError(f"Incomplete or unvalidated study: {label}")
            grouped = collections.defaultdict(list)
            for run in runs:
                grouped[run["case"], run["method"]].append(run)
                all_runs.append(run)
            lines += [
                f"## {workers} worker(s), {seconds} seconds per code",
                "",
                "| Code | Target | Original C++ | Six-pivot RIS | Incremental RIS | dist-m4ri |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for case in CASES:
                target = code_target(cases[case])
                entries = []
                for method in METHODS:
                    values = [best(r) for r in grouped[case, method]]
                    hits = sum(v <= target for v in values)
                    median = statistics.median(values)
                    entries.append(f"{hits}/{len(values)}; {median:g}")
                lines.append(f"| {case} | {target} | " + " | ".join(entries) + " |")
            for case in CASES:
                rates = {}
                for method in ("ris-block6", "ris-incremental"):
                    method_rates = []
                    reductions, acceptance, workspace = [], [], []
                    for run in grouped[case, method]:
                        ws = [w for group in run["workers"].values() for w in group]
                        elapsed = sum(w["search_seconds"] for w in ws)
                        method_rates.append(sum(w["trials"] for w in ws) / elapsed)
                        reductions.append(sum(w["reductions"] for w in ws) / elapsed)
                        proposals = sum(w["proposals"] for w in ws)
                        acceptance.append(sum(w["exchanges"] for w in ws) / proposals if proposals else 0)
                        workspace.append(max(w["workspace_bytes"] for w in ws) / 1024)
                    rates[method] = statistics.median(method_rates)
                    if method == "ris-incremental":
                        fresh_rate = statistics.median(reductions)
                        accept = statistics.median(acceptance)
                        size = statistics.median(workspace)
                a, b = rates["ris-block6"], rates["ris-incremental"]
                work_lines.append(
                    f"| {workers} | {seconds}s | {case} | {a:,.0f} | {b:,.0f} | "
                    f"{b / a:.2f}× | {fresh_rate:,.0f} | {accept:.1%} | {size:,.0f} |"
                )
            indexed = {(r["case"], r["method"], r["seed"]): r for r in runs}
            lines += ["", "Paired best-weight comparisons (incremental lower / equal / higher):", ""]
            for method in ("cpp", "ris-block6", "m4ri"):
                outcomes = collections.Counter()
                for r in runs:
                    if r["method"] != "ris-incremental":
                        continue
                    a, b = best(r), best(indexed[r["case"], method, r["seed"]])
                    outcomes["lower" if a < b else "higher" if a > b else "equal"] += 1
                lines.append(f"- Versus {method}: {outcomes['lower']} / {outcomes['equal']} / {outcomes['higher']}.")
            late = sum(
                w["search_seconds"] > seconds / 2
                for r in runs
                if r["method"] == "m4ri"
                for ws in r["workers"].values()
                for w in ws
            )
            count = sum(r["method"] == "m4ri" for r in runs) * 2
            lines += [
                "",
                f"dist-m4ri exports after the deadline: {late}/{count} sides.",
                "",
                f"[Raw evidence]({label}/results.jsonl), [seed intervals]({label}/SUCCESS_RATES.md), "
                f"[detailed report]({label}/REPORT.md).",
                "",
            ]
    lines += [
        "## Interpretation",
        "",
        "Paired comparisons describe this fixed corpus and seed grid. Matching seed labels "
        "do not imply matching random trials between engines. Do not pool these results into "
        "a universal success probability. Guest affinity does not guarantee exclusive physical "
        "cores on the host; repeat on the intended deployment machine.",
        "",
        "[Scored-base throughput and workspace](WORK.md) separate correlated scans from fresh reductions. "
        "Compare the recorded payloads with the deployment CPU's cache sizes; larger instances are needed "
        "to measure behavior when the matrices exceed cache capacity.",
        "",
    ]
    if (root / "hardware.json").exists():
        lines += ["[Guest hardware snapshot](hardware.json).", ""]
    indexed = {(r["case"], r["method"], r["seed"], r["threads"], r["budget_seconds"]): r for r in all_runs}
    paired = {}
    for method in ("cpp", "ris-block6", "m4ri"):
        outcomes = collections.Counter()
        for run in all_runs:
            if run["method"] != "ris-incremental":
                continue
            other = indexed[run["case"], method, run["seed"], run["threads"], run["budget_seconds"]]
            a, b = best(run), best(other)
            outcomes["lower" if a < b else "higher" if a > b else "equal"] += 1
        paired[method] = dict(outcomes)
    atomic_json(
        root / "summary.json",
        {
            "validated_configurations": len(all_runs),
            "saved_candidates": sum(r["saved_candidates"] for r in all_runs),
            "paired_incremental_best_weights": paired,
            "note": "Descriptive counts across the fixed code/budget/worker grid; not independent Bernoulli trials.",
        },
    )
    (root / "WORK.md").write_text("\n".join(work_lines) + "\n")
    (root / "README.md").write_text("\n".join(lines))
    print(f"Summarized {len(all_runs)} validated configurations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
