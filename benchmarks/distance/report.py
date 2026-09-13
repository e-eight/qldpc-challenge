"""Export portable raw evidence and a descriptive, non-certifying pilot report."""

import argparse
import collections
import json
import math
import shutil
import tarfile
from pathlib import Path

from common import atomic_json, code_target


def portable(value):
    """Drop machine-local paths; the actual witness supports remain embedded."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items() if key != "raw_file"}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def wilson_interval(hits, trials):
    """Return a 95% Wilson interval for repeated independent seeds on one code."""
    z = 1.959963984540054
    rate = hits / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0, center - radius), min(1, center + radius)


def main(source, output):
    """Describe frozen runs without turning missing results into successful ones."""
    output.mkdir(parents=True, exist_ok=True)
    environment = json.loads((source / "environment.json").read_text())
    corpus = json.loads((source / "corpus.json").read_text())
    cases = {case["id"]: case for case in corpus["cases"]}
    records = [json.loads(path.read_text()) for path in sorted(source.glob("*/*/result.json"))]
    if not records or any(record["validation_status"] != "passed" for record in records):
        raise ValueError("No validated results, or a run still awaits witness validation")
    selected = environment.get("cases", list(cases))
    expected = {
        (case, method, seed)
        for case in selected
        for method in environment["methods"]
        for seed in range(environment["seed_start"], environment["seed_start"] + environment["seeds"])
    }
    observed = {(record["case"], record["method"], record["seed"]) for record in records}
    if observed != expected or len(records) != len(expected):
        raise ValueError(f"Incomplete or duplicated study: missing {len(expected - observed)} expected runs")
    atomic_json(output / "environment.json", environment)
    atomic_json(output / "corpus.json", corpus)
    if (source / "matrices").exists():
        destination = output / "matrices"
        destination.mkdir(exist_ok=True)
        selected_corpus = {**corpus, "cases": [cases[case] for case in selected]}
        atomic_json(destination / "manifest.json", selected_corpus)
        for case_id in selected:
            filename = cases[case_id]["file"]
            shutil.copyfile(source / "matrices" / filename, destination / filename)
    if (source / "source_snapshot").exists():
        with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
            archive.add(source / "source_snapshot", arcname="source_snapshot")
    with (output / "results.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(portable(record), allow_nan=False) + "\n")
    stats = collections.defaultdict(collections.Counter)
    actual_refutations = []
    for record in records:
        method, case = record["method"], cases[record["case"]]
        sides = record["sides"]
        applicable = any(side["status"] != "not_applicable" for side in sides.values())
        if not applicable:
            stats[method]["not_applicable"] += 1
            continue
        stats[method]["runs"] += 1
        ref_targets = [len(support) for support in case.get("reference", {}).values()]
        target = code_target(case)
        timely = [
            event
            for workers in record["workers"].values()
            for worker in workers
            for event in worker["events"]
            if event["within_budget"]
        ]
        hit = target is not None and any(event["weight"] <= target for event in timely)
        if target is not None:
            stats[method]["target_runs"] += 1
            stats[method]["target_hits"] += int(hit)
        if ref_targets:
            stats[method]["validated_target_runs"] += 1
            stats[method]["validated_target_hits"] += int(any(event["weight"] <= min(ref_targets) for event in timely))
        claim = case.get("claimed_distance")
        refuted = claim is not None and any(event["weight"] < claim for event in timely)
        stats[method]["actual_refutations"] += int(refuted)
        if refuted:
            best = min(event["weight"] for event in timely)
            actual_refutations.append((case["id"], method, record["seed"], claim, best))
    lines = [
        "# CPU distance-search pilot",
        "",
        "This is an exploratory implementation comparison, not an exact-distance or cap-raising certificate.",
        "",
        f"Platform: {environment['platform']}; {environment['machine']}. "
        f"Workers: {environment['workers']}. Budget: {2 * environment['seconds_per_side']:g} seconds per code, "
        "split equally between X and Z. Imports/JIT and shared basis preparation are outside this search budget.",
        "",
        f"Seeds per method/case: {environment['seeds']}. Method order is shuffled within each case/seed. "
        "A single-seed pilot does not establish repeated-run reliability. Core affinity: "
        f"{environment['affinity'] or 'unavailable/not fixed'}.",
        "",
        "## Target recovery",
        "",
        "A hit means at least one validated nontrivial logical reached the code's target within its side's deadline. "
        "The validated-reference column uses only witnesses available before the run. "
        "The all-target column also includes analytic toric and initially unverified paper targets. "
        "Not-applicable structural searches are excluded, so their denominator differs.",
        "",
        "| Method | Validated targets reached | All targets reached | Actual claim refutations | Not applicable |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in sorted(stats):
        stat = stats[method]
        lines.append(
            f"| {method} | {stat['validated_target_hits']}/{stat['validated_target_runs']} | "
            f"{stat['target_hits']}/{stat['target_runs']} | {stat['actual_refutations']} | {stat['not_applicable']} |"
        )
    lines += [
        "",
        "## Refutation witnesses",
        "",
        "These observations tighten existing upper bounds. Each supporting vector is retained in results.jsonl "
        "and was checked with the repository's GF(2) routines. No leaderboard entries were changed.",
        "",
        "| Input | Method | Seed | Existing claim | Found weight |",
        "|---|---|---:|---:|---:|",
    ]
    for case, method, seed, claim, best in actual_refutations:
        lines.append(f"| {case} | {method} | {seed} | {claim} | {best} |")
    if not actual_refutations:
        lines.append("| None observed within the pilot deadlines | — | — | — | — |")
    methods = environment["methods"]
    if environment.get("cpp_control_repair"):
        lines.insert(
            4,
            "The C++ control was rerun separately after correcting overlapping batch seeds; "
            "it was not interleaved with the retained measurements of the other methods. "
            "The environment and source archive retain both measurement origins.\n",
        )
    legacy_cpp = environment.get("cpp_batch_seed_policy") != "blake2b(study_seed,batch_index)"
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["case"], record["method"]].append(record)
    if environment["seeds"] > 1:
        success_lines = [
            "# Repeated-seed target recovery",
            "",
            "Intervals are 95% Wilson intervals for a fixed code and method. They describe seed variability, "
            "not uncertainty over the population of codes. Missing targets and inapplicable methods are omitted.",
            "",
            "| Code | Method | Target | Hits / seeds | 95% interval |",
            "|---|---|---:|---:|---:|",
        ]
        for (case_id, method), runs in sorted(grouped.items()):
            case = cases[case_id]
            target = code_target(case)
            applicable = [
                run for run in runs if any(side["status"] != "not_applicable" for side in run["sides"].values())
            ]
            if target is None or not applicable:
                continue
            hits = sum(
                any(
                    event["within_budget"] and event["weight"] <= target
                    for workers in run["workers"].values()
                    for worker in workers
                    for event in worker["events"]
                )
                for run in applicable
            )
            low, high = wilson_interval(hits, len(applicable))
            interval = (
                "overlapping batch streams" if method.startswith("cpp") and legacy_cpp else f"{low:.1%}–{high:.1%}"
            )
            success_lines.append(f"| {case_id} | {method} | {target} | {hits}/{len(applicable)} | {interval} |")
        (output / "SUCCESS_RATES.md").write_text("\n".join(success_lines) + "\n")
    lines += [
        "",
        "## Per-code best weights",
        "",
        "Each cell is the minimum weight observed within budget over both sides and all listed seeds. "
        "Where no timely result is available, an asterisk marks the best late result; "
        "it receives no timely-success credit. "
        "With multiple seeds, these minima are descriptive and do not replace success rates. "
        "The complete per-seed/per-side events and timing are in results.jsonl.",
        "",
        "| Input | n | k | " + " | ".join(methods) + " |",
        "|---|---:|---:|" + "---:|" * len(methods),
    ]
    for case_id in selected:
        case = cases[case_id]
        values = []
        for method in methods:
            sides = [side for record in grouped[case_id, method] for side in record["sides"].values()]
            timely = [side["best_in_budget"] for side in sides if side["best_in_budget"] is not None]
            returned = [side["best_returned"] for side in sides if side["best_returned"] is not None]
            if timely:
                values.append(str(min(timely)))
            elif returned:
                values.append(f"{min(returned)}*")
            else:
                values.append("N/A" if all(side["status"] == "not_applicable" for side in sides) else "—")
        lines.append(f"| {case_id} | {case['n']} | {case['k']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Limits and follow-up",
        "",
        "* This pilot scores search-stage latency. The raw records separate common preparation, dispatch/search, "
        "validation and saving; they do not assert an end-to-end production gate speedup.",
    ]
    if any(method.startswith("cpp") or method == "numpy" for method in methods):
        lines.append(
            "* The original C++ and NumPy adapters use unchanged trial kernels with cached bases, "
            "but reseed each observation batch; streams differ from one long public-API call."
        )
        if legacy_cpp and any(method.startswith("cpp") for method in methods):
            lines.insert(
                4,
                "The legacy C++ adapter reuses overlapping batch-seed streams across adjacent study seeds. "
                "Its repeated-run counts are correlated; no independent-seed intervals are reported for it.\n",
            )
        elif any(method.startswith("cpp") for method in methods):
            lines.append(
                "* C++ observation batches use a BLAKE2b-derived seed for each (study seed, batch index) pair, "
                "avoiding the old additive mapping's systematic overlap between adjacent study seeds."
            )
    if any(method.startswith("ris") for method in methods):
        lines.append(
            "* The standalone RIS engine retains native workers and RNG state across batches. "
            "It has different elimination kernels. Equal-time runs against the original adapter therefore "
            "do not visit identical random trials, even with the same initial seed."
        )
    if "qdistevol" in methods:
        lines.append(
            "* QDistEvol runs independent populations of 100 candidates, with ten offspring per retained parent. "
            "Repeated-seed and longer-budget measurements are needed before judging evolutionary guidance."
        )
    if "m4ri" in methods:
        lines.append(
            "* dist-m4ri exports its witness file at exit. Late exports are retained as best_returned but "
            "receive no in-budget credit. This especially affects runs without early stopping."
        )
        if "m4ri_export_reserve_seconds" in environment:
            lines.append(
                f"* The unmodified dist-m4ri CLI reserves {environment['m4ri_export_reserve_seconds']:g} "
                "seconds per side for process startup and witness export, deducted from its native timeout. "
                "Witness arrival is measured at exit; earlier internal discovery times are unavailable."
            )
    if "ris-incremental" in methods:
        lines.append(
            "* Incremental RIS scores correlated bases between independent random-permutation restarts. "
            "Its trial counter counts scored bases, not independent information sets. Raw records include "
            "full reductions, exchange proposals, and accepted exchanges."
        )
    if any(code_target(cases[case_id]) is None for case_id in selected):
        lines.append(
            "* Some selected cases have no frozen targets. Establish independent references before "
            "held-out seed comparisons; do not tune and evaluate on the same seeds."
        )
    else:
        lines.append(
            "* Targets come from the supplied frozen corpus. Validated witnesses are upper bounds, "
            "and paper/analytic targets remain distinguishable in the raw records."
        )
    lines += ["* Repeat on the intended deployment CPU before selecting a production default.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(f"Exported {len(records)} validated runs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
