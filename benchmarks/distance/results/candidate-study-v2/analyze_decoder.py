"""Read saved candidate-grid evidence; never import or execute a search engine.

Usage: python analyze_decoder.py GRID [--output-prefix PREFIX] [--cpu 7]
Inputs remain unchanged. Output JSON includes file hashes and per-sector records.
"""

# Markdown prose and table templates are intentionally kept on single source lines.
# ruff: noqa: E501

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values):
    return statistics.median(values) if values else None


def distribution(values):
    return {"median": median(values), "maximum": max(values) if values else None, "values": values}


def delivered(record, stage=None):
    deadline = record["budget_seconds"] / 2
    weights = [
        event["weight"]
        for workers in record["workers"].values()
        for worker in workers
        for event in worker["events"]
        if event["seconds"] <= deadline and (stage is None or event["stage"] == stage)
    ]
    return min(weights) if weights else None


def summarize(case, records):
    decoders = sorted((row for row in records if row["method"] == "decoder"), key=lambda row: row["seed"])
    guided = {row["seed"]: row for row in records if row["method"] == "guided"}
    sectors = []
    for record in decoders:
        if record["validation_status"] != "passed":
            raise ValueError("Refusing unvalidated decoder evidence")
        for side, workers in record["workers"].items():
            if len(workers) != 1:
                raise ValueError("This analysis requires one worker per sector")
            worker = workers[0]
            counters = worker["counters"]["candidate"]
            deadline = record["budget_seconds"] / 2
            events = [event for event in worker["events"] if event["stage"] == "decoder"]
            if len(events) != counters["distinct_exports"]:
                raise ValueError("Decoder export count differs from event evidence")
            if counters["successful_decodes"] + counters["failed_decodes"] != counters["trials"]:
                raise ValueError("Decoder trial counters are inconsistent")
            sectors.append(
                {
                    "seed": record["seed"],
                    "side": side,
                    "deadline_seconds": deadline,
                    "search_seconds": worker["search_seconds"],
                    "cpu_seconds": worker["cpu_seconds"],
                    "wall_overrun_seconds": max(0.0, worker["search_seconds"] - deadline),
                    "initial_best": worker["counters"]["initial_best"],
                    "timely_exports": sum(event["seconds"] <= deadline for event in events),
                    "late_exports_from_event_clock": sum(event["seconds"] > deadline for event in events),
                    "timely_decoder_best": min(
                        (event["weight"] for event in events if event["seconds"] <= deadline), default=None
                    ),
                    "candidate_counters": counters,
                }
            )
    totals = {
        key: sum(sector["candidate_counters"][key] for sector in sectors)
        for key in (
            "trials",
            "successful_decodes",
            "failed_decodes",
            "bp_converged",
            "bp_iterations",
            "distinct_exports",
            "duplicate_outputs",
            "construction_seconds",
            "decoder_seconds",
            "elapsed_seconds",
            "trial_seconds",
            "detector_weight_sum",
        )
    }
    trial_count, elapsed = totals["trials"], totals["elapsed_seconds"]
    totals.update(
        {
            "timely_exports": sum(sector["timely_exports"] for sector in sectors),
            "late_exports_from_event_clock": sum(sector["late_exports_from_event_clock"] for sector in sectors),
            "trials_per_actual_candidate_second": trial_count / elapsed if elapsed else None,
            "construction_fraction": totals["construction_seconds"] / elapsed if elapsed else None,
            "decode_fraction": totals["decoder_seconds"] / elapsed if elapsed else None,
            "mean_detector_weight": totals["detector_weight_sum"] / trial_count if trial_count else None,
            "mean_bp_iterations": totals["bp_iterations"] / trial_count if trial_count else None,
            "max_native_decode_seconds": max(
                (s["candidate_counters"]["max_decoder_seconds"] for s in sectors), default=None
            ),
            "max_whole_trial_seconds": max(
                (s["candidate_counters"]["max_trial_seconds"] for s in sectors), default=None
            ),
            "median_adapter_setup_seconds": median([s["candidate_counters"]["setup_seconds"] for s in sectors]),
            "cpu_to_search_wall_ratio": sum(s["cpu_seconds"] for s in sectors)
            / sum(s["search_seconds"] for s in sectors)
            if sectors
            else None,
        }
    )
    code_overruns = [
        max(
            0.0,
            sum(worker["search_seconds"] for ws in row["workers"].values() for worker in ws) - row["budget_seconds"],
        )
        for row in decoders
    ]
    outcomes = []
    for row in decoders:
        other = guided.get(row["seed"])
        initial = min(worker["counters"]["initial_best"] for ws in row["workers"].values() for worker in ws)
        outcomes.append(
            {
                "seed": row["seed"],
                "common_initial_best": initial,
                "decoder_delivered_best": delivered(row),
                "decoder_only_delivered_best": delivered(row, "decoder"),
                "guided_delivered_best": delivered(other) if other else None,
            }
        )
    paired = {"lower": 0, "equal": 0, "higher": 0, "missing": 0}
    for outcome in outcomes:
        a, b = outcome["decoder_delivered_best"], outcome["guided_delivered_best"]
        label = "missing" if a is None or b is None else "lower" if a < b else "higher" if a > b else "equal"
        paired[label] += 1
    return {
        "case": case["id"],
        "metadata": {
            key: case[key]
            for key in (
                "n",
                "k",
                "shape_x",
                "shape_z",
                "rank_x",
                "rank_z",
                "max_row_weight",
                "max_column_weight",
                "matrix_sha256",
            )
        },
        "decoder_runs": len(decoders),
        "totals": totals,
        "sectors": sectors,
        "outcomes": outcomes,
        "sector_wall_overrun_seconds": distribution([s["wall_overrun_seconds"] for s in sectors]),
        "code_wall_overrun_seconds": distribution(code_overruns),
        "paired_vs_guided": paired,
    }


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(report):
    lines = [
        "# Decoder evidence analysis",
        "",
        f"Grid: `{report['grid']}`. Saved records: {report['records_present']}/{report['records_expected']}.",
        "",
        "This reads existing records and matrix metadata only. It executes no decoding or distance search. Every input file is SHA-256 pinned in the accompanying JSON.",
        "",
        "## Delivered bounds",
        "",
        "Lower is better. Decoder combined includes common initialization; decoder-only excludes it. Missing decoder-only delivery means no valid decoder output arrived by the deadline.",
        "",
        "| Case | Seeds | Common init | Decoder combined | Decoder only | Guided |",
        "|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        outcomes = case["outcomes"]

        def values(key):
            return ", ".join("—" if row[key] is None else str(row[key]) for row in outcomes)

        lines.append(
            f"| {case['case']} | {values('seed')} | {values('common_initial_best')} | {values('decoder_delivered_best')} | {values('decoder_only_delivered_best')} | {values('guided_delivered_best')} |"
        )
    lines += [
        "",
        "## Actual decoder work",
        "",
        "Time shares divide by candidate elapsed time, including final overrun, and exclude shared preparation/initialization. Decoder time includes BP and OSD together; construction includes conversion, graph creation, and native setup.",
        "",
        "| Case | Trials | Trials/s | Construction | Decode | BP converged | Mean BP iterations | Timely exports | Mean detector weight / original max check |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        t = case["totals"]
        lines.append(
            f"| {case['case']} | {t['trials']} | {fmt(t['trials_per_actual_candidate_second'])} | {fmt(100 * t['construction_fraction'], 2)}% | {fmt(100 * t['decode_fraction'], 2)}% | {t['bp_converged']} | {fmt(t['mean_bp_iterations'], 1)} | {t['timely_exports']} | {fmt(t['mean_detector_weight'], 2)} / {case['metadata']['max_row_weight']} |"
        )
    lines += [
        "",
        "## Deadline granularity",
        "",
        "Overrun is actual worker search wall time beyond its allocation. Code overrun sums X/Z search time, excluding validation. Late supports are preserved and receive no budget credit.",
        "",
        "| Case | Sector overrun median / max (s) | Code overrun median / max (s) | Max whole trial (s) | Max decode call (s) | CPU / search wall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        t, s, c = case["totals"], case["sector_wall_overrun_seconds"], case["code_wall_overrun_seconds"]
        lines.append(
            f"| {case['case']} | {fmt(s['median'], 6)} / {fmt(s['maximum'], 6)} | {fmt(c['median'], 6)} / {fmt(c['maximum'], 6)} | {fmt(t['max_whole_trial_seconds'], 6)} | {fmt(t['max_native_decode_seconds'], 6)} | {fmt(t['cpu_to_search_wall_ratio'], 6)} |"
        )
    lines += ["", "## Interpretation and limits", ""]
    lines += [f"- {text}" for text in report["caveats"]]
    lines += [
        "",
        "The JSON includes augmented-input source dimensions, exact per-sector counters, per-seed delivered weights, all input hashes, and original source/binary pins copied from the grid environment. Matrix file hashes pin their bytes; semantic matrix hashes are those recorded by the existing corpus manifest.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--cpu", type=int, default=7)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    grid = args.grid.resolve()
    raw = {name: (grid / name).read_bytes() for name in ("results.json", "manifest.json", "environment.json")}
    records, manifest, environment = (
        json.loads(raw[name]) for name in ("results.json", "manifest.json", "environment.json")
    )
    cases = []
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    for case in manifest["cases"]:
        selected = [row for row in records if row["case"] == case["id"]]
        if not any(row["method"] == "decoder" for row in selected):
            continue
        item = summarize(case, selected)
        item["augmented_shapes"] = {"X": [case["shape_z"][0] + 1, case["n"]], "Z": [case["shape_x"][0] + 1, case["n"]]}
        cases.append(item)
        matrix_path = Path("matrices") / case["file"]
        hashes[str(matrix_path)] = digest(grid / matrix_path)
    report = {
        "grid": grid.name,
        "analysis_cpu": args.cpu,
        "analysis_script_sha256": digest(Path(__file__)),
        "records_present": len(records),
        "records_expected": len(environment["cases"]) * len(environment["methods"]) * environment["seeds"],
        "seconds_per_code": environment["seconds_per_code"],
        "input_sha256": hashes,
        "measurement_source_hashes": environment["source_hashes"],
        "measurement_binary_hashes": environment["binary_hashes"],
        "cases": cases,
        "caveats": [
            "This is evidence analysis, not an independent witness audit or candidate-gate pass. Validation status and saved event identity come from the existing audited harness.",
            "All stated bounds are witnessed upper bounds. Common initialization can determine the combined answer even when the decoder produces nothing useful or timely.",
            "The native constructor and decoder are noninterruptible. A low trial count or zero timely decoder exports limits what the deadline comparison says about search quality.",
            "BP nonconvergence does not mean its iterations were useless: the resulting reliabilities feed OSD. The counters do not split BP time from OSD time.",
            "Dense logical detector weights are measured. Their causal contribution to poor BP convergence, sparse elimination fill, or overall search quality remains an inference.",
            "Constructor timing does not distinguish dense-to-CSR conversion, graph insertion and native OSD setup. Profiling is required before attributing the bottleneck to one component.",
            "Larger augmented dimensions alone do not explain the case differences; sparsity, rank redundancy and elimination structure may matter. No controlled ablation is included here.",
            "A future reuse experiment could amortize decoder construction across several reliability perturbations of one detector, but must verify that channel updates reach both BP and OSD. That would change sampling and requires a separately frozen comparison.",
            "Lighter logical detectors and fewer BP iterations are plausible follow-ups, not measured improvements. Stronger time limits must retain any final valid witness and preserve late-delivery accounting.",
            "These few seeds are an initial screen, not a calibrated reliability estimate. Candidate throughput counts are not comparable to guided basis trials.",
        ],
    }
    output = args.output_prefix or grid.parent / (grid.name + "-decoder-analysis")
    if output.resolve().is_relative_to(grid):
        raise ValueError("Analysis outputs must remain outside the measured grid")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in ((".json", json.dumps(report, indent=2) + "\n"), (".md", markdown(report))):
        destination = Path(str(output) + suffix)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(data)
        temporary.replace(destination)
    print(json.dumps({"cases": len(cases), "records": len(records), "output_prefix": str(output)}))


if __name__ == "__main__":
    main()
