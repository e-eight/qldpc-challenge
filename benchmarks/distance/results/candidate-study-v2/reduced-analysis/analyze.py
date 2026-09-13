"""Read-only analysis of saved results and matrix row spaces; performs no search.

Only writes its JSON/Markdown analysis artifacts. All classified vectors already
occur in completed benchmark records. Run on a non-benchmark CPU, e.g. CPU7.
"""

# Long report-table rows are intentional.
# ruff: noqa: E501
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np  # noqa: E402

CASES = ("board-700-222-28", "board-682-172-79", "regression-690-182")
SEEDS = (1200, 1201, 1202)
METHODS = ("reduced-space", "circulant")
BRANCHES = ("reduced_both", "reduced_single_left", "reduced_single_right")
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "verify/gf2.py").exists())
SPEC = importlib.util.spec_from_file_location("trusted_gf2_analysis", ROOT / "verify/gf2.py")
gf2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gf2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def divide(a, b):
    if not b:
        raise ValueError("Zero polynomial divisor")
    q = 0
    while a and a.bit_length() >= b.bit_length():
        shift = a.bit_length() - b.bit_length()
        q ^= 1 << shift
        a ^= b << shift
    return q, a


def gcd(a, b):
    while b:
        a, b = b, divide(a, b)[1]
    return a


def pack(row):
    return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")


def unpack(word, n):
    return np.array([(word >> j) & 1 for j in range(n)], dtype=np.uint8)


def rotate(word, length):
    mask = (1 << length) - 1
    a, b = word & mask, word >> length
    return (((a << 1) & mask) | (a >> (length - 1))) | ((((b << 1) & mask) | (b >> (length - 1))) << length)


def spaces_and_dimensions(own, opposite):
    """Reconstruct frozen generator row spaces solely for exact membership checks."""
    n = own.shape[1]
    length = n // 2
    assert n % 2 == 0
    mask, modulus = (1 << length) - 1, (1 << length) | 1
    first = pack(own[0])
    a, b = first & mask, first >> length
    f = gcd(a, b)
    h = gcd(f, modulus)
    qa, ra = divide(a, f)
    qb, rb = divide(b, f)
    g, rg = divide(modulus, h)
    assert not (ra or rb or rg)
    seeds = {"reduced_both": qa | (qb << length), "reduced_single_left": g, "reduced_single_right": g << length}
    spaces, ranks, retained = {}, {}, {}
    for label, initial in seeds.items():
        rows = []
        word = initial
        for _ in range(length):
            vector = unpack(word, n)
            if gf2.commutes(vector, opposite):
                rows.append(vector)
            word = rotate(word, length)
        matrix = np.array(rows, dtype=np.uint8).reshape(-1, n)
        reduced, pivots = gf2.rref(matrix)
        spaces[label] = reduced
        ranks[label] = len(pivots)
        retained[label] = len(rows)
    block_ranks = [gf2.rank(opposite[:, i * length : (i + 1) * length]) for i in range(2)]
    return spaces, {
        "block_length": length,
        "opposite_block_ranks": block_ranks,
        "full_single_block_kernel_dimensions": [length - rank for rank in block_ranks],
        "reduced_space_ranks": ranks,
        "retained_generator_counts": retained,
        "common_polynomial_degree": f.bit_length() - 1,
        "shared_modulus_factor_degree": h.bit_length() - 1,
        "own_polynomial_individual_modulus_gcd_degrees": [
            gcd(a, modulus).bit_length() - 1,
            gcd(b, modulus).bit_length() - 1,
        ],
    }


def stage_summary(events, deadline):
    result = {}
    for stage in sorted({e["stage"] for e in events}):
        entries = [(i, e) for i, e in enumerate(events) if e["stage"] == stage and e["seconds"] <= deadline]
        if not entries:
            continue
        index, best = min(entries, key=lambda item: (item[1]["weight"], item[1]["seconds"]))
        result[stage] = {
            "minimum_timely_weight": best["weight"],
            "first_delivery_of_minimum_seconds": best["seconds"],
            "event_index": index,
            "timely_exports": len(entries),
        }
    return result


def analyze(screen):
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "verify/gf2.py",
        ROOT / "benchmarks/distance/strategy_prototypes/reduced_space_v2/adapter.py",
        ROOT / "benchmarks/distance/strategy_prototypes/reduced_space_v2/search.cpp",
        ROOT / "benchmarks/distance/candidate_search.py",
        ROOT / "benchmarks/distance/native.cpp",
        ROOT / "verify/gf2_fast.cpp",
    ]
    report = {
        "protocol": "Read-only stage attribution and exact GF(2) row-space membership of existing saved witnesses; no searches.",
        "scope": {"cases": CASES, "seeds": SEEDS, "methods": METHODS},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python_numpy_version": np.__version__,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "input_sha256": {},
        "cases": {},
        "caveat": "A saved witness outside the tested spaces is unreachable by this prototype. This does not prove that every witness of that weight, or every equivalent representative, is outside the spaces.",
    }

    def record_input(path):
        report["input_sha256"][str(path.resolve().relative_to(ROOT))] = sha(path)

    record_input(screen / "manifest.json")
    record_input(screen / "environment.json")
    environment = json.loads((screen / "environment.json").read_text())
    for relative, digest in report["source_sha256"].items():
        frozen = environment["source_hashes"].get(relative)
        if frozen is not None:
            assert digest == frozen, f"Measured source changed: {relative}"
    for case in CASES:
        matrix_path = screen / "matrices" / f"{case}.npz"
        record_input(matrix_path)
        with np.load(matrix_path) as archive:
            hx, hz = archive["hx"], archive["hz"]
        sector_spaces, dimensions = {}, {}
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            sector_spaces[side], dimensions[side] = spaces_and_dimensions(own, opposite)
        case_report = {"dimensions": dimensions, "runs": [], "circulant_membership": []}
        reduced_runs = []
        for method in METHODS:
            for seed in SEEDS:
                run_path = screen / case / f"{method}-s{seed}" / "result.json"
                record_input(run_path)
                record = json.loads(run_path.read_text())
                assert record["case"] == case and record["method"] == method and record["seed"] == seed
                deadline = record["budget_seconds"] / 2
                run = {"method": method, "seed": seed, "seconds_per_sector": deadline, "sides": {}}
                for side in ("X", "Z"):
                    record_input(run_path.parent / f"{side}.jsonl")
                    worker = record["workers"][side][0]
                    events = worker["events"]
                    logged = [json.loads(line) for line in (run_path.parent / f"{side}.jsonl").read_text().splitlines()]
                    assert logged == events, "Record and event log disagree"
                    stages = stage_summary(events, deadline)
                    counters = worker["counters"]
                    run["sides"][side] = {
                        "stages": stages,
                        "common_preparation_seconds": counters["preparation_seconds"],
                        "common_initialization_seconds": counters["initialization_seconds"],
                        "initial_best": counters["initial_best"],
                        "candidate": counters["candidate"],
                    }
                    if method == "reduced-space":
                        candidate = counters["candidate"]
                        for branch in BRANCHES:
                            assert (
                                candidate["branches"][branch]["rank"] == dimensions[side]["reduced_space_ranks"][branch]
                            )
                    elif "circulant" in stages:
                        index = stages["circulant"]["event_index"]
                        event = events[index]
                        vector = np.zeros(hx.shape[1], dtype=np.uint8)
                        vector[event["support"]] = 1
                        own, opposite = (hx, hz) if side == "X" else (hz, hx)
                        assert int(vector.sum()) == event["weight"]
                        assert gf2.commutes(vector, opposite)
                        assert not gf2.in_rowspace(vector, own)
                        membership = {
                            label: gf2.in_rowspace(vector, matrix) for label, matrix in sector_spaces[side].items()
                        }
                        case_report["circulant_membership"].append(
                            {
                                "seed": seed,
                                "side": side,
                                "weight": event["weight"],
                                "event_index": index,
                                "record": str(run_path.relative_to(ROOT)),
                                "support": event["support"],
                                "physical_support_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                                "in_reduced_spaces": membership,
                                "in_any_reduced_space": any(membership.values()),
                            }
                        )
                run["code_minimum_timely_weight"] = min(
                    stage["minimum_timely_weight"]
                    for side in run["sides"].values()
                    for stage in side["stages"].values()
                )
                case_report["runs"].append(run)
                if method == "reduced-space":
                    reduced_runs.extend(run["sides"].values())
        candidate_counters = [run["candidate"] for run in reduced_runs]
        coupled_times = [c["branches"]["reduced_both"]["seconds"] for c in candidate_counters]
        coupled_fraction = [
            t / sum(b["seconds"] for b in c["branches"].values()) for t, c in zip(coupled_times, candidate_counters)
        ]
        case_report["summary"] = {
            "median_coupled_fraction_of_branch_time": statistics.median(coupled_fraction),
            "median_coupled_seconds_per_sector": statistics.median(coupled_times),
            "median_single_block_combined_seconds_per_sector": statistics.median(
                sum(c["branches"][name]["seconds"] for name in BRANCHES[1:]) for c in candidate_counters
            ),
            "setup_seconds_range": [
                min(c["setup_seconds"] for c in candidate_counters),
                max(c["setup_seconds"] for c in candidate_counters),
            ],
            "native_workspace_bytes_range": [
                min(c["workspace_bytes"] for c in candidate_counters),
                max(c["workspace_bytes"] for c in candidate_counters),
            ],
            "coupled_minimum_weight_range": [
                min(c["branches"]["reduced_both"]["best"] for c in candidate_counters),
                max(c["branches"]["reduced_both"]["best"] for c in candidate_counters),
            ],
            "coupled_improves_initialization_count": sum(
                c["branches"]["reduced_both"]["best"] < run["initial_best"]
                for c, run in zip(candidate_counters, reduced_runs)
            ),
        }
        report["cases"][case] = case_report
    return report


def markdown(report):
    lines = [
        "# Reduced-space branch attribution and exact membership audit",
        "",
        "The coupled polynomial branch did not improve common initialization in any of the 18 measured sectors. "
        "Every reduced-space code-level gain on these three cases came from its single-block branches.",
        "",
        "This is read-only analysis of saved benchmark records and their matrices. No witness search ran. "
        "The accompanying JSON pins every input record, event log, matrix, and analysis/source file by SHA256.",
        "",
        "| Case | Reduced weights by seed | Circulant weights by seed | Coupled-only range | Coupled share of branch time |",
        "|---|---|---|---|---|",
    ]
    for case, data in report["cases"].items():
        by_method = {
            method: [r["code_minimum_timely_weight"] for r in data["runs"] if r["method"] == method]
            for method in METHODS
        }
        summary = data["summary"]
        lines.append(
            f"| {case} | {by_method['reduced-space']} | {by_method['circulant']} | {summary['coupled_minimum_weight_range']} | {summary['median_coupled_fraction_of_branch_time']:.1%} |"
        )
    lines += [
        "",
        "The circulant column reports the benchmark method including its configured guided fallback: "
        "Board700 fails the legacy circulant detector, so those three values are guided fallback results. "
        "All weights above use timely deliveries at 5 seconds per sector (10 seconds/code), seeds 1200–1202. "
        "Time allocation is median over six sectors. Native branch counters include their full final trial; late events receive no quality credit.",
        "",
        "## Exact subspace checks",
        "",
        "`verify/gf2.py` computes the rank of each actual opposite-check block. Its nullity is the full supported-on-one-block kernel dimension. "
        "The audit separately reconstructs the frozen polynomial spaces, verifies generator syndromes, computes their exact ranks, "
        "and checks each saved sector-best circulant witness for membership using the same trusted GF(2) routines.",
        "",
        "| Case/sector | Full left/right kernel dimensions | Reduced left/right dimensions | Coupled dimension |",
        "|---|---|---|---|",
    ]
    for case, data in report["cases"].items():
        for side, dims in data["dimensions"].items():
            ranks = dims["reduced_space_ranks"]
            lines.append(
                f"| {case}/{side} | {dims['full_single_block_kernel_dimensions']} | {[ranks[k] for k in BRANCHES[1:]]} | {ranks['reduced_both']} |"
            )
    lines += [
        "",
        "| Case/seed/sector | Saved circulant weight | Coupled membership | Left membership | Right membership |",
        "|---|---|---|---|---|",
    ]
    for case, data in report["cases"].items():
        for item in data["circulant_membership"]:
            membership = item["in_reduced_spaces"]
            lines.append(
                f"| {case}/{item['seed']}/{item['side']} | {item['weight']} | "
                + " | ".join(str(membership[b]) for b in BRANCHES)
                + " |"
            )
    lines += [
        "",
        "All six saved weight-48 Regression690 witnesses are outside every tested space. "
        "This is an expressivity limitation for those particular vectors, beyond the poor time allocation. "
        "**It does not prove that the reduced spaces contain no weight-48 vector, no lighter logical, or no equally light representative of the same logical classes.**",
        "",
        "Board682's saved X witnesses 75/84/76 belong to the reduced right-block space, so its poorer results there cannot be attributed solely to missing vectors. "
        "Its saved Z witnesses 83/81/81 are outside every tested space.",
        "",
        "## Time and storage",
        "",
        "| Case | Reduction/setup ms per sector | Median coupled seconds | Median combined single-block seconds | Native workspace bytes |",
        "|---|---|---|---|---|",
    ]
    for case, data in report["cases"].items():
        summary = data["summary"]
        lo, hi = summary["setup_seconds_range"]
        lines.append(
            f"| {case} | {1000 * lo:.1f}–{1000 * hi:.1f} | {summary['median_coupled_seconds_per_sector']:.3f} | {summary['median_single_block_combined_seconds_per_sector']:.3f} | {summary['native_workspace_bytes_range']} |"
        )
    lines += [
        "",
        "Setup above excludes the separately charged common preparation and initialization, recorded per sector in JSON. "
        "Native workspace excludes Python/NumPy allocations and is not process RSS.",
        "",
        "## Recommendation",
        "",
        "Discontinue the coupled quotient beam as configured. Its approximately 4,000 bases and 13 million scored combinations per sector "
        "used most of the time without improving initialization. Preserve the cheap single-block opportunity on check-deletion inputs, "
        "and retain the existing full single-block circulant search for the full bicycle cases. A follow-up should use full individual-block "
        "kernels and explicit wall-time allocation before interpreting a larger budget as evidence for this reduced-space design.",
        "",
        "Reproduce from the repository root (CPU7, no searches):",
        "",
        "```sh",
        "taskset -c7 .venv-benchmark/bin/python benchmarks/distance/results/candidate-study-v2/reduced-analysis/analyze.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, default=Path(__file__).resolve().parent.parent / "screen-10s")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = analyze(args.screen.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "README.md").write_text(markdown(report))
    print(
        f"Saved analysis of {len(CASES) * len(SEEDS) * len(METHODS)} existing records and {sum(len(data['circulant_membership']) for data in report['cases'].values())} existing witnesses to {args.output}"
    )


if __name__ == "__main__":
    main()
