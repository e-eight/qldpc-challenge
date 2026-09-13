"""POST-HOC classification of frozen evaluation references; no witness search.

This script never passes references to any search implementation. It reconstructs
only the three already frozen polynomial spaces for exact row-space membership,
using the previous analysis helper unchanged and trusted verifier GF(2) routines.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "verify/gf2.py").exists())
HELPER = HERE.parent / "reduced-analysis/analyze.py"
SPEC = importlib.util.spec_from_file_location("frozen_space_readonly_analysis", HELPER)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
CASES = ("board-700-222-28", "board-682-172-79", "regression-690-182")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(screen):
    manifest_path = screen / "manifest.json"
    environment_path = screen / "environment.json"
    manifest = json.loads(manifest_path.read_text())
    environment = json.loads(environment_path.read_text())
    cases = {case["id"]: case for case in manifest["cases"]}
    source_paths = [
        Path(__file__).resolve(),
        HELPER,
        ROOT / "verify/gf2.py",
        ROOT / "benchmarks/distance/strategy_prototypes/reduced_space_v2/adapter.py",
        ROOT / "benchmarks/distance/strategy_prototypes/reduced_space_v2/search.cpp",
    ]
    result = {
        "protocol": (
            "POST-HOC read-only membership analysis of frozen evaluation reference supports. "
            "Only existing coupled/left/right polynomial spaces are reconstructed. No new spaces, "
            "logical search, parameter change, or reference injection into search."
        ),
        "scope": {"cases": CASES, "sectors": ["X", "Z"]},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numpy_version": np.__version__,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (manifest_path, environment_path)},
        "cases": {},
        "limitations": [
            "Membership says whether this exact saved reference vector is represented by a tested space.",
            "Nonmembership does not prove all vectors of that weight, or equally light representatives, are absent.",
            "Membership does not establish that the randomized basis/beam policy can find the vector within a budget.",
            "Reference weights are upper bounds, not independent optimality proofs or full candidate-gate passes.",
            "Observations were obtained after freezing and evaluating the algorithms and must be labeled post-hoc.",
        ],
    }
    for relative, digest in result["source_sha256"].items():
        pinned = environment["source_hashes"].get(relative)
        if pinned is not None:
            assert digest == pinned, f"Frozen source changed: {relative}"
    for case_id in CASES:
        case = cases[case_id]
        path = screen / "matrices" / case["file"]
        result["input_sha256"][str(path.relative_to(ROOT))] = sha(path)
        with np.load(path) as archive:
            hx, hz = archive["hx"], archive["hz"]
        result["cases"][case_id] = {}
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            spaces, dimensions = analysis.spaces_and_dimensions(own, opposite)
            support = case["reference"][side]
            assert len(set(support)) == len(support)
            assert all(isinstance(j, int) and 0 <= j < own.shape[1] for j in support)
            vector = np.zeros(own.shape[1], dtype=np.uint8)
            vector[support] = 1
            syndrome_ok = analysis.gf2.commutes(vector, opposite)
            nontrivial = not analysis.gf2.in_rowspace(vector, own)
            assert syndrome_ok and nontrivial, "Frozen reference is not a valid logical on its archived matrices"
            membership = {label: analysis.gf2.in_rowspace(vector, matrix) for label, matrix in spaces.items()}
            length = own.shape[1] // 2
            result["cases"][case_id][side] = {
                "reference_field": f"cases[id={case_id}].reference.{side}",
                "source_manifest": str(manifest_path.relative_to(ROOT)),
                "reference_provenance": case.get("reference_provenance", {}).get(side),
                "support": support,
                "weight": len(support),
                "physical_support_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                "block_weights": [int(vector[:length].sum()), int(vector[length:].sum())],
                "syndrome_ok": syndrome_ok,
                "nontrivial_against_actual_own_rowspace": nontrivial,
                "in_frozen_spaces": membership,
                "in_any_frozen_space": any(membership.values()),
                "dimensions": dimensions,
            }
    return result


def markdown(result):
    lines = [
        "# Post-hoc membership of frozen reference witnesses",
        "",
        "This diagnostic uses evaluation references **after the search implementations and runs were frozen**. "
        "It does not run a search, change parameters, or feed references to any search algorithm. "
        "It checks only membership in the same three spaces already tested.",
        "",
        "| Case | Sector | Reference weight | Left/right weights | Coupled | Left-only | Right-only |",
        "|---|---|---|---|---|---|---|",
    ]
    for case, sides in result["cases"].items():
        for side, data in sides.items():
            membership = data["in_frozen_spaces"]
            values = " | ".join("yes" if membership[name] else "no" for name in analysis.BRANCHES)
            lines.append(f"| {case} | {side} | {data['weight']} | {data['block_weights']} | {values} |")
    lines += [
        "",
        "All six saved supports passed syndrome and nontriviality checks against the archived actual matrices, "
        "using `verify/gf2.py`. Exact block ranks and frozen-space ranks are included in the accompanying JSON.",
        "",
        "## Interpretation",
        "",
    ]
    regression = result["cases"]["regression-690-182"]
    for side, data in regression.items():
        labels = [label for label, member in data["in_frozen_spaces"].items() if member]
        if labels:
            lines.append(
                f"Regression690's saved {side} reference of weight {data['weight']} belongs to {', '.join(labels)}. "
                "For this exact vector, lack of representation does not explain missing the target; "
                "the tested search policy and its budget remain the relevant limitation."
            )
        else:
            lines.append(
                f"Regression690's {side} reference of weight {data['weight']} lies outside all three frozen spaces. "
                "This establishes a representation gap for that exact target vector."
            )
        lines.append("")
    lines += [
        "These statements concern **the supplied vectors**. Excluding one reference or an observed weight-48 "
        "witness does not establish that all logicals of that weight are absent. Conversely, containing a reference "
        "does not establish an efficient path to finding it with randomized information sets and a bounded beam.",
        "",
        "The earlier weight-48 membership diagnostic classified search-produced witnesses; this separate artifact "
        "classifies evaluation references. They answer different questions and should not be conflated.",
        "",
        "## Evidence and reproduction",
        "",
        "`analysis.json` includes the exact existing supports, their physical-vector hashes, manifest field paths, "
        "source and input SHA256 hashes, checks, and dimensions. The unchanged earlier read-only helper reconstructs "
        "the frozen generator spaces; trusted GF(2) routines determine membership. Source hashes overlapping the "
        "benchmark environment must match its frozen pins.",
        "",
        "Run from the repository root on CPU7:",
        "",
        "```sh",
        "taskset -c7 .venv-benchmark/bin/python "
        "benchmarks/distance/results/candidate-study-v2/reference-membership/analyze.py",
        "```",
        "",
        "Witness weights remain checked upper bounds; this diagnostic supplies no exact-distance proof.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, default=HERE.parent / "screen-10s")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    result = analyze(args.screen.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "README.md").write_text(markdown(result))
    print(
        json.dumps(
            {
                case: {
                    side: {key: data[key] for key in ("weight", "block_weights", "in_frozen_spaces")}
                    for side, data in sides.items()
                }
                for case, sides in result["cases"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
