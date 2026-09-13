"""Export fixed-work comparisons and reject mismatched search results."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from common import atomic_json


def main(source, output):
    environment = json.loads((source / "environment.json").read_text())
    corpus = json.loads((source / "corpus.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted(source.glob("*/*.json"))]
    methods = environment["methods"]
    if "cpp" not in methods:
        raise ValueError("A fixed-work speedup report requires the cpp baseline")
    expected = {
        (case["id"], method, seed)
        for case in corpus["cases"]
        for method in methods
        for seed in range(environment["seed_start"], environment["seed_start"] + environment["repeats"])
    }
    actual = {(r["case"], r["method"], r["seed"]) for r in records}
    if (
        actual != expected
        or len(records) != len(expected)
        or any(r.get("validation_status") != "passed" for r in records)
    ):
        raise ValueError("Incomplete or unvalidated fixed-work comparison")
    indexed = {(r["case"], r["method"], r["seed"]): r for r in records}
    rates = collections.defaultdict(list)
    ratios = collections.defaultdict(list)
    for record in records:
        case, method, seed = record["case"], record["method"], record["seed"]
        sides = record["sides"]
        rates[case, method].append(sum(s["trials"] for s in sides.values()) / sum(s["seconds"] for s in sides.values()))
        if "cpp" in methods:
            baseline = indexed[case, "cpp", seed]
            for side, measured in sides.items():
                original = baseline["sides"][side]
                best = min(measured["events"], key=lambda event: event["weight"])
                if (
                    measured["trials"] != original["trials"]
                    or measured["best_weight"] != original["best_weight"]
                    or best["support"] != original["events"][0]["support"]
                ):
                    raise ValueError(f"Same-trial result mismatch: {case}, {method}, {seed}, {side}")
            ratios[case, method].append(
                sum(s["seconds"] for s in baseline["sides"].values()) / sum(s["seconds"] for s in sides.values())
            )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "environment.json", environment)
    atomic_json(output / "corpus.json", corpus)
    with (output / "results.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    lines = [
        "# Fixed-work native RIS comparison",
        "",
        f"Workers: {environment['threads']}; affinity: {environment['affinity']}; "
        f"trials per side: {environment['trials_per_side']}; repeats: {environment['repeats']}.",
        "",
        "Rates include dispatch and result conversion, exclude preparation and witness validation, "
        "and do not stop at targets. Entries are medians over repeats, combining X and Z time. "
        "Ratios are medians of paired baseline/engine times. Tiny runs can be dominated by dispatch and scheduling.",
        "",
        "All returned witnesses passed independent GF(2) checks and were saved through the kit. "
        "All methods matched the baseline's best weight and support for identical trial counts and seeds.",
        "",
        "| Case | "
        + " | ".join(f"{m} trials/s" for m in methods)
        + " | "
        + " | ".join(f"{m} speedup" for m in methods if m != "cpp")
        + " |",
        "|---|" + "---:|" * (len(methods) + len(methods) - 1),
    ]
    for case in corpus["cases"]:
        name = case["id"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{statistics.median(rates[name, m]):,.0f}" for m in methods)
            + " | "
            + " | ".join(f"{statistics.median(ratios[name, m]):.2f}x" for m in methods if m != "cpp")
            + " |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
