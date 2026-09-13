"""Aggregate completed frozen grids; read-only with respect to measured inputs."""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import atomic_json, sha256  # noqa: E402
from report_candidates import best  # noqa: E402
from report_strategies import checked_records  # noqa: E402


def main():
    grids = {}
    for name, count, seconds, seeds in (
        ("screen-10s", 126, 10, 3),
        ("relabel-10s", 36, 10, 1),
        ("hard-60s", 36, 60, 2),
    ):
        directory = HERE / name
        env, records = checked_records(directory)
        assert len(records) == count
        assert env["seconds_per_code"] == seconds and env["seeds"] == seeds
        audit = json.loads((directory / "audit-candidates.json").read_text())
        assert audit["status"] == "passed" and audit["configurations"] == count
        grids[name] = (env, records, audit)
    first = grids["screen-10s"][0]
    for env, _, _ in grids.values():
        assert env["source_hashes"] == first["source_hashes"]
        assert env["binary_hashes"] == first["binary_hashes"]
    short_signatures = {
        (x["case"], x["side"]): x["signature_sha256"]
        for x in grids["screen-10s"][2]["identical_complete_initialization"]
    }
    for x in grids["hard-60s"][2]["identical_complete_initialization"]:
        assert short_signatures[x["case"], x["side"]] == x["signature_sha256"]

    cells, timing = [], []
    for name, (env, records, _) in grids.items():
        for case in env["cases"]:
            for method in env["methods"]:
                rs = sorted(
                    (r for r in records if r["case"] == case and r["method"] == method),
                    key=lambda r: r["seed"],
                )
                ws = [w for r in rs for workers in r["workers"].values() for w in workers]
                weights = [best(r) for r in rs]
                cells.append(
                    dict(
                        grid=name,
                        case=case,
                        method=method,
                        target=rs[0]["target"],
                        seeds=[r["seed"] for r in rs],
                        weights=weights,
                        median=statistics.median(weights),
                        candidate_only=[best(r, True) for r in rs],
                    )
                )
                elapsed = sum(w["search_seconds"] for w in ws)
                cpu = sum(w["cpu_seconds"] for w in ws)
                timing.append(
                    dict(
                        grid=name,
                        case=case,
                        method=method,
                        total_cpu_seconds=cpu,
                        total_elapsed_seconds=elapsed,
                        cpu_to_elapsed=cpu / elapsed,
                        sector_overrun_median=statistics.median(
                            max(0, w["search_seconds"] - env["seconds_per_code"] / 2) for w in ws
                        ),
                        sector_overrun_max=max(max(0, w["search_seconds"] - env["seconds_per_code"] / 2) for w in ws),
                        preparation_seconds_median=statistics.median(w["counters"]["preparation_seconds"] for w in ws),
                        initialization_seconds_median=statistics.median(
                            w["counters"]["initialization_seconds"] for w in ws
                        ),
                        candidate_seconds_median=statistics.median(
                            w["counters"].get("candidate_seconds", 0) for w in ws
                        ),
                        fallback_seconds_median=statistics.median(w["counters"].get("fallback_seconds", 0) for w in ws),
                        late_events=sum(s["late_events"] for r in rs for s in r["sides"].values()),
                    )
                )
    coupled = []
    hard_cases = set(grids["hard-60s"][0]["cases"])
    for name in ("screen-10s", "hard-60s"):
        env, records, _ = grids[name]
        for record in records:
            if record["method"] != "reduced-space" or record["case"] not in hard_cases:
                continue
            for side, workers in record["workers"].items():
                worker = workers[0]
                value = min(
                    (
                        e["weight"]
                        for e in worker["events"]
                        if e["stage"] == "reduced_both" and e["seconds"] <= env["seconds_per_code"] / 2
                    ),
                    default=None,
                )
                initial = worker["counters"]["initial_best"]
                coupled.append(
                    dict(
                        grid=name,
                        case=record["case"],
                        seed=record["seed"],
                        side=side,
                        initial=initial,
                        coupled_best=value,
                        improved_initial=value is not None and value < initial,
                    )
                )
    files = [
        HERE / name / file
        for name in grids
        for file in ("environment.json", "results.json", "audit-candidates.json", "completed.json")
    ]
    summary = dict(
        configurations=sum(len(rs) for _, rs, _ in grids.values()),
        saved_candidates=sum(a["saved_witness_documents_checked"] for _, _, a in grids.values()),
        source_hashes_identical_across_main_grids=True,
        binary_hashes_identical_across_main_grids=True,
        initialization_identical_across_budgets=True,
        input_hashes={str(p.relative_to(HERE)): sha256(p) for p in files},
        aggregate_source_sha256=sha256(Path(__file__)),
        coupled_branch_checks=coupled,
        coupled_branch_improved_initial=sum(row["improved_initial"] for row in coupled),
        cells=cells,
        timing=timing,
    )
    atomic_json(HERE / "comparison.json", summary)
    lines = ["# Frozen-grid comparison tables", "", "Lower witnessed upper bounds are better.", ""]
    for name, (env, _, _) in grids.items():
        lines += [
            f"## {name}",
            "",
            "| Code | Target | " + " | ".join(env["methods"]) + " |",
            "|---|---:|" + "---:|" * len(env["methods"]),
        ]
        for case in env["cases"]:
            cs = [
                next(c for c in cells if c["grid"] == name and c["case"] == case and c["method"] == method)
                for method in env["methods"]
            ]
            lines.append(f"| {case} | {cs[0]['target']} | " + " | ".join(str(c["median"]) for c in cs) + " |")
        lines.append("")
    lines += [
        "Candidate-only outputs, raw seed values and per-case resource accounting are in comparison.json.",
        "All main grids used identical measured sources/binaries and complete common initialization.",
        "The separate routing diagnostic is excluded from these frozen-grid totals.",
    ]
    (HERE / "TABLES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
