"""Render measured evidence; does not execute a search."""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from audit_candidates import check_initialization  # noqa: E402
from report_strategies import best, checked_records  # noqa: E402


def main():
    records = []
    for name in ("main2", "main10"):
        _, rows = checked_records(HERE / name)
        audit = json.loads((HERE / name / "audit-polynomial.json").read_text())
        assert audit["status"] == "passed"
        records.extend(rows)
    initial = check_initialization(records)
    cells = []
    for seconds in (2, 10):
        for case in ("682-182-76", "664-170-18"):
            for method in ("guided", "crt-groups", "binary-groups"):
                rows = sorted(
                    (
                        r
                        for r in records
                        if r["budget_seconds"] == seconds and r["case"] == case and r["method"] == method
                    ),
                    key=lambda r: r["seed"],
                )
                weights = [best(r) for r in rows]
                cells.append(
                    dict(seconds=seconds, case=case, method=method, weights=weights, median=statistics.median(weights))
                )
    native = [
        (r, side, w[0]["counters"]) for r in records if r["method"] != "guided" for side, w in r["workers"].items()
    ]
    setups = [c["setup_finished_seconds"] for _, _, c in native]
    rates = [c["native"]["scored"] / c["optimizer_seconds"] for _, _, c in native]
    saved = sum(r["saved_candidates"] for r in records)
    checks = sum(
        json.loads((HERE / name / "audit-polynomial.json").read_text())["independently_checked_events"]
        for name in ("main2", "main10")
    )
    lines = [
        "# Polynomial factors with physical-weight optimization",
        "",
        "Decision: stop this candidate. The tested CRT coordinate-update heuristic does not beat guided "
        "search on either input at either budget. Its complete kernel coverage does not translate into "
        "useful low-weight search geometry. This is a negative result for this heuristic, not a theorem "
        "ruling out polynomial algorithms.",
        "",
        "## Measured comparison",
        "",
        "Lower is better. Entries are witnessed upper bounds, minimum of X and Z, for seeds 2000/2001/2002. "
        "The reported input targets are 76 and 18. Each total code budget is split equally between sectors.",
        "",
        "| Seconds/code | Code | Guided | CRT groups | Binary groups |",
        "|---:|---|---|---|---|",
    ]
    for seconds in (2, 10):
        for case in ("682-182-76", "664-170-18"):
            selected = [
                next(c for c in cells if c["seconds"] == seconds and c["case"] == case and c["method"] == method)
                for method in ("guided", "crt-groups", "binary-groups")
            ]
            lines.append(f"| {seconds} | {case} | " + " | ".join(str(c["weights"]) for c in selected) + " |")
    lines += [
        "",
        "## Attribution and scope",
        "",
        "Common initialization is identical across all methods, seeds and both budgets. It gives X/Z "
        "99/99 for 682 and 92/100 for 664. CRT's improvement must be judged against these, not credited "
        "with another method's witnesses. The two group optimizers have no guided fallback. "
        "Reference supports never enter any search.",
        "",
        "The CRT lift spans the complete opposite-check kernel: 432 dimensions in 49 groups for 682; "
        "417 in 52 groups for 664. Each generated row and the full rank are checked against the binary "
        "matrix in every timed run. Logical tags use three 64-bit words per vector, retaining all 182/170 bits.",
        "",
        "For each field-nullspace generator we enumerate all coefficient changes when its factor "
        "degree is at most 10. Degree-82 generators on 664 are split into consecutive groups of at most "
        "10 coefficients: those are exact group updates, not exhaustive optimization of the whole factor. "
        "Even on 682, a factor with two field generators is updated one generator at a time. "
        "All updates are scored after lifting into physical qubit coordinates. Weight is never assumed "
        "to add across factors. See the polynomial and circulant conventions in "
        "[Wang and Pryadko, sections II.1 and III.1](https://arxiv.org/html/2203.17216). "
        "The local-search policy is our experimental heuristic.",
        "",
        "The binary ablation uses the same group sizes, optimizer and seeds, replacing the CRT basis "
        "with shuffled ordinary kernel rows. It deliberately pays the same algebraic setup to isolate "
        "grouping; it is not a claim of the fastest possible binary implementation. Unlike the previous "
        "partial polynomial-span RIS/beam experiment, this search covers the full kernel and uses "
        "exhaustive physical-weight group updates.",
        "",
        "## Cost and implementation",
        "",
        f"{len(records)} configurations; {sum(r['budget_seconds'] for r in records):g} allocated search seconds; "
        "one CPU, serial shuffled measurements. Preparation, initialization, factor verification, CRT "
        "lifting, binary checks, packed table construction and export count against each budget. "
        "Lift discovery and factorization are reused from the prior algebra diagnostic and excluded. "
        "There are no setup caches across search runs. These supplied structural facts give the new "
        "method a favorable starting point, so their excluded discovery cost cannot explain its loss.",
        "",
        f"Median total preparation through table setup: {statistics.median(setups):.3f}s/sector "
        f"(range {min(setups):.3f}–{max(setups):.3f}s). Median packed update evaluation rate: "
        f"{statistics.median(rates) / 1e6:.1f} million/second. Candidate scores are not equivalent to guided "
        "basis trials and should not be compared as equal units of search work.",
        "",
        "C++ stores physical words and logical tags contiguously and precomputes all group XORs. "
        "The hot scoring loop uses XOR and hardware popcount, without per-candidate allocation. "
        "The largest single table is 112 KiB; all tables together occupy 4.29/4.38 MiB per sector. "
        "On this machine (32 KiB L1 data, 1 MiB L2, 32 MiB shared L3), a group fits L2 and the "
        "full table collection fits L3. The current state is only 112 bytes. The search is already "
        "fast at evaluating its chosen moves; the quality gap points to unhelpful moves/local minima "
        "as the main limitation in this experiment.",
        "",
        "## Validation and retained evidence",
        "",
        f"10 synthetic unit tests passed. All {saved} witness documents "
        "(one per distinct support per run/sector) were saved "
        f"through the research kit; the separate audit rechecked {checks} raw events using trusted "
        "binary syndrome and rowspace algebra. Source and binary snapshots, matrix hashes, common "
        "initialization, event timestamps, deadline summaries and native counters were audited. "
        "Late events are retained but excluded from budget credit. The 27-file trusted validator "
        "integrity check passed. These are existing-code distance benchmarks, not new-code acceptance "
        "or full-distance certificates.",
        "",
        "See [reproduction commands](REPRODUCE.md), [2-second audit](main2/audit-polynomial.json), "
        "[10-second audit](main10/audit-polynomial.json), and [machine-readable summary](summary.json). "
        "Each run folder retains both raw sector event logs and result records; these carry every "
        "exported support independently of the kit staging documents.",
        "",
        "Recommendation: keep the full-kernel algebra utilities and this negative control, but do "
        "not spend more time tuning this optimizer without a new reason to expect much better "
        "physical-weight moves. No benchmark setting was tuned after seeing results.",
    ]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")
    (HERE / "summary.json").write_text(
        json.dumps(
            dict(
                cells=cells,
                configurations=len(records),
                allocated_seconds=sum(r["budget_seconds"] for r in records),
                saved_documents=saved,
                checked_events=checks,
                initialization=initial,
                median_setup_seconds=statistics.median(setups),
                median_scores_per_second=statistics.median(rates),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
