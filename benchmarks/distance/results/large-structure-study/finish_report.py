"""Assemble audited results and retain a portable copy of the affine witness."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def read(path):
    return json.loads(path.read_text())


def main():
    decision = read(ROOT / "decision.json")
    assert decision["status"] == "complete"
    grids = ("screen-2s", "confirmation-20s")
    cells = {(grid, c["case"], c["method"]): c for grid in grids for c in read(ROOT / grid / "summary.json")["cells"]}
    accounting = []
    for grid in grids + (("finite-order6",) if decision["go"] else ()):
        records = read(ROOT / grid / "results.json")
        accounting.append(
            dict(
                grid=grid,
                configurations=len(records),
                allocated_search_seconds=sum(r["budget_seconds"] for r in records),
                actual_search_seconds=sum(
                    w["search_seconds"] for r in records for ws in r["workers"].values() for w in ws
                ),
                validation_seconds=sum(r["validation_seconds"] for r in records),
                saved_witness_documents=sum(r["saved_candidates"] for r in records),
                exported_observations=sum(
                    len(w["events"]) for r in records for ws in r["workers"].values() for w in ws
                ),
            )
        )
    (ROOT / "accounting.json").write_text(json.dumps(accounting, indent=2) + "\n")
    lines = [
        "# Bounded large-code structural experiment",
        "",
        "The frozen continuation criterion was "
        + ("met on the affine code." if decision["go"] else "not met: stop this line."),
        "The local submitted corpus contains 593 entries, with maximum n=700. "
        "This experiment uses three large cases from distinct constructions, informed by their originating PRs. "
        "The [research notes](PR_RESEARCH.md) explain selection, exclusions and primary sources; "
        "the [frozen plan](../../strategy_prototypes/large_structure/PLAN.md) defines the hypotheses "
        "and go/no-go rule.",
        "",
        "## Main comparison",
        "",
        "54 configurations: three methods, three cases, three seeds, at 2 and 20 seconds/code. "
        "The total allocated search budget was 594 seconds. Runs were serial on CPU0, one worker, "
        "with common initialization and all search preparation inside the clock, split equally between X and Z. "
        "Matrices were loaded before timing. Witness validation and packaging occurred between runs and are excluded "
        "from search time; [accounting.json](accounting.json) reports that substantial overhead separately.",
        "The frozen 20s auditor initially rejected slice containment for one torus-control worker because "
        "engine slices and harness events use different clock origins. The supplemental "
        "[clock-origin audit](confirmation-20s/clock-origin-audit.json) checks that one bounded constant offset "
        "explains all its events, then rechecks all allocation decisions on an in-memory aligned copy. "
        "Original records, event times and deadline scoring are unchanged. The "
        "[supplemental auditor](audit_clock_offset.py) and two synthetic offset tests accompany the report.",
        "",
        "Weights below list all three seeds. Lower is better. Targets are the stored witness bounds, "
        "not independently established minimum distances.",
        "",
        "| Input | Target | Guided, 2s | Structured, 2s | Guided, 20s | Dispatcher, 20s | Structured, 20s |",
        "|---|---:|---|---|---|---|---|",
    ]
    for case in decision["cases"]:
        name = case["case"]
        values = [
            cells[g, name, m]["weights"]
            for g, m in (
                ("screen-2s", "guided"),
                ("screen-2s", "structure"),
                ("confirmation-20s", "guided"),
                ("confirmation-20s", "race"),
                ("confirmation-20s", "structure"),
            )
        ]
        lines.append(f"| {name} | {case['target']} | " + " | ".join(str(v) for v in values) + " |")
    lines += [
        "",
        "Product-block and thickened-torus-strip restrictions matched the controls' bounds, which were already "
        "recovered at 2 seconds. Stop those directions here. The [decision](decision.json) evaluates the exact "
        "predeclared criterion, using structured-stage outputs rather than initialization or guided fallback. "
        "The complete [2s report](screen-2s/REPORT.md) and [20s report](confirmation-20s/REPORT.md) include censored "
        "per-sector first-hit times. Those times are not whole-code times: X and Z run serially.",
    ]
    if decision["go"]:
        finite = read(ROOT / "finite-order6/results.json")
        assert read(ROOT / "finite-order6/finite-audit.json")["status"] == "passed"
        lines += [
            "",
            "## Deterministic follow-up and sibling controls",
            "",
            "After observing the main result, fix the successful left-y³/order-six restriction and enumerate its "
            "entire kernel. This is a post-hoc follow-up, with the restriction held fixed before searching the two "
            "sibling codes. No sibling-dependent tuning or additional budget sweep was performed.",
            "",
            "| Input | Current stored target | Kernel dimensions X/Z | Restricted minima X/Z | "
            "Actual search seconds, X+Z |",
            "|---|---:|---|---|---:|",
        ]
        for r in finite:
            cs = [r["workers"][s][0]["counters"] for s in ("X", "Z")]
            actual = sum(r["workers"][s][0]["search_seconds"] for s in ("X", "Z"))
            lines.append(
                f"| {r['case']} | {r['target']} | {[c['dimension'] for c in cs]} | "
                f"{[c['restricted_best'] for c in cs]} | {actual:.6f} |"
            )
        lines += [
            "",
            "These minima are exact only inside each chosen orbit-constant space. The full code may have shorter "
            "logicals outside it. The timings are a single deterministic diagnostic, including common initialization "
            "and construction of the restriction, not a repeated latency benchmark.",
            "The fixed restriction misses both sibling targets: its best weights are 102 versus 70 and 126 versus 81. "
            "Exhaustive enumeration shows that a larger search budget within these same spaces "
            "cannot close those gaps. "
            "Retain the inexpensive affine probe and the improved main-code witness, but stop this expansion here. "
            "This is one substantial code-specific success; a broadly effective large-code method remains unproven.",
            "",
            "For the main affine code, coordinates represent Aff(F19), indexed 18x+y. Left multiplication by y³ "
            "maps (x,y) to (8x,y+3), modulo 19 and 18. Selecting whole six-element orbits gives 114 binary variables "
            "across the two qubit blocks. Solving the actual checks leaves ten degrees of freedom, hence 1,024 total "
            "combinations. The successful Z witness selects 15 orbits in only the second block. That block's kernel "
            "has dimension 37 before the restriction and seven afterward. This is a concrete algebraic reduction; "
            "it does not require the proposed action to be a quantum-code automorphism.",
            "",
            "[finite_diagnostics.py](finite_diagnostics.py) reconstructs all three affine inputs exactly and records "
            "the kernel dimensions in [finite-diagnostics.json](finite-diagnostics.json). "
            "The [enumerator](finite_probe.py), "
            "[complete results](finite-order6/results.json) and [audit](finite-order6/finite-audit.json) are retained.",
        ]
        r = next(r for r in finite if r["case"] == "684-10-101")
        event = min(
            (e for e in r["workers"]["Z"][0]["events"] if e["stage"] == "orbits_left_0_3"),
            key=lambda e: e["weight"],
        )
        sequence = 0
        for side in ("X", "Z"):
            seen = set()
            for e in r["workers"][side][0]["events"]:
                support = tuple(e["support"])
                if support in seen:
                    continue
                seen.add(support)
                if side == "Z" and e == event:
                    directory = (
                        REPO
                        / "research/candidates/distance-benchmark/684-10-101/finite-order6-exact-s0"
                        / str(sequence)
                    )
                    files = list(directory.glob("*.json"))
                    assert len(files) == 1
                    destination = ROOT / "affine-Z-witness.json"
                    shutil.copyfile(files[0], destination)
                    assert (
                        hashlib.sha256(files[0].read_bytes()).digest()
                        == hashlib.sha256(destination.read_bytes()).digest()
                    )
                sequence += 1
        assert read(ROOT / "affine-Z-witness.json")["distance"]["Z"]["value"] == event["weight"]
        lines += [
            "",
            "A portable copy of the kit-saved [affine Z witness](affine-Z-witness.json) accompanies the report.",
        ]
    lines += [
        "",
        "## Scope and verification",
        "",
        "This is construction-assisted search: the group/lattice/block parameters are explicit inputs. "
        "Targets, code names and reference supports are excluded from Search. The general controls use the "
        "unchanged guided and competing-dispatcher implementations; the dispatcher often follows guided, so "
        "their agreement is not independent algorithmic replication. External tools were not rerun on this new "
        "three-code corpus. There is no claim of a universal speedup, complete search coverage, matrix-only "
        "structure detection, or reliable near-minimum distance search for unfamiliar n≈1000 codes. A half-block-only "
        "search ablation was not measured, so the subgroup restriction is not claimed "
        "to be the only effective reduction.",
        "",
        "The native wrapper constructs restricted kernels and reuses the existing RIS scoring core. This "
        "experiment measures a change in search space, not another rewrite of the hot loop. "
        "42 kernel/dispatcher/control tests, two deterministic-enumeration tests and two clock-audit tests passed; "
        "lint passed. "
        "All emitted supports were retained and every distinct support was packaged via the research kit. "
        "The runner checked commutation and exclusion from the stabilizer row space with the trusted algebra. "
        "The evidence audits check saved documents, raw logs, deadlines, common initialization, matrices, "
        "source archives and binary hashes. These are witness audits; the full candidate promotion gate was not run. "
        "No leaderboard entry, trusted verification file, commit or publication was changed.",
        "",
        "[Hardware](hardware.json), [package versions](packages.json), [test output](tests.txt), "
        "[enumeration tests](finite-tests.txt), [validator integrity](validator-integrity.txt), and "
        "per-grid source archives/environment records accompany the results. "
        "SHA256SUMS covers the retained study artifacts. See [REPRODUCE.md](REPRODUCE.md) for commands.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")
    print("Report and accounting written; portable witness copied when applicable.")


if __name__ == "__main__":
    main()
