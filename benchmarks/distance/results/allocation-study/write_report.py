"""Render the completed allocation comparison without running search."""

import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    comparison = json.loads((BASE / "comparison.json").read_text())
    decisions = json.loads((BASE / "decision-summary.json").read_text())
    lines = [
        "# Competing guided and structured search",
        "",
        "Implemented two resumable search sessions with conservative time allocation. "
        "The existing detector and C++ search engines are unchanged; the new Python scheduler compares "
        "their witnessed weights and allocates time between them.",
        "",
        "The new policy retains the strong Board700/Regression690 structural results and matches "
        "guided on Board682 in this grid. Individual seeds still favor the previous dispatcher. "
        "This supports the conservative allocation policy, with the limitations below.",
        "",
        "## Matched results",
        "",
        "All entries are medians of three fresh seeds, in witnessed upper-bound weight; lower is better. "
        "Two-second and ten-second grids use different seeds and are independent runs. "
        "Each code budget splits equally between X and Z. Every configuration runs serially on CPU0 "
        "with one search thread. Preparation, common initialization, detection, setup, search, "
        "and witness delivery are included; imports, loading and independent validation/persistence are excluded.",
    ]
    for grid, title in [
        ("screen-2s", "Two seconds/code, including relabeling controls"),
        ("original-10s", "Ten seconds/code, original corpus"),
    ]:
        summary = comparison["grids"][grid]
        lines += [
            "",
            "### " + title,
            "",
            "| Code/layout | Guided | Previous dispatcher | Competing sessions |",
            "|---|---:|---:|---:|",
        ]
        for case in dict.fromkeys(c["case"] for c in summary["cells"]):
            cells = {c["method"]: c for c in summary["cells"] if c["case"] == case}
            lines.append(
                "| " + case + " | " + " | ".join(str(cells[m]["median"]) for m in ("guided", "dispatch", "race")) + " |"
            )
        lines += ["", f"[{grid} report]({grid}/REPORT.md) and [per-seed results]({grid}/summary.json)."]
    lines += [
        "",
        "### Paired outcomes across both budgets",
        "",
        "| Control | New policy lower | Equal | Higher |",
        "|---|---:|---:|---:|",
    ]
    for control, counts in decisions["paired"].items():
        lines.append(f"| {control} | {counts['lower']} | {counts['equal']} | {counts['higher']} |")
    unsupported = decisions["unsupported"]
    ties = sum(r["race"] == r["guided"] for r in unsupported)
    lines += [
        "",
        f"{ties}/{len(unsupported)} matched outcomes with rejected partitions tie guided. "
        "All higher-weight outcomes are listed explicitly in [decision-summary.json](decision-summary.json). "
        "These paired layouts/seeds are not independent code families or a statistical guarantee.",
        "",
        "The previous study used different seeds: its Board682 median gap cannot be directly compared "
        "with this grid as a causal before/after result. At ten seconds here, both dispatchers and guided "
        "have median 83; the new policy matches guided in each seed and avoids one worse old-dispatcher outcome. "
        "At two seconds it improves Board682 medians over the old dispatcher on original and column-shuffled layouts.",
        "",
        "## Allocation and interpretation",
        "",
        "Each accepted partition gets guided and routed pilots, each min(100 ms, 10% of remaining time), "
        "including setup. The route receives the larger share only when it improves common initialization "
        "and is at least 10% lighter than guided’s effective best. Otherwise guided receives the larger share. "
        "Every epoch of at most 250 ms gives the unfavored engine 20% first, then the favored engine the remainder. "
        "Both sessions retain RNG/population/prepared-kernel state and batch counters across slices.",
        "",
        "| Original code, ten-second grid | Median guided seconds/sector | Median route s/sector | Switches |",
        "|---|---:|---:|---:|",
    ]
    allocations = comparison["grids"]["original-10s"]["allocation"]
    for case in ("board-700-222-28", "board-682-172-79", "regression-690-182"):
        rows = [r for r in allocations if r["case"] == case]
        g, r = [statistics.median(row["active_seconds"][name] for row in rows) for name in ("guided", "route")]
        lines.append(f"| {case} | {g:.3f} | {r:.3f} | {sum(row['switches'] for row in rows)} |")
    switches = sum(row["switches"] for row in decisions["allocation"])
    lines += [
        "",
        f"There are {switches} preference switches after the pilots across the full study. "
        "Repeated reassessment is implemented and tested synthetically, but these runs do not establish "
        "an advantage from switching during search. The demonstrated gain is choosing a more appropriate "
        "allocation from the pilot results. An initial-decision-only ablation would be needed to isolate "
        "the value of later reassessment.",
        "",
        "The 10% margin and 80/20 split are fixed hypotheses motivated by the prior allocation failure, "
        "not tuned on this run. The corpus and relabelings are reused known inputs. No target weights, "
        "code names, family metadata or reference supports enter search. Row shuffling still defeats "
        "structure recovery. No new best hard reference bound is found: the references remain 28, 72, "
        "and 28 for Board700, Board682, and Regression690.",
        "",
        f"Maximum measured detection time is {1000 * decisions['max_detection_seconds']:.3f} ms; "
        f"maximum detector budget overrun is {1000 * decisions['max_detection_overrun_seconds']:.3f} ms. "
        "Both search sessions remain resident during competition, increasing memory relative to "
        "sequential sessions. Total process RSS was not measured. Native setup and batches remain "
        "noninterruptible; slice splits and deadlines are cooperative targets.",
        "",
        "## Validation and reproduction",
        "",
        f"Completed {comparison['configurations']} configurations and audited "
        f"{comparison['saved_documents']:,} saved witness documents. All 46 unit/regression tests passed "
        "(17 new and 29 existing). Each returned corpus witness was retained through the existing "
        "validate_and_stage → make_submission/save_submission path. Complete-grid, common-initialization, "
        "source/binary/archive, actual matrix-transform, partition, decision, engine-export, and saved-document "
        "checks passed. Only the existing large-fixture size-cap exemption applies.",
        "",
        f"The grids allocate 864 search seconds. Timed wall time was {comparison['timed_elapsed_seconds']:.2f} seconds "
        f"and CPU time {comparison['timed_cpu_seconds']:.2f} seconds. All {comparison['late_exports']} late exports "
        "were saved without deadline credit. Timings exclude independent validation and persistence.",
        "",
        "[Implementation and reproduction](IMPLEMENTATION.md), "
        "[frozen policy/grid](../../strategy_prototypes/ALLOCATION_PLAN.md), "
        "[search API](../../allocation_search.py), [tests](tests.txt), [lint](lint.txt), "
        "[hardware](hardware.json), [packages](environment-freeze.txt), "
        "[validation index](validation-summary.json), and [artifact checksums](SHA256SUMS.json).",
        "",
        "The source archives and binary hashes match across budgets, and common initialization is identical. "
        "The trusted verifier still matches its 27-file pin. No verifier, leaderboard, CI or publication changes. "
        "These are experimental witnessed upper bounds, not exact-distance certificates or full candidate-gate passes.",
        "",
        "Keep this as the leading allocation candidate for broader evaluation. The useful next comparison "
        "would test unseen code families/layouts and longer budgets; this study does not justify more tuning "
        "of the margin on the same small corpus.",
    ]
    (BASE / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
