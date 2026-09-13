"""Distribution and overlap analysis of completed, audited observations only."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qldpc-distribution-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "full-suite-60s"


def read(path):
    return json.loads(path.read_text())


def main():
    cases = read(SOURCE / "per-case.json")
    assert read(SOURCE / "audit.json")["status"] == "passed"
    assert read(SOURCE / "completed.json")["cases"] == len(cases) == 593
    below, histogram, overlap, differences = [], Counter(), {}, []
    for case in cases:
        values = case["bounds"]["60"]
        weight = min(values.values())
        target = case["target"]
        histogram["below" if weight < target else "matches" if weight == target else "above"] += 1
        if weight < target:
            marker = read(SOURCE / "runs" / case["case"] / "guided/completed.json")
            r = read(SOURCE / marker["result"])
            initial = min(
                e["weight"]
                for side in ("X", "Z")
                for e in r["workers"][side][0]["events"]
                if e["stage"] in ("initialization", "initial_seed")
            )
            below.append(
                dict(
                    case=case["case"],
                    target=target,
                    best=weight,
                    initial=initial,
                    absolute_reduction=target - weight,
                    percent_reduction=100 * (target - weight) / target,
                    best_methods=[m for m, w in values.items() if w == weight],
                )
            )
    below.sort(key=lambda r: (-r["percent_reduction"], r["case"]))
    for budget in ("30", "60"):
        counts = Counter()
        target_sets = {key: [] for key in ("both", "guided_only", "incremental_only", "neither")}
        for case in cases:
            g, i = (case["bounds"][budget][m] for m in ("guided", "incremental"))
            target = case["target"]
            counts["guided_lighter" if g < i else "incremental_lighter" if i < g else "equal_weight"] += 1
            key = (
                "both"
                if g <= target and i <= target
                else "guided_only"
                if g <= target
                else "incremental_only"
                if i <= target
                else "neither"
            )
            target_sets[key].append(case["case"])
            if budget == "60" and g != i:
                differences.append(
                    dict(
                        case=case["case"],
                        target=target,
                        guided=g,
                        incremental=i,
                        guided_minus_incremental=g - i,
                        family=case["family"],
                    )
                )
        overlap[budget] = dict(
            weight_comparison=dict(counts),
            target_counts={k: len(v) for k, v in target_sets.items()},
            target_sets=target_sets,
        )
    portfolio = dict(targets_recovered=0, versus_guided=Counter(), versus_incremental=Counter())
    for case in cases:
        combined = min(case["bounds"]["30"][m] for m in ("guided", "incremental"))
        portfolio["targets_recovered"] += combined <= case["target"]
        for other in ("guided", "incremental"):
            weight = case["bounds"]["60"][other]
            portfolio["versus_" + other][
                "lower" if combined < weight else "higher" if combined > weight else "equal"
            ] += 1
    misses = Counter()
    for c in cases:
        ratio = min(c["bounds"]["60"].values()) / c["target"]
        if ratio > 1:
            misses[
                "up_to_10_percent"
                if ratio <= 1.1
                else "10_to_25_percent"
                if ratio <= 1.25
                else "25_to_50_percent"
                if ratio <= 1.5
                else "50_to_100_percent"
                if ratio <= 2
                else "over_100_percent"
            ] += 1
    extra = [
        c
        for c in cases
        if c["circulant"] is not None and c["circulant"] <= c["target"] < min(c["bounds"]["60"].values())
    ]
    result = dict(
        scope="593 frozen submissions; five 60-second general searches; one seed; no new searches",
        source_hashes={
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in ("per-case.json", "audit.json", "completed.json")
        },
        distribution=dict(histogram),
        below_submission=below,
        overlap=overlap,
        weight_differences=differences,
        retrospective_30_plus_30=portfolio,
        remaining_gap_bins=dict(misses),
        circulant_only_target_recoveries=[c["case"] for c in extra],
    )
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    # A compact scientific figure: magnitude of tightened bounds and target overlap.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(below))
    colors = ["#3b7bba" if r["initial"] == r["best"] else "#e18b2c" for r in below]
    axes[0].barh(y, [r["percent_reduction"] for r in below], color=colors)
    axes[0].set_yticks(y, [r["case"] for r in below])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 30)
    axes[0].set_xlabel("Reduction from submitted bound (%)")
    axes[0].set_title("10 submissions tightened by general searches")
    for j, r in enumerate(below):
        axes[0].text(r["percent_reduction"] + 0.4, j, f"{r['target']} → {r['best']}", va="center", fontsize=9)
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(color="#3b7bba", label="Best already in initialization"),
            Patch(color="#e18b2c", label="Search improved initialization"),
        ],
        loc="lower right",
        fontsize=8,
    )
    counts = overlap["60"]["target_counts"]
    table = np.array([[counts["both"], counts["guided_only"]], [counts["incremental_only"], counts["neither"]]])
    axes[1].imshow(table, cmap="Blues", vmin=0, vmax=600)
    axes[1].set_xticks([0, 1], ["Recovered", "Missed"])
    axes[1].set_xlabel("Incremental RIS")
    axes[1].set_yticks([0, 1], ["Recovered", "Missed"])
    axes[1].set_ylabel("Guided")
    axes[1].set_title("Submitted-bound recovery at 60 s/code")
    for (a, b), v in np.ndenumerate(table):
        axes[1].text(b, a, str(v), ha="center", va="center", fontsize=23, color="white" if v > 300 else "black")
    fig.suptitle("Full-suite results: magnitude and overlap", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Lower weight tightens an upper bound; no exact-distance or statistical-significance claim. One seed.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(HERE / "distribution-overlap.png", dpi=180)
    fig.savefig(HERE / "distribution-overlap.svg")
    plt.close(fig)
    lines = [
        "# Submitted bounds and guided/incremental overlap",
        "",
        "Read-only analysis of the completed full-suite study. No additional searches or witness generation.",
        "",
        "At 60 seconds per general method, the best of five methods is below the submission on 10 codes, "
        "equal on 557, and above on 26. Four reductions are at least 20%. This is an oracle union of five "
        "runs, costing up to 300 seconds/code, not a 60-second combined method.",
        "",
        "| Submission ID | Actual submitted bound | Best | Reduction | Initialization | Best method(s) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in below:
        lines.append(
            f"| {r['case']} | {r['target']} | {r['best']} | {r['percent_reduction']:.1f}% | {r['initial']} | "
            + ", ".join(r["best_methods"])
            + " |"
        )
    lines += [
        "",
        "Five submissions were already beaten by common initialization; four final winning bounds "
        "are entirely initialization results. On 684-12-77, initialization gives 69 and guided lowers it "
        "to 61. Guided also beats four other submissions, while incremental uniquely beats 360-8-48. "
        "These comparisons use JSON distance claims, not filenames. They are not novelty claims: "
        "for example the earlier structural work already found 54 on 684-8-85, below this sweep's 63.",
        "",
        "At 60 seconds, guided and incremental tie in actual weight on 565 codes. Guided is lighter on "
        "17, incremental on 11. Both recover the submitted bound on 558; guided alone on six; "
        "incremental alone on three; neither on 26. Their union recovers 567, the same coverage as "
        "all five general methods. That union costs 120 seconds/code if both receive 60 seconds.",
        "",
        "A retrospective equal-total-budget portfolio using each method's 30-second checkpoint "
        "recovers 566 targets, versus 564 for guided alone at 60 seconds and 561 for incremental. "
        "Against guided 60 seconds, its weights are lower on seven, equal on 582, and higher on four. "
        "This is calculated from existing checkpoints, not a separately executed portfolio policy; "
        "one seed does not establish reliability or optimal allocation.",
        "",
        "Incremental-only target recoveries: " + ", ".join(overlap["60"]["target_sets"]["incremental_only"]) + ".",
        "",
        "Guided-only target recoveries: " + ", ".join(overlap["60"]["target_sets"]["guided_only"]) + ".",
        "",
        "The separately budgeted circulant pass recovers five additional missed targets (572 total "
        "with the general-method union). It does not tighten any further submitted claim. Four hard "
        "700-qubit weight-28 claims still have general-search witnesses at 71–75; broad scaling "
        "limitations remain despite the favorable recovery count.",
        "",
        "![Magnitude and overlap](distribution-overlap.png)",
        "",
        "All per-code differences, target sets, source hashes and checkpoint-portfolio counts are "
        "in analysis.json. The completed source study and its checksums are unchanged.",
    ]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(distribution=dict(histogram), overlap=overlap["60"]["target_counts"], portfolio=portfolio)))


if __name__ == "__main__":
    main()
