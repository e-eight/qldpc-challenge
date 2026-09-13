"""Evaluate the frozen continuation criteria from audited, complete grids."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    grids = {}
    saved = 0
    environments = []
    for name in ("screen-2s", "confirmation-20s"):
        folder = ROOT / name
        audit = read(folder / "audit-large-structure.json")
        assert audit["status"] == "passed" and audit["configurations"] == 27
        saved += audit["saved_witness_documents_checked"]
        grids[name] = {(c["case"], c["method"]): c for c in read(folder / "summary.json")["cells"]}
        environments.append(read(folder / "environment.json"))
    for key in ("source_hashes", "binary_hashes", "cases", "methods"):
        assert environments[0][key] == environments[1][key], key
    cases = []
    for name in environments[0]["cases"]:
        short = grids["screen-2s"][name, "structure"]
        long_controls = {m: grids["confirmation-20s"][name, m] for m in ("guided", "race")}
        target_recovery = all(w is not None and w <= short["target"] for w in short["structured_only"])
        target_advantage = target_recovery and all(c["hits"] <= 1 for c in long_controls.values())
        weight_advantage = all(
            w is not None
            and all(w <= 0.8 * grids["screen-2s"][name, method]["weights"][i] for method in ("guided", "race"))
            for i, w in enumerate(short["structured_only"])
        )
        cases.append(
            dict(
                case=name,
                target=short["target"],
                short_structured_weights=short["structured_only"],
                long_control_weights={m: c["weights"] for m, c in long_controls.items()},
                long_control_hits={m: c["hits"] for m, c in long_controls.items()},
                target_advantage=target_advantage,
                twenty_percent_advantage=weight_advantage,
            )
        )
    result = dict(
        status="complete",
        go=any(c["target_advantage"] for c in cases) or sum(c["twenty_percent_advantage"] for c in cases) >= 2,
        cases=cases,
        configurations=54,
        allocated_search_seconds=594,
        saved_witness_documents_checked=saved,
        sources_and_binaries_unchanged_between_grids=True,
        scope="Frozen screening decision; three seeds do not establish universal reliability or an exact distance",
    )
    (ROOT / "decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
