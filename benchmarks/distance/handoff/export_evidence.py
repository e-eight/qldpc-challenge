"""Extract checkpoint witnesses from an already completed full-suite run; no search."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    assert read(source / "audit.json")["status"] == "passed"
    frozen = dict(line.split("  ", 1)[::-1] for line in (source / "SHA256SUMS").read_text().splitlines())
    manifest = read(source / "manifest.json")
    methods = read(source / "environment.json")["methods"]
    records = []
    for case in manifest["cases"]:
        for method in methods + ["circulant"]:
            marker = source / "runs" / case["id"] / method / "completed.json"
            assert digest(marker) == frozen[str(marker.relative_to(source))]
            meta = read(marker)
            path = source / meta["result"]
            assert digest(path) == meta["sha256"] == frozen[meta["result"]]
            result = read(path)
            assert result["validation_status"] == "passed" and not result["errors"]
            for side in ("X", "Z"):
                events = result["workers"][side][0]["events"]
                checkpoints = {}
                for budget in [2] if method == "circulant" else [30, 60]:
                    timely = [e for e in events if e["seconds"] <= budget / 2]
                    best = min(timely, key=lambda e: e["weight"], default=None)
                    checkpoints[str(budget)] = best
                    assert (best["weight"] if best else None) == result["checkpoints"][str(budget)][side][
                        "best_in_budget"
                    ]
                initial = [e for e in events if e["stage"] in ("initialization", "initial_seed")]
                records.append(
                    dict(
                        case=case["id"],
                        method=method,
                        side=side,
                        checkpoints=checkpoints,
                        initialization=min(initial, key=lambda e: e["weight"], default=None),
                        original_result=meta["result"],
                        original_result_sha256=meta["sha256"],
                    )
                )
    with (destination / "witnesses.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    for name in (
        "manifest.json",
        "per-case.json",
        "summary.json",
        "audit.json",
        "completed.json",
        "environment.json",
        "hardware.txt",
        "tests.txt",
        "validator-integrity.txt",
    ):
        assert digest(source / name) == frozen[name]
        shutil.copyfile(source / name, destination / name)
    print(json.dumps(dict(records=len(records), destination=str(destination))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    main(args.source, args.destination)
