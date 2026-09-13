"""Preserve raw exports from interrupted attempts without crediting incomplete runs."""

import json
import shutil
import time
from pathlib import Path

from run import np, validate_and_stage
from common import atomic_json, matrix_hash

ROOT = Path(__file__).resolve().parent


def main():
    recovery = ROOT / "interruption-20260913"
    recovery.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / "progress.json", recovery / "last-progress.json")
    cases = {c["id"]: c for c in json.loads((ROOT / "manifest.json").read_text())["cases"]}
    reports = []
    for base in sorted((ROOT / "runs").glob("*/*")):
        if (base / "completed.json").exists():
            continue
        for attempt in sorted(base.glob("attempt-*")):
            sides = {}
            for side in ("X", "Z"):
                raw = attempt / (side + ".jsonl")
                events = []
                if raw.exists():
                    for line in raw.read_text().splitlines():
                        events.append(json.loads(line))
                sides[side] = [dict(events=events)]
            if not any(w[0]["events"] for w in sides.values()):
                continue
            case = cases[base.parent.name]
            with np.load(ROOT / "matrices" / case["file"], allow_pickle=False) as data:
                hx, hz = data["hx"], data["hz"]
            if matrix_hash(hx, hz) != case["matrix_sha256"]:
                raise ValueError("Frozen input changed")
            run_id = f"{ROOT.name}-recovery-20260913-{base.name}-{attempt.name}"
            record = dict(case=case["id"], method=base.name, attempt=str(attempt.relative_to(ROOT)),
                          status="interrupted_excluded_from_comparison", validation_status="pending", run_id=run_id)
            atomic_json(attempt / "recovery.json", record)
            seconds, saved = validate_and_stage(hx, hz, case, sides, case["reference"], run_id)
            record.update(validation_status="passed", saved_candidates=saved, validation_seconds=seconds)
            atomic_json(attempt / "recovery.json", record)
            reports.append(record)
    atomic_json(recovery / "recovery.json", dict(status="passed", recovered_unix=time.time(), attempts=reports,
                saved_documents=sum(r["saved_candidates"] for r in reports)))
    print(json.dumps(dict(attempts=len(reports), saved_documents=sum(r["saved_candidates"] for r in reports))))


if __name__ == "__main__":
    main()
