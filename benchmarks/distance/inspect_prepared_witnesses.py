"""Persist and score deterministic logical-basis witnesses already built at setup."""

import argparse
import json
import sys
import time
from pathlib import Path

from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from run import np, validate_and_stage
from study_strategies import CASES

sys.path.insert(0, str(ROOT / "native" / "ris"))
import ris_native


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    corpus = HERE / "results" / "reference-corpus"
    manifest = json.loads((corpus / "manifest.json").read_text())
    records = []
    for case in manifest["cases"]:
        if case["id"] not in CASES:
            continue
        with np.load(corpus / case["file"]) as data:
            hx, hz = data["hx"], data["hz"]
        assert matrix_hash(hx, hz) == case["matrix_sha256"]
        start = time.perf_counter()
        # Prepared.logicals are duals used to test nontriviality. Reversing the
        # matrices gives valid candidate logicals for the requested side.
        candidates = {
            "X": ris_native.Prepared(hz, hx).logicals,
            "Z": ris_native.Prepared(hx, hz).logicals,
        }
        preparation = time.perf_counter() - start
        fallback = {side: np.flatnonzero(rows[0]).tolist() for side, rows in candidates.items()}
        sides = {}
        start = time.perf_counter()
        for side, rows in candidates.items():
            best = case["n"] + 1
            events = []
            for row_index, row in enumerate(rows):
                weight = int(row.sum())
                if weight < best:
                    best = weight
                    events.append(
                        {
                            "seconds": time.perf_counter() - start,
                            "weight": weight,
                            "support": np.flatnonzero(row).tolist(),
                            "basis_row": row_index,
                        }
                    )
            sides[side] = [{"events": events}]
        scan = time.perf_counter() - start
        record = {
            "case": case["id"],
            "matrix_sha256": case["matrix_sha256"],
            "target": code_target(case),
            "workers": sides,
            "preparation_seconds": preparation,
            "scan_seconds": scan,
            "best": min(e["weight"] for ws in sides.values() for w in ws for e in w["events"]),
            "validation_status": "pending",
        }
        atomic_json(output / f"{case['id']}.json", record)
        validation, saved = validate_and_stage(hx, hz, case, sides, fallback, output.name)
        record.update(validation_status="passed", validation_seconds=validation, saved_candidates=saved)
        atomic_json(output / f"{case['id']}.json", record)
        records.append(record)
    atomic_json(output / "results.json", records)
    atomic_json(
        output / "metadata.json",
        {
            "source_sha256": sha256(Path(__file__)),
            "native_binary_sha256": sha256(ris_native.__file__),
            "scope": "Post-screen deterministic setup diagnostic; no reference supports and no random search",
            "configurations": len(records),
            "saved_candidates": sum(r["saved_candidates"] for r in records),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
