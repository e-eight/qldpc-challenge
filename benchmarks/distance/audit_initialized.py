"""Audit initialized studies and the separate finite orbit diagnostic."""

import argparse
import json
from pathlib import Path

from audit_strategies import main as audit_grid
from common import ROOT, atomic_json, matrix_hash, sha256
from run import np


def audit_orbits(directory, stage="orbit"):
    env = json.loads((directory / "environment.json").read_text())
    for group in ("source_hashes", "binary_hashes"):
        for relative, expected in env[group].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Changed measured artifact: {relative}")
    records = json.loads((directory / "results.json").read_text())
    expected = {
        (case, variant)
        for case in ("board-700-222-28", "board-682-172-79", "regression-690-182")
        for variant in ("original", "relabeled")
    }
    if len(records) != len(expected) or {(r["source_case"], r["variant"]) for r in records} != expected:
        raise ValueError("Incomplete orbit diagnostic")
    total = 0
    originals = {}
    for record in records:
        folder = directory / record["case"]
        if json.loads((folder / "result.json").read_text()) != record or record["validation_status"] != "passed":
            raise ValueError("Changed/unvalidated record")
        metadata = json.loads((folder / "input.json").read_text())
        with np.load(folder / "matrices.npz", allow_pickle=False) as data:
            hx, hz = data["hx"], data["hz"]
        if matrix_hash(hx, hz) != metadata["matrix_sha256"]:
            raise ValueError("Changed matrix")
        if record["variant"] == "original":
            originals[record["source_case"]] = (hx, hz)
        else:
            ox, oz = originals[record["source_case"]]
            permutation = metadata["permutation"]
            if sorted(permutation) != list(range(hx.shape[1])):
                raise ValueError("Invalid relabeling permutation")
            if not np.array_equal(hx, ox[:, permutation]) or not np.array_equal(hz, oz[:, permutation]):
                raise ValueError("Relabeled matrices mismatch")
        sequence = 0
        orbit_weights = []
        saved = ROOT / "research/candidates/distance-benchmark" / record["case"] / directory.name
        for side in ("X", "Z"):
            events = [json.loads(line) for line in (folder / f"{side}.jsonl").read_text().splitlines()]
            worker = record["workers"][side][0]
            if events != worker["events"]:
                raise ValueError("Raw witness log mismatch")
            if sum(e["stage"] == stage for e in events) != worker["counters"]["logical_witnesses"]:
                raise ValueError("Orbit emission count mismatch")
            seen = set()
            for event in events:
                if event["stage"] == stage:
                    orbit_weights.append(event["weight"])
                support = tuple(event["support"])
                if support in seen:
                    continue
                seen.add(support)
                files = list((saved / str(sequence)).glob("*.json"))
                if len(files) != 1:
                    raise ValueError("Missing/ambiguous saved witness")
                document = json.loads(files[0].read_text())
                witness = document["distance"][side]
                if (
                    witness["witness"] != sorted(support)
                    or witness["value"] != event["weight"]
                    or witness["confidence"] != "upper_bound"
                ):
                    raise ValueError("Saved support differs from original emission")
                sequence += 1
        if min(orbit_weights, default=None) != record[f"{stage}_best"] or sequence != record["saved_candidates"]:
            raise ValueError("Incorrect summary or saved count")
        total += sequence
    atomic_json(
        directory / "audit.json",
        {
            "status": "passed",
            "configurations": len(records),
            "saved_witness_documents_checked": total,
            "source_and_binary_hashes_match": True,
            "event_logs_match": True,
            "matrices_and_relabelings_match": True,
            "scope": "Persistence/provenance audit; trusted algebra checked by runner; no full gate",
        },
    )


def main(directory):
    for child in ("screen-10s", "hard-60s"):
        audit_grid(directory / child)
    audit_orbits(directory / "orbit-completion")
    audit_orbits(directory / "polynomial-completion", "polynomial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
