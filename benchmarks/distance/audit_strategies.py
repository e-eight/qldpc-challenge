"""Audit saved strategy witnesses against the original emitted supports."""

import argparse
import importlib.metadata
import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from common import HERE, ROOT, atomic_json, matrix_hash, sha256
from report_strategies import checked_records


def check_saved(job):
    record, output_name = job
    directory = (
        ROOT
        / "research"
        / "candidates"
        / "distance-benchmark"
        / record["case"]
        / f"{output_name}-{record['method']}-s{record['seed']}"
    )
    sequence = 0
    for side in ("X", "Z"):
        seen = set()
        for worker in record["workers"][side]:
            for event in worker["events"]:
                support = tuple(event["support"])
                if support in seen:
                    continue
                seen.add(support)
                files = list((directory / str(sequence)).glob("*.json"))
                if len(files) != 1:
                    raise ValueError(f"Missing or ambiguous saved witness: {directory / str(sequence)}")
                document = json.loads(files[0].read_text())
                witness = document["distance"][side]
                if witness["witness"] != sorted(support) or witness["value"] != event["weight"]:
                    raise ValueError(f"Saved witness does not match emission: {files[0]}")
                if witness["confidence"] != "upper_bound":
                    raise ValueError("Unexpected exact-distance claim")
                sequence += 1
    if sequence != record["saved_candidates"]:
        raise ValueError("Saved candidate count mismatch")
    return sequence


def main(directory):
    env, records = checked_records(directory)
    for relative, expected in env["binary_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Measured binary changed: {relative}")
    for relative, expected in env["source_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Measured source changed: {relative}")
    manifest = json.loads((directory / "manifest.json").read_text())
    for case in manifest["cases"]:
        with np.load(directory / "matrices" / case["file"]) as data:
            if matrix_hash(data["hx"], data["hz"]) != case["matrix_sha256"]:
                raise ValueError("Archived matrix hash mismatch")
    with ThreadPoolExecutor(4) as pool:
        counts = list(pool.map(check_saved, [(r, directory.name) for r in records]))
    # Supplement the original source snapshot without rewriting it. This includes
    # the trusted source included by the existing circulant wrapper, and exporters.
    extras = [
        ROOT / "verify" / "gf2_fast.cpp",
        HERE / "setup_native.py",
        HERE / "report_strategies.py",
        Path(__file__),
        HERE / "test_strategies.py",
        HERE / "strategy_prototypes" / "PLAN.md",
        HERE / "strategy_prototypes" / "README.md",
        ROOT / "native" / "ris" / "LICENSE",
        HERE / "strategy_prototypes" / "guided" / "LICENSE",
        ROOT / "schema" / "code.schema.json",
        HERE / "requirements.txt",
        HERE / "inspect_prepared_witnesses.py",
        HERE / "replay_descent_seeds.py",
        HERE / "study_logical_initialization.py",
    ]
    extras.extend((ROOT / "verify").glob("*.py"))
    extras.extend((ROOT / "research" / "kit").glob("*.py"))
    extras = sorted(set(extras))
    with tarfile.open(directory / "supplemental-sources.tar.gz", "w:gz") as archive:
        for path in extras:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    atomic_json(
        directory / "audit.json",
        {
            "status": "passed",
            "configurations": len(records),
            "saved_witness_documents_checked": sum(counts),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "package_versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "pybind11", "setuptools", "pytest")
            },
            "source_and_binary_hashes_match": True,
            "matrix_hashes_match": True,
            "event_logs_and_deadline_summaries_match": True,
            "supplemental_source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in extras},
            "scope": "Witness persistence audit; trusted algebra checked by runner; no full candidate gate",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
