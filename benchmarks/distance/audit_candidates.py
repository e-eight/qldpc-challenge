"""Audit a complete candidate grid; no search or candidate generation is run.

Parent event timestamps determine deadline credit. Backend applicability flags
are bookkeeping, not proofs of useful matrix structure. This extends the shared
witness-identity audit; trusted algebra was checked by the persisted runner.
"""

import argparse
import hashlib
import json
import math
import os
import tarfile
from pathlib import Path

from run import np

# isort: split
# run fixes numerical thread settings before numerical dependencies are imported.
from audit_strategies import main as audit_standard
from common import ROOT, atomic_json, matrix_hash, sha256
from report_strategies import checked_records

INITIAL = {"initialization", "initial_seed"}
REDUCED = {"reduced_both", "reduced_single_left", "reduced_single_right"}
ORBIT = {
    "orbit_own_adjacent_rows",
    "orbit_opposite_adjacent_rows",
    "orbit_supplied_two_block_layout",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archive_hashes(path):
    """Read bytes directly from tar; never extract untrusted archive paths."""
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            require(member.isfile(), f"Unexpected source archive member: {member.name}")
            require(member.name not in result, f"Duplicate source archive member: {member.name}")
            stream = archive.extractfile(member)
            require(stream is not None, "Missing source archive bytes")
            result[member.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def check_archives(directory, expected):
    archived = {}
    for name in ("sources.tar.gz", "candidate-sources.tar.gz"):
        for relative, digest in archive_hashes(directory / name).items():
            require(relative in expected, f"Unrecorded source archive member: {relative}")
            require(digest == expected[relative], f"Archived measured source differs: {relative}")
            require(relative not in archived or archived[relative] == digest, f"Conflicting archives: {relative}")
            archived[relative] = digest
    require(set(archived) == set(expected), "Measured source missing from archives")
    return len(archived)


def read_matrices(directory, case):
    with np.load(directory / case["file"], allow_pickle=False) as data:
        hx, hz = data["hx"], data["hz"]
    require(matrix_hash(hx, hz) == case["matrix_sha256"], f"Corpus matrix hash mismatch: {case['id']}")
    return hx, hz


def permutation(value, length, label):
    result = np.asarray(value)
    require(result.shape == (length,), f"Malformed {label} permutation shape")
    require(np.issubdtype(result.dtype, np.integer), f"Noninteger {label} permutation")
    require(np.array_equal(np.sort(result), np.arange(length)), f"Nonbijective {label} permutation")
    return result


def check_transforms(directory, corpus):
    manifest = json.loads((corpus / "manifest.json").read_text())
    indexed = {case["id"]: case for case in manifest["cases"]}
    transforms = json.loads((corpus / "transforms.json").read_text())
    require(json.loads((directory / "transforms.json").read_text()) == transforms, "Copied transformations differ")
    require(len(indexed) == len(manifest["cases"]), "Duplicate corpus case IDs")
    variants = {key for key, case in indexed.items() if case.get("variant")}
    require(set(transforms) == variants, "Missing or unexpected corpus transformation")
    matrices = {key: read_matrices(corpus, case) for key, case in indexed.items()}
    inverse_references = 0
    for name, transform in transforms.items():
        case = indexed[name]
        source_id = transform["source_case"]
        require(case["source_case"] == source_id, "Transformed source case mismatch")
        source = indexed[source_id]
        hx, hz = matrices[source_id]
        columns = permutation(transform["columns"], hx.shape[1], "column")
        rows_x = permutation(transform["rows_x"], hx.shape[0], "X row")
        rows_z = permutation(transform["rows_z"], hz.shape[0], "Z row")
        actual_x, actual_z = matrices[name]
        require(np.array_equal(actual_x, hx[rows_x][:, columns]), f"X transformation mismatch: {name}")
        require(np.array_equal(actual_z, hz[rows_z][:, columns]), f"Z transformation mismatch: {name}")
        if case["variant"] == "columns":
            require(np.array_equal(rows_x, np.arange(len(hx))), "Column-only control changed X rows")
            require(np.array_equal(rows_z, np.arange(len(hz))), "Column-only control changed Z rows")
        else:
            require(case["variant"] == "columns-rows", "Unexpected transformation variant")
            sibling = transforms[source_id + "-columns"]
            require(sibling["columns"] == transform["columns"], "Row control changed the column permutation")
        inverse = np.argsort(columns)
        for field in ("reference", "previous_reference"):
            require((field in source) == (field in case), f"Reference field mismatch: {name}")
            if field not in source:
                continue
            for side in ("X", "Z"):
                expected = sorted(inverse[source[field][side]].tolist())
                require(case[field][side] == expected, f"Inverse reference mapping mismatch: {name}/{side}/{field}")
                inverse_references += 1
    grid = json.loads((directory / "manifest.json").read_text())
    require(grid["sources"] == manifest["sources"], "Grid source provenance differs from corpus")
    for case in grid["cases"]:
        require(case == indexed[case["id"]], f"Grid case metadata differs: {case['id']}")
        x, z = read_matrices(directory / "matrices", case)
        cx, cz = matrices[case["id"]]
        require(np.array_equal(x, cx) and np.array_equal(z, cz), "Grid matrix differs from actual corpus")
    return dict(
        corpus_matrices=len(indexed), transformations=len(transforms), inverse_reference_supports=inverse_references
    )


def check_initialization(records):
    signatures, counts = {}, {}
    for record in records:
        for side in ("X", "Z"):
            require(len(record["workers"][side]) == 1, "Expected one worker per side")
            worker = record["workers"][side][0]
            counters = worker["counters"]
            initial = [event for event in worker["events"] if event["stage"] in INITIAL]
            require(initial and counters["initialization"]["complete"], "Incomplete common initialization")
            require(min(e["weight"] for e in initial) == counters["initial_best"], "Initialization bound mismatch")
            require(
                all(e["seconds"] <= record["budget_seconds"] / 2 for e in initial),
                "Common initialization was delivered late",
            )
            signature = dict(
                events=[{k: e[k] for k in ("stage", "weight", "support")} for e in initial],
                counters=counters["initialization"],
                best=counters["initial_best"],
            )
            key = record["case"], side
            if key in signatures:
                require(signatures[key] == signature, f"Initialization differs across methods/seeds: {key}")
            else:
                signatures[key] = signature
            counts[key] = counts.get(key, 0) + 1
    return [
        dict(
            case=case,
            side=side,
            workers_compared=counts[case, side],
            best=value["best"],
            signature_sha256=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
        )
        for (case, side), value in sorted(signatures.items())
    ]


def check_exports(record, side):
    worker = record["workers"][side][0]
    counters, method = worker["counters"], record["method"]
    all_events = worker["events"]
    times = [event["seconds"] for event in all_events]
    require(all(math.isfinite(t) and t >= 0 for t in times), "Malformed event time")
    require(times == sorted(times), "Nonmonotonic event times")
    require(not times or times[-1] <= worker["search_seconds"], "Event after reported search exit")
    initial_finished = False
    for event in all_events:
        if event["stage"] not in INITIAL:
            initial_finished = True
        else:
            require(not initial_finished, "Initialization appeared after search events")
    allowed = INITIAL | {"guided"}
    if method in {"incremental", "guided", "circulant"}:
        allowed |= {method}
    elif method == "decoder":
        allowed |= {"decoder"}
    elif method == "matrix-structure":
        allowed |= ORBIT | {"sector_transfer"}
    elif method == "reduced-space":
        allowed |= REDUCED
    else:
        raise ValueError("Unexpected method")
    require(all(event["stage"] in allowed for event in all_events), f"Unexpected stage for {method}")
    if method != "guided" and any(event["stage"] == "guided" for event in all_events):
        require("fallback" in counters, "Unrecorded guided fallback")
    candidate = counters.get("candidate", {})
    events = [event for event in all_events if event["stage"] not in INITIAL | {"guided", "incremental", "circulant"}]
    weights = [event["weight"] for event in events]
    best = min(weights, default=None)
    if method == "decoder":
        require(candidate["distinct_exports"] == len(events), "Decoder export counter mismatch")
        require(candidate["best_weight"] == best, "Decoder best counter mismatch")
        require(
            candidate["trials"] == candidate["successful_decodes"] + candidate["failed_decodes"],
            "Decode count mismatch",
        )
        require(
            candidate["successful_decodes"] == candidate["distinct_exports"] + candidate["duplicate_outputs"],
            "Decoder duplicate count mismatch",
        )
        require(candidate["bp_converged"] <= candidate["trials"], "BP convergence count exceeds trials")
        require(len({tuple(e["support"]) for e in events}) == len(events), "Decoder exported duplicate support")
    elif method == "matrix-structure":
        require(candidate["emitted"] == len(events), "Structure export counter mismatch")
        require(candidate.get("best") == best, "Structure best counter mismatch")
        require(candidate["logical_candidates"] >= len(events), "Structure logical count below exports")
        require(len({tuple(e["support"]) for e in events}) == len(events), "Structure exported duplicate support")
        if any(e["stage"] == "sector_transfer" for e in events):
            require(candidate["sector_maps"] > 0, "Transfer without recorded verified sector map")
    elif method == "reduced-space":
        require(candidate["exported"] == len(events), "Reduced-space export counter mismatch")
        require(candidate.get("best") == best, "Reduced-space best counter mismatch")
        for label, branch in candidate["branches"].items():
            branch_weights = [e["weight"] for e in events if e["stage"] == label]
            require(branch["best"] == min(branch_weights, default=None), "Reduced branch best mismatch")
            require(
                all(a > b for a, b in zip(branch_weights, branch_weights[1:])), "Nonimproving reduced branch export"
            )
        require(all(e["stage"] in candidate["branches"] for e in events), "Unrecorded reduced branch")
    return len(events)


def main(directory, corpus=None):
    directory = Path(directory)
    corpus = Path(corpus) if corpus else directory.parent / "corpus"
    require((directory / "completed.json").is_file(), "Grid has not completed; do not audit an active run")
    env, records = checked_records(directory)
    completed = json.loads((directory / "completed.json").read_text())
    require(completed["configurations"] == len(records), "Completion configuration count mismatch")
    require(
        completed["saved_candidates"] == sum(r["saved_candidates"] for r in records), "Completion save count mismatch"
    )
    require(env["threads"] == 1, "Expected a single search thread")
    require(all(p["num_threads"] == 1 for p in env["threadpools"]), "Recorded numerical thread limit violation")
    archives = check_archives(directory, env["source_hashes"])
    transforms = check_transforms(directory, corpus)
    initialization = check_initialization(records)
    candidate_exports = sum(check_exports(record, side) for record in records for side in ("X", "Z"))
    audit_standard(directory)
    standard = json.loads((directory / "audit.json").read_text())
    supplemental = archive_hashes(directory / "supplemental-sources.tar.gz")
    require(
        supplemental == standard["supplemental_source_hashes"], "Supplemental archive bytes differ from audit hashes"
    )
    source = Path(__file__).resolve()
    with tarfile.open(directory / "candidate-audit-source.tar.gz", "w:gz") as archive:
        archive.add(source, arcname=str(source.relative_to(ROOT)))
    atomic_json(
        directory / "audit-candidates.json",
        dict(
            status="passed",
            configurations=len(records),
            saved_witness_documents_checked=standard["saved_witness_documents_checked"],
            candidate_export_events_checked=candidate_exports,
            measured_archived_sources_checked=archives,
            supplemental_archived_sources_checked=len(supplemental),
            transformations=transforms,
            identical_complete_initialization=initialization,
            source_and_binary_hashes_match=True,
            stage_export_counters_match=True,
            event_logs_and_deadline_summaries_match=True,
            audit_source_sha256=sha256(source),
            limits=[
                "Parent timestamps determine deadline credit; backend late counters may use later callback clocks.",
                "Candidate applicability and recovered-map counters are not structural or optimality proofs.",
                "Trusted witness algebra was checked by the runner; this audit checks identity and persistence.",
                "No full candidate gate or independently replayed exact-distance certificate is claimed.",
                "Live sources/binaries and locally staged witness documents are required by the shared audit.",
            ],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--cpu", type=int, default=7)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    main(args.directory, args.corpus)
