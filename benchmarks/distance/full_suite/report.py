"""Independent final audit and per-code, per-family integration comparison."""

import collections
import hashlib
import json
from pathlib import Path

from run import gf2, np

# isort: split
from audit_candidates import INITIAL, archive_hashes, require
from common import ROOT, atomic_json, matrix_hash, sha256
from initialized_search import pack_rows
from study_strategies import summarize


def best(record, budget):
    values = [s["best_in_budget"] for s in record["checkpoints"][str(budget)].values()]
    return min((w for w in values if w is not None), default=None)


def finalize(output):
    from full_suite.study import checked_pins

    env = checked_pins(output)
    require(archive_hashes(output / "sources.tar.gz") == env["source_hashes"], "Source snapshot differs")
    require(archive_hashes(output / "binaries.tar.gz") == env["binary_hashes"], "Binary snapshot differs")
    cases = json.loads((output / "manifest.json").read_text())["cases"]
    records, counts, initials = [], collections.Counter(), {}
    for case in cases:
        source = output / "inputs" / Path(case["source"]).name
        require(sha256(source) == case["source_sha256"], "Frozen submission differs")
        with np.load(output / "matrices" / case["file"], allow_pickle=False) as data:
            hx, hz = data["hx"], data["hz"]
        require(matrix_hash(hx, hz) == case["matrix_sha256"], "Matrix hash differs")
        checks, rowspaces = {}, {}
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            checks[side] = pack_rows(opposite)
            reduced, pivots = gf2.rref(own)
            rowspaces[side] = list(zip(pivots, pack_rows(reduced)))
        for method in env["methods"] + ["circulant"]:
            marker = output / "runs" / case["id"] / method / "completed.json"
            require(marker.exists(), "Incomplete grid")
            meta = json.loads(marker.read_text())
            require(sha256(output / meta["result"]) == meta["sha256"], "Result hash differs")
            r = json.loads((output / meta["result"]).read_text())
            require(r["case"] == case["id"] and r["method"] == method and r["seed"] == env["seed"], "Job identity")
            require(r["validation_status"] == "passed" and not r["errors"], "Invalid job")
            require(r["budget_seconds"] == (2 if method == "circulant" else env["seconds_per_code"]), "Budget differs")
            require(r["cpu"] in env["cpus"], "Unexpected worker CPU")
            sequence = 0
            for side in ("X", "Z"):
                worker = r["workers"][side][0]
                path = output / r["directory"] / (side + ".jsonl")
                es = [json.loads(line) for line in path.read_text().splitlines()]
                require(es == worker["events"], "Raw log differs")
                require(worker["error"] is None, "Search error")
                times = [e["seconds"] for e in es]
                require(
                    times == sorted(times) and all(0 <= t <= worker["search_seconds"] for t in times), "Event times"
                )
                initial = [{k: e[k] for k in ("stage", "support", "weight")} for e in es if e["stage"] in INITIAL]
                if method != "circulant":
                    require(worker["counters"]["initialization"]["complete"], "Initialization incomplete")
                    key = case["id"], side
                    require(key not in initials or initial == initials[key], "Common initialization differs")
                    initials[key] = initial
                search = [e["weight"] for e in es if e["stage"] == "search"]
                require(search == sorted(set(search), reverse=True), "Non-improving engine exports")
                require(worker["counters"].get("exported", 0) == len(search), "Export accounting")
                seen = set()
                for e in es:
                    support = e["support"]
                    require(
                        support == sorted(set(support)) and len(support) == e["weight"] and support, "Support/weight"
                    )
                    require(min(support) >= 0 and max(support) < case["n"], "Coordinate range")
                    word = sum(1 << q for q in support)
                    require(not any((word & h).bit_count() & 1 for h in checks[side]), "Nonzero syndrome")
                    residual = word
                    for pivot, row in rowspaces[side]:
                        if residual >> pivot & 1:
                            residual ^= row
                    require(residual, "Trivial witness")
                    counts["events"] += 1
                    if tuple(support) in seen:
                        continue
                    seen.add(tuple(support))
                    folder = ROOT / "research/candidates/distance-benchmark" / case["id"] / r["run_id"] / str(sequence)
                    files = list(folder.glob("*.json"))
                    require(len(files) == 1, "Missing/ambiguous kit document")
                    document = json.loads(files[0].read_text())
                    witness = document["distance"][side]
                    require(
                        witness["witness"] == support
                        and witness["value"] == e["weight"]
                        and witness["confidence"] == "upper_bound",
                        "Saved witness differs",
                    )
                    sequence += 1
                for budget, summaries in r["checkpoints"].items():
                    require(summaries[side] == summarize(es, int(budget) / 2, r["target"]), "Checkpoint differs")
                    counts["late_events_" + budget] += sum(e["seconds"] > int(budget) / 2 for e in es)
            require(sequence == r["saved_candidates"], "Saved count differs")
            counts["saved_documents"] += sequence
            records.append(r)
    methods = env["methods"]
    indexed = {(r["case"], r["method"]): r for r in records}
    summary = []
    for budget in env["checkpoints"]:
        for family in ["ALL"] + sorted({c["family"] for c in cases}):
            subset = [c for c in cases if family == "ALL" or c["family"] == family]
            for method in methods:
                values = collections.Counter()
                for c in subset:
                    r = indexed[c["id"], method]
                    w = best(r, budget)
                    baseline = best(indexed[c["id"], "cpp"], budget)
                    require(w is not None and baseline is not None, "Missing main witness")
                    values[
                        "below_claim" if w < r["target"] else "matches_claim" if w == r["target"] else "above_claim"
                    ] += 1
                    values["beats_cpp" if w < baseline else "ties_cpp" if w == baseline else "loses_cpp"] += 1
                    init = min(
                        e["weight"] for s in ("X", "Z") for e in r["workers"][s][0]["events"] if e["stage"] in INITIAL
                    )
                    values["improves_initialization"] += w < init
                summary.append(dict(budget=budget, family=family, method=method, cases=len(subset), **values))
    per_case = []
    for c in cases:
        per_case.append(
            dict(
                case=c["id"],
                n=c["n"],
                k=c["k"],
                family=c["family"],
                target=c["paper_target"],
                bounds={str(b): {m: best(indexed[c["id"], m], b) for m in methods} for b in env["checkpoints"]},
                circulant=best(indexed[c["id"], "circulant"], 2),
            )
        )
    lines = [
        "# Full submitted-code search comparison",
        "",
        f"{len(cases)} frozen codes, five general methods, one seed, four concurrent single-thread jobs.",
        "",
        "All numbers are witnessed upper bounds. Matching a submitted claim is not an exact-distance proof. "
        "A smaller witness is independently validated here, but no new-candidate gate or publication was run.",
        "",
        "| Budget/code | Method | Below claim | Matches claim | Above claim | Beats / ties / loses to old C++ |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for s in summary:
        if s["family"] == "ALL":
            lines.append(
                f"| {s['budget']} | {s['method']} | {s.get('below_claim', 0)} | {s.get('matches_claim', 0)} | "
                f"{s.get('above_claim', 0)} | {s.get('beats_cpp', 0)} / "
                f"{s.get('ties_cpp', 0)} / {s.get('loses_cpp', 0)} |"
            )
    lines += [
        "",
        "Preparation and common initialization count against the budget. Checkpoints use the first "
        "15 or 30 seconds of EACH sector, giving balanced 30/60-second code budgets. Methods start cold; "
        "reference witnesses never enter search. Initialization attribution and family breakdowns are in summary.json.",
        "",
        "dist-m4ri uses a pinned observer-only build for strict-improvement receipt times. The observer does not "
        "change RNG or search decisions; its I/O overhead is included. Original final tied codewords and CLI logs "
        "are retained. Fixed-work observer/control tests are recorded in tests.txt.",
        "",
        "The existing circulant pass is a separate two-second/code diagnostic. Its bounds appear in per-case.json "
        "and do not enter the general-method table. A miss or an inapplicable layout is not evidence of distance.",
        "",
        "Four workers share the VM and last-level cache; this is a throughput comparison, not isolated-core timing. "
        "Only one seed was used, so close wins should be repeated before choosing defaults.",
        "",
        f"Audit passed: {counts['events']} raw events and {counts['saved_documents']} kit-saved documents checked. "
        "Source/binary/input hashes, all raw logs, common initialization and both budget summaries were verified.",
    ]
    atomic_json(output / "summary.json", dict(cells=summary))
    atomic_json(output / "per-case.json", per_case)
    atomic_json(output / "audit.json", dict(status="passed", cases=len(cases), configurations=len(records), **counts))
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        output / "completed.json", dict(status="complete", configurations=len(records), cases=len(cases), **counts)
    )


def freeze(output):
    lines = []
    for p in sorted(output.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name in ("SHA256SUMS", "run.log") or p.suffix == ".tmp":
            continue
        with p.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{digest}  {p.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
