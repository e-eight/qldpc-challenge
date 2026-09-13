"""Validate archived checkpoint supports with the unchanged trusted GF(2) code.

Needs NumPy, but no compiled search engine, run cache or staged submissions.
This verifies witnesses and tabulation, not original wall-clock measurements.
"""

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "verify"))
import gf2
from qldpc_verify import _matrix

HERE = Path(__file__).resolve().parent / "evidence"


def read(path):
    return json.loads(path.read_text())


def packed(matrix):
    return [sum(1 << int(q) for q in row.nonzero()[0]) for row in matrix]


def main():
    data = HERE / "full-suite"
    manifest = read(data / "manifest.json")["cases"]
    records = [json.loads(line) for line in (data / "witnesses.jsonl").read_text().splitlines()]
    indexed = {(r["case"], r["method"], r["side"]): r for r in records}
    methods = ("cpp", "fresh", "incremental", "guided", "m4ri")
    expected = {(c["id"], m, s) for c in manifest for m in (*methods, "circulant") for s in ("X", "Z")}
    assert len(records) == len(indexed) and set(indexed) == expected
    per_case = {c["case"]: c for c in read(data / "per-case.json")}
    counts, checked = {}, 0
    for case in manifest:
        path = ROOT / case["source"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["source_sha256"]
        doc = read(path)
        hx, hz = (_matrix(doc["checks"][s], doc["n"]) for s in ("X", "Z"))
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            checks = packed(opposite)
            reduced, pivots = gf2.rref(own)
            rows = list(zip(pivots, packed(reduced)))
            for method in (*methods, "circulant"):
                r = indexed[case["id"], method, side]
                assert set(r["checkpoints"]) == ({"2"} if method == "circulant" else {"30", "60"})
                for budget, event in r["checkpoints"].items():
                    if event is not None:
                        assert 0 <= event["seconds"] <= int(budget) / 2
                for event in [*r["checkpoints"].values(), r["initialization"]]:
                    if event is None:
                        continue
                    support = event["support"]
                    assert support == sorted(set(support)) and len(support) == event["weight"] > 0
                    assert 0 <= min(support) <= max(support) < doc["n"]
                    word = sum(1 << q for q in support)
                    assert not any((word & row).bit_count() % 2 for row in checks)
                    for pivot, row in rows:
                        if word >> pivot & 1:
                            word ^= row
                    assert word, "Stabilizer is not a logical witness"
                    checked += 1
        c = per_case[case["id"]]
        assert c["target"] == case["paper_target"]
        for method in (*methods, "circulant"):
            for budget in ["2"] if method == "circulant" else ["30", "60"]:
                events = [indexed[case["id"], method, side]["checkpoints"][budget] for side in ("X", "Z")]
                weight = min((e["weight"] for e in events if e is not None), default=None)
                assert weight == (c["circulant"] if method == "circulant" else c["bounds"][budget][method])
                if method != "circulant":
                    counter = counts.setdefault((int(budget), method), Counter())
                    counter[
                        "below_claim"
                        if weight < c["target"]
                        else "matches_claim"
                        if weight == c["target"]
                        else "above_claim"
                    ] += 1
    for row in read(data / "summary.json")["cells"]:
        if row["family"] == "ALL":
            for key, value in counts[row["budget"], row["method"]].items():
                assert row[key] == value
    print(json.dumps(dict(status="passed", cases=len(manifest), records=len(records), witness_checks=checked)))


if __name__ == "__main__":
    main()
