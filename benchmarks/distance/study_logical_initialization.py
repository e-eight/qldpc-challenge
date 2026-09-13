"""Cheap logical-basis combinations, then one native stabilizer-refinement pass."""

import argparse
import json
import sys
import time
from pathlib import Path

from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from run import np, validate_and_stage
from study_strategies import CASES

sys.path.insert(0, str(ROOT / "native" / "ris"))
sys.path.insert(0, str(HERE / "strategy_prototypes" / "descent"))
import ris_native
from _stabilizer_descent import Descent


def support_of(word):
    support = []
    while word:
        bit = word & -word
        support.append(bit.bit_length() - 1)
        word ^= bit
    return support


def combinations(rows):
    """Enumerate a representative span for small k; rows and pairs otherwise."""
    started = time.perf_counter()
    values = [int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little") for row in rows]
    best = rows.shape[1] + 1
    events = []
    examined = 0

    def observe(word):
        nonlocal best, examined
        examined += 1
        weight = word.bit_count()
        if weight < best:
            best = weight
            events.append({"seconds": time.perf_counter() - started, "weight": weight, "support": support_of(word)})

    if len(values) <= 16:
        word = 0
        for index in range(1, 1 << len(values)):
            word ^= values[(index & -index).bit_length() - 1]
            observe(word)
        strategy = "all nonzero combinations of the chosen representatives"
    else:
        for word in values:
            observe(word)
        for i, word in enumerate(values):
            for j in range(i + 1, len(values)):
                observe(word ^ values[j])
        strategy = "single representatives and all pairs"
    return {"events": events, "examined": examined, "strategy": strategy, "seconds": time.perf_counter() - started}


def main(output, seconds, seed):
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
        started = time.perf_counter()
        rows = {"X": ris_native.Prepared(hz, hx).logicals, "Z": ris_native.Prepared(hx, hz).logicals}
        preparation = time.perf_counter() - started
        fallback = {side: np.flatnonzero(basis[0]).tolist() for side, basis in rows.items()}
        combo = {side: [combinations(basis)] for side, basis in rows.items()}
        record = {
            "case": case["id"],
            "target": code_target(case),
            "matrix_sha256": case["matrix_sha256"],
            "basis_best": min(int(row.sum()) for basis in rows.values() for row in basis),
            "preparation_seconds": preparation,
            "combinations": combo,
            "combination_best": min(ws[0]["events"][-1]["weight"] for ws in combo.values()),
            "validation_status": "pending",
        }
        path = output / f"{case['id']}.json"
        atomic_json(path, record)
        _, saved_combo = validate_and_stage(hx, hz, case, combo, fallback, f"{output.name}-combinations")
        polished = {}
        for side, own in (("X", hx), ("Z", hz)):
            initial = combo[side][0]["events"][-1]
            started = time.perf_counter()
            engine = Descent(own)
            result = engine.run(
                initial["support"],
                max(0, seconds / 2 - (time.perf_counter() - started)),
                seed + (499979 if side == "Z" else 0),
            )
            elapsed = time.perf_counter() - started
            events = [{"seconds": 0.0, "weight": initial["weight"], "support": initial["support"]}]
            events.extend(
                {"seconds": elapsed, "weight": weight, "support": support} for weight, support in result["improvements"]
            )
            polished[side] = [
                {
                    "events": events,
                    "seconds": elapsed,
                    "counters": {k: v for k, v in result.items() if k != "improvements"},
                }
            ]
        record["polished"] = polished
        record["polished_best_returned"] = min(ws[0]["events"][-1]["weight"] for ws in polished.values())
        atomic_json(path, record)
        _, saved_polished = validate_and_stage(hx, hz, case, polished, fallback, f"{output.name}-polished")
        record.update(validation_status="passed", saved_candidates=saved_combo + saved_polished)
        atomic_json(path, record)
        records.append(record)
    atomic_json(output / "results.json", records)
    atomic_json(
        output / "metadata.json",
        {
            "source_sha256": sha256(Path(__file__)),
            "seed": seed,
            "polishing_seconds_per_code": seconds,
            "scope": (
                "One-seed initialization diagnostic; polishing uses an uninstrumented full slice "
                "and reports returned bounds"
            ),
            "warning": (
                "Representative-span enumeration is not full quantum-distance enumeration; "
                "stabilizer variants are omitted"
            ),
            "saved_candidates": sum(r["saved_candidates"] for r in records),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--seed", type=int, default=700)
    args = parser.parse_args()
    main(args.output, args.seconds, args.seed)
