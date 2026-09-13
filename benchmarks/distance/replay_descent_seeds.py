"""Replay a hybrid's exact RIS seeds to isolate improvements from local moves."""

import argparse
import json
import sys
import time
from pathlib import Path

from common import ROOT, atomic_json, sha256
from run import np, validate_and_stage

sys.path.insert(0, str(ROOT / "native" / "ris"))
import ris_native


def main(study, output, case_id):
    output.mkdir(parents=True, exist_ok=False)
    environment = json.loads((study / "environment.json").read_text())
    binary = Path(ris_native.__file__)
    if sha256(binary) != environment["binary_hashes"][str(binary.relative_to(ROOT))]:
        raise ValueError("RIS binary differs from the original experiment")
    manifest = json.loads((study / "manifest.json").read_text())
    case = next(c for c in manifest["cases"] if c["id"] == case_id)
    with np.load(study / "matrices" / case["file"]) as data:
        hx, hz = data["hx"], data["hz"]
    prepared = {"X": ris_native.Prepared(hx, hz), "Z": ris_native.Prepared(hz, hx)}
    originals = [
        r for r in json.loads((study / "results.json").read_text()) if r["case"] == case_id and r["method"] == "descent"
    ]
    results = []
    for original in originals:
        if original["validation_status"] != "passed":
            raise ValueError("Original run has not been validated")
        sides = {}
        for side in ("X", "Z"):
            counters = original["workers"][side][0]["counters"]
            config = counters["config"]
            if counters["seed_bases"] != counters["seed_sessions"] * config["seed_bases"]:
                raise ValueError("Seed-session work count mismatch")
            seed = original["seed"] * 1000003 + (499979 if side == "Z" else 0)
            rng = np.random.default_rng(seed)
            events = []
            start = time.perf_counter()
            for number in range(counters["seed_sessions"]):
                session = ris_native.Session(
                    prepared[side],
                    threads=1,
                    seed=int(rng.bit_generator.random_raw()),
                    block_size=config["block_size"],
                    restart_interval=config["restart_interval"],
                    exchange_proposals=config["exchange_proposals"],
                )
                batch = session.advance(config["seed_bases"])
                for event in batch.improvements:
                    events.append(
                        {
                            "seconds": time.perf_counter() - start,
                            "weight": event.weight,
                            "support": event.support,
                            "seed_session": number,
                        }
                    )
                # The hybrid consumes a second random seed for descent between
                # RIS sessions. The last unused draw cannot affect any replay.
                rng.bit_generator.random_raw()
            sides[side] = [{"events": events}]
        record = {
            "case": case_id,
            "seed": original["seed"],
            "workers": sides,
            "hybrid_best": min(
                s["best_in_budget"] for s in original["sides"].values() if s["best_in_budget"] is not None
            ),
            "seed_only_best": min(e["weight"] for ws in sides.values() for w in ws for e in w["events"]),
            "validation_status": "pending",
        }
        path = output / f"seed-{original['seed']}.json"
        atomic_json(path, record)
        # All replayed witnesses should already exist in the hybrid's saved
        # evidence. Record the specific document for every replay observation.
        saved_dir = (
            ROOT
            / "research"
            / "candidates"
            / "distance-benchmark"
            / case_id
            / f"{study.name}-descent-s{original['seed']}"
        )
        lookup, sequence = {}, 0
        for side in ("X", "Z"):
            seen = set()
            for event in original["workers"][side][0]["events"]:
                support = tuple(event["support"])
                if support in seen:
                    continue
                seen.add(support)
                files = list((saved_dir / str(sequence)).glob("*.json"))
                if len(files) != 1:
                    raise ValueError("Original saved witness is missing")
                lookup[(side, support)] = files[0]
                sequence += 1
        unmatched = False
        for side in ("X", "Z"):
            for event in sides[side][0]["events"]:
                saved = lookup.get((side, tuple(event["support"])))
                if saved is None:
                    unmatched = True
                    continue
                document = json.loads(saved.read_text())
                witness = document["distance"][side]
                if witness["witness"] != event["support"] or witness["value"] != event["weight"]:
                    raise ValueError("Original saved witness does not match replay")
                event["existing_saved_document"] = str(saved.relative_to(ROOT))
        if unmatched:
            # Never discard an unexpectedly different returned witness.
            validate_and_stage(hx, hz, case, sides, case["reference"], f"{output.name}-unexpected-s{original['seed']}")
            raise ValueError("Replay did not match original seed evidence; unexpected witnesses staged")
        record["validation_status"] = "matched_previously_validated_and_saved_witnesses"
        atomic_json(path, record)
        results.append({k: v for k, v in record.items() if k != "workers"})
    atomic_json(output / "results.json", results)
    atomic_json(
        output / "metadata.json",
        {
            "source_sha256": sha256(Path(__file__)),
            "native_binary_sha256": sha256(ris_native.__file__),
            "scope": "Replay of every original RIS seed session, including late seeds; not an equal-time comparison",
            "persistence": "Each replay witness linked to its existing independently validated kit submission",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", default="fresh-bb-960")
    args = parser.parse_args()
    main(args.study, args.output, args.case)
