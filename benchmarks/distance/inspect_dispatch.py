"""Matrix-only detector diagnostics; never invokes witness search."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from run import np

# isort: split
from common import HERE, ROOT, atomic_json, matrix_hash, sha256
from dispatch_search import CONFIG, detector


def create_corpus(output):
    source = HERE / "results/candidate-study-v2/corpus"
    manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    cases, transforms = [], {}
    for original in manifest["cases"]:
        if original.get("variant"):
            continue
        case = copy.deepcopy(original)
        shutil.copyfile(source / case["file"], output / case["file"])
        cases.append(case)
        if original["id"] not in {"board-700-222-28", "board-682-172-79", "regression-690-182"}:
            continue
        with np.load(source / original["file"], allow_pickle=False) as data:
            hx, hz = data["hx"], data["hz"]
        rng = np.random.default_rng(1460)
        columns = rng.permutation(hx.shape[1])
        inverse = np.argsort(columns)
        for variant in ("columns", "columns-rows"):
            rx = rng.permutation(len(hx)) if variant == "columns-rows" else np.arange(len(hx))
            rz = rng.permutation(len(hz)) if variant == "columns-rows" else np.arange(len(hz))
            x, z = hx[rx][:, columns], hz[rz][:, columns]
            case = copy.deepcopy(original)
            case.update(
                id=original["id"] + "-" + variant,
                variant=variant,
                source_case=original["id"],
                matrix_sha256=matrix_hash(x, z),
            )
            case["file"] = case["id"] + ".npz"
            for field in ("reference", "previous_reference"):
                if field in case:
                    case[field] = {s: sorted(inverse[case[field][s]].tolist()) for s in ("X", "Z")}
            np.savez_compressed(output / case["file"], hx=x, hz=z)
            cases.append(case)
            transforms[case["id"]] = dict(
                source_case=original["id"], columns=columns.tolist(), rows_x=rx.tolist(), rows_z=rz.tolist()
            )
    atomic_json(output / "manifest.json", dict(sources=manifest["sources"], cases=cases))
    atomic_json(output / "transforms.json", transforms)


def main(corpus, output):
    corpus = corpus.resolve()
    manifest = json.loads((corpus / "manifest.json").read_text())
    results, inputs = [], [corpus / "manifest.json", corpus / "transforms.json"]
    for case in manifest["cases"]:
        path = corpus / case["file"]
        inputs.append(path)
        with np.load(path, allow_pickle=False) as data:
            for side in ("hx", "hz"):
                result = detector.detect(
                    np.ascontiguousarray(data[side], dtype=np.uint8), 0.050, CONFIG["minimum_confidence"]
                )
                results.append(dict(case=case["id"], side=side, **result))
    sources = [Path(__file__), HERE / "dispatch_search.py", ROOT / "native/structure_dispatch/detect.cpp"]
    atomic_json(
        output,
        dict(
            scope="Detection only; no witness search; parameters chosen before corpus inspection",
            results=results,
            input_hashes={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
            source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sources},
            binary_sha256=sha256(Path(detector.__file__)),
        ),
    )
    for row in results:
        print(
            row["case"],
            row["side"],
            row["status"],
            round(row["confidence"], 3),
            round(row["elapsed_seconds"] * 1000, 3),
            "ms",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--create-corpus", action="store_true")
    args = parser.parse_args()
    if args.create_corpus:
        create_corpus(args.corpus)
    main(args.corpus, args.output)
