"""Common-factor polynomial diagnostic on original and randomly relabeled hard inputs."""

import argparse
import json
import os
import platform
import shutil
import tarfile
from pathlib import Path

from common import HERE, ROOT, atomic_json, code_target, matrix_hash, sha256
from polynomial_completion import PolynomialSearch
from run import np, validate_and_stage
from study_initialized import run_side
from study_strategies import snapshot


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {0})
    corpus = HERE / "results/reference-corpus"
    manifest = json.loads((corpus / "manifest.json").read_text())
    selected = {"board-700-222-28", "board-682-172-79", "regression-690-182"}
    sources, binaries = snapshot(output, ["guided", "descent"])
    extras = [
        Path(__file__),
        HERE / "polynomial_completion.py",
        HERE / "orbit_completion.py",
        HERE / "test_polynomial_completion.py",
        HERE / "initialized_search.py",
        HERE / "study_initialized.py",
    ]
    with tarfile.open(output / "polynomial-sources.tar.gz", "w:gz") as archive:
        for p in extras:
            name = str(p.relative_to(ROOT))
            sources[name] = sha256(p)
            archive.add(p, arcname=name)
    atomic_json(
        output / "environment.json",
        {
            "platform": platform.platform(),
            "cpu": 0,
            "source_hashes": sources,
            "binary_hashes": binaries,
            "permutation_seed": 920,
            "scope": "finite one-shot diagnostic, preparation included",
            "reference_policy": (
                "Only matrices enter PolynomialSearch; no parent code, reference support or target input"
            ),
        },
    )
    records = []
    for source in manifest["cases"]:
        if source["id"] not in selected:
            continue
        with np.load(corpus / source["file"], allow_pickle=False) as data:
            original_x, original_z = data["hx"], data["hz"]
        if matrix_hash(original_x, original_z) != source["matrix_sha256"]:
            raise ValueError("Input hash mismatch")
        permutation = np.random.default_rng(920).permutation(source["n"])
        for variant in ("original", "relabeled"):
            hx, hz = (
                (original_x, original_z)
                if variant == "original"
                else (original_x[:, permutation], original_z[:, permutation])
            )
            case = {
                "id": source["id"] + "-" + variant,
                "n": source["n"],
                "k": source["k"],
                "matrix_sha256": matrix_hash(hx, hz),
            }
            folder = output / case["id"]
            folder.mkdir()
            np.savez_compressed(folder / "matrices.npz", hx=hx, hz=hz)
            atomic_json(
                folder / "input.json",
                dict(
                    case, source_case=source["id"], permutation=permutation.tolist() if variant == "relabeled" else None
                ),
            )
            sides = {
                s: [run_side(PolynomialSearch(a, b), 0, 0, folder / f"{s}.jsonl")]
                for s, a, b in (("X", hx, hz), ("Z", hz, hx))
            }
            record = {
                "case": case["id"],
                "source_case": source["id"],
                "variant": variant,
                "target": code_target(source),
                "workers": sides,
                "validation_status": "pending",
                "seconds": sum(ws[0]["search_seconds"] for ws in sides.values()),
                "polynomial_best": min(
                    (e["weight"] for ws in sides.values() for e in ws[0]["events"] if e["stage"] == "polynomial"),
                    default=None,
                ),
            }
            atomic_json(folder / "result.json", record)
            fallback = {s: sides[s][0]["events"][0]["support"] for s in sides}
            duration, saved = validate_and_stage(hx, hz, case, sides, fallback, output.name)
            record.update(validation_status="passed", saved_candidates=saved, validation_seconds=duration)
            atomic_json(folder / "result.json", record)
            records.append(record)
            atomic_json(output / "results.json", records)
            print(
                json.dumps({k: record[k] for k in ("case", "polynomial_best", "seconds", "saved_candidates")}),
                flush=True,
            )
    lines = [
        "# Common-factor polynomial diagnostic",
        "",
        "A finite matrix-only pass rotates the first supplied check row within two equal blocks "
        "and divides both polynomials by their common factor. "
        "It tries gcd(a,b) and gcd(a,b,x^L+1). Every proposal is checked for zero syndrome "
        "and nontrivial logical parity. "
        "No source witnesses, parent matrices, target weights or code names enter the search.",
        "",
        "| Input | Layout | Best polynomial witness | Preparation + search seconds |",
        "|---|---|---:|---:|",
    ]
    for r in records:
        lines.append(f"| {r['source_case']} | {r['variant']} | {r['polynomial_best']} | {r['seconds']:.6f} |")
    lines += [
        "",
        "The relabeled control uses a common random qubit permutation for both matrices. "
        "Each measurement is a single observation, not a repeated timing estimate. "
        "This is a layout-dependent candidate generator, not a general distance algorithm or "
        "automatic symmetry discovery. Packaging-basis witnesses are retained but excluded from "
        "the polynomial result. All returned supports are independently validated and saved via the kit; "
        "no exact-distance or full-gate claim is made. The common-factor reduction is motivated by "
        "[Wang and Pryadko, Section III.1](https://arxiv.org/html/2203.17216#S3.SS1).",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    shutil.copyfile(HERE / "strategy_prototypes/INITIALIZED_PLAN.md", output / "parent-plan.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
