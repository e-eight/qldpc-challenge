"""Freeze a construction-diverse corpus without witness search."""

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from run import np

# isort: split
from common import ROOT, atomic_json, matrix_hash, sha256
from corpus import supports_matrix

CASES = ("684-12-73", "684-8-85", "682-182-76", "700-140-22", "664-170-18", "676-4-13")


def main():
    root = HERE / "results/component-study"
    corpus = root / "corpus"
    corpus.mkdir(exist_ok=False)
    perm = root / "permuted-corpus"
    perm.mkdir(exist_ok=False)
    cases = []
    variants = []
    transforms = {}
    rng = np.random.default_rng(1840)
    for name in CASES:
        source = ROOT / "codes" / f"{name}.json"
        doc = json.loads(source.read_text())
        n = doc["n"]
        hx, hz = [supports_matrix(doc["checks"][s], n) for s in ("X", "Z")]
        refs = {s: doc["distance"][s]["witness"] for s in ("X", "Z")}
        updates = []
        if name.startswith("684"):
            prior = HERE / "results/block-collision-study"
            index = json.loads((prior / "witness-index.json").read_text())
            for s in ("X", "Z"):
                winner = next(w for w in index if w["case"] == name and w["side"] == s)
                file = prior / winner["artifact"]
                refs[s] = json.loads(file.read_text())["distance"][s]["witness"]
                updates.append(dict(source=str(file.relative_to(ROOT)), sha256=sha256(file)))
        if name.startswith(("684", "682")):
            blocks = [list(range(n // 2)), list(range(n // 2, n))]
        elif name == "700-140-22":
            blocks = [
                list(range(a * 140, (a + 1) * 140)) + list(range(b * 140, (b + 1) * 140))
                for a, b in itertools.combinations(range(5), 2)
            ]
        elif name == "664-170-18":
            blocks = [list(range(i * 83, (i + 1) * 83)) for i in range(8)]
        else:
            blocks = []
        case = dict(
            id=name,
            n=n,
            k=doc["k"],
            family=doc.get("family", "affine"),
            file=name + ".npz",
            reference=refs,
            source=str(source.relative_to(ROOT)),
            source_sha256=sha256(source),
            matrix_sha256=matrix_hash(hx, hz),
            blocks=blocks,
            reference_updates=updates,
            construction=doc["provenance"]["construction"],
        )
        cases.append(case)
        np.savez_compressed(corpus / case["file"], hx=hx, hz=hz)
        columns = rng.permutation(n)
        rx = rng.permutation(len(hx))
        rz = rng.permutation(len(hz))
        inverse = np.argsort(columns)
        px, pz = hx[rx][:, columns], hz[rz][:, columns]
        variant = dict(
            case,
            id=name + "-permuted",
            source_case=name,
            file=name + "-permuted.npz",
            blocks=[],
            reference={s: sorted(int(inverse[q]) for q in refs[s]) for s in refs},
            matrix_sha256=matrix_hash(px, pz),
        )
        variants.append(variant)
        np.savez_compressed(perm / variant["file"], hx=px, hz=pz)
        transforms[variant["id"]] = dict(
            source_case=name, columns=columns.tolist(), rows_x=rx.tolist(), rows_z=rz.tolist()
        )
    atomic_json(corpus / "manifest.json", dict(sources={"repository": "frozen local submissions"}, cases=cases))
    atomic_json(
        perm / "manifest.json", dict(sources={"repository": "permuted frozen local submissions"}, cases=variants)
    )
    atomic_json(perm / "transforms.json", transforms)
    print("Frozen", CASES)


if __name__ == "__main__":
    main()
