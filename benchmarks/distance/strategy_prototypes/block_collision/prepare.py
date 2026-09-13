"""Freeze three affine inputs and inspect sparse checks without distance search."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from run import gf2, np  # noqa: E402

# isort: split
from common import ROOT, atomic_json, matrix_hash, sha256  # noqa: E402
from corpus import supports_matrix  # noqa: E402
from strategy_prototypes.large_structure.adapter import native  # noqa: E402
from study_strategies import ris_native  # noqa: E402

CASES = ("684-10-101", "684-12-73", "684-8-85")


def main():
    output = HERE / "results/block-collision-study"
    corpus = output / "corpus"
    corpus.mkdir(parents=True, exist_ok=False)
    cases, diagnostics = [], []
    for name in CASES:
        source = ROOT / "codes" / f"{name}.json"
        doc = json.loads(source.read_text())
        hx, hz = [supports_matrix(doc["checks"][s], doc["n"]) for s in ("X", "Z")]
        references = {s: doc["distance"][s]["witness"] for s in ("X", "Z")}
        previous = min(map(len, references.values()))
        updates = []
        if name == "684-10-101":
            witness = HERE / "results/large-structure-study/affine-Z-witness.json"
            references["Z"] = json.loads(witness.read_text())["distance"]["Z"]["witness"]
            updates.append(dict(source=str(witness.relative_to(ROOT)), sha256=sha256(witness)))
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            vector = np.zeros(doc["n"], dtype=np.int8)
            vector[references[side]] = 1
            assert gf2.commutes(vector, opposite) and not gf2.in_rowspace(vector, own)
            prepared = ris_native.Prepared(own, opposite)
            for half in (None, 0, 1):
                coords = list(range(doc["n"])) if half is None else list(range(half * 342, (half + 1) * 342))
                checks = opposite[:, coords]
                space = native.Space(opposite, prepared.logicals, [[q] for q in coords])
                parent = list(range(len(coords)))

                def find(q):
                    while parent[q] != q:
                        parent[q] = parent[parent[q]]
                        q = parent[q]
                    return q

                for row in checks:
                    support = np.flatnonzero(row)
                    for q in support[1:]:
                        parent[find(int(q))] = find(int(support[0]))
                components = {}
                for q in range(len(coords)):
                    key = find(q)
                    components[key] = components.get(key, 0) + 1
                diagnostics.append(
                    dict(
                        case=name,
                        side=side,
                        half=half,
                        columns=len(coords),
                        dimension=space.dimension,
                        logical_rank=space.logical_rank,
                        check_weight_max=int(checks.sum(axis=1).max()),
                        column_degree_max=int(checks.sum(axis=0).max()),
                        component_sizes=sorted(components.values()),
                    )
                )
        case = dict(
            id=name,
            n=doc["n"],
            k=doc["k"],
            file=name + ".npz",
            family=doc.get("family", ""),
            reference=references,
            original_target=previous,
            reference_updates=updates,
            source=str(source.relative_to(ROOT)),
            source_sha256=sha256(source),
            matrix_sha256=matrix_hash(hx, hz),
        )
        cases.append(case)
        np.savez_compressed(corpus / case["file"], hx=hx, hz=hz)
    atomic_json(corpus / "manifest.json", dict(sources={"repository": "frozen local submissions"}, cases=cases))
    atomic_json(output / "matrix-diagnostic.json", diagnostics)
    print("Corpus frozen; targets", [min(map(len, c["reference"].values())) for c in cases])


if __name__ == "__main__":
    main()
