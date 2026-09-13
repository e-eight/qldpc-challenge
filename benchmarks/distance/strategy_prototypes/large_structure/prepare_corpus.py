"""Copy fixed board inputs and inspect restricted dimensions without searching."""

import json
import subprocess

from run import np

# isort: split
from common import HERE, ROOT, atomic_json, matrix_hash, sha256
from corpus import supports_matrix
from strategy_prototypes.large_structure.adapter import native, proposals
from study_strategies import ris_native

CASES = {
    "700-140-22": dict(kind="product", group_order=140),
    "700-6-32": dict(kind="torus", length=350, twist=207),
    "684-10-101": dict(kind="affine", prime=19, order=18, action=2),
}


def main():
    root = HERE / "results/large-structure-study"
    out = root / "corpus"
    out.mkdir(exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cases = []
    inspection = []
    for name, spec in CASES.items():
        source = ROOT / "codes" / f"{name}.json"
        doc = json.loads(source.read_text())
        hx, hz = [supports_matrix(doc["checks"][side], doc["n"]) for side in ("X", "Z")]
        np.savez_compressed(out / f"{name}.npz", hx=hx, hz=hz)
        case = dict(
            id=name,
            n=doc["n"],
            k=doc["k"],
            file=f"{name}.npz",
            source=str(source.relative_to(ROOT)),
            source_commit=commit,
            source_sha256=sha256(source),
            family=doc.get("family", ""),
            matrix_sha256=matrix_hash(hx, hz),
            structure_spec=spec,
            reference={side: doc["distance"][side]["witness"] for side in ("X", "Z")},
            target_status="repository witnessed upper bound, not an exact certificate",
        )
        cases.append(case)
        for side, own, opposite in [("X", hx, hz), ("Z", hz, hx)]:
            duals = ris_native.Prepared(own, opposite).logicals
            for label, groups in proposals(spec, doc["n"]):
                space = native.Space(opposite, duals, groups)
                inspection.append(
                    dict(
                        case=name,
                        side=side,
                        label=label,
                        columns=len(groups),
                        physical_support=sum(map(len, groups)),
                        dimension=space.dimension,
                        logical_rank=space.logical_rank,
                        workspace_bytes=space.workspace_bytes,
                    )
                )
    atomic_json(out / "manifest.json", dict(sources={"repository_base": commit}, cases=cases))
    atomic_json(out / "transforms.json", {})
    atomic_json(
        root / "dimension-diagnostic.json",
        dict(scope="Matrix/kernel dimensions only; no witness search", spaces=inspection),
    )
    for name in CASES:
        rows = [r for r in inspection if r["case"] == name]
        print(name, "spaces", len(rows), "with logicals", sum(r["logical_rank"] > 0 for r in rows))
        print(
            [
                (r["side"], r["label"], r["columns"], r["dimension"], r["logical_rank"])
                for r in rows
                if r["logical_rank"]
            ]
        )


if __name__ == "__main__":
    main()
