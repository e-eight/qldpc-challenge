"""Reconstruct the three affine inputs and inspect kernels without searching."""

import json
import sys

import finite_probe as probe

sys.path.insert(0, str(probe.REPO))
from research.kit.group_algebra import build_2bga, metacyclic  # noqa: E402

SUPPORTS = {
    "684-10-101": ([43, 79, 102, 184, 219, 275], [43, 135, 168, 206, 314, 335]),
    "684-12-73": ([182, 323, 76, 217], [294, 208, 142, 82]),
    "684-8-85": ([294, 208, 142, 284], [6, 323, 76, 217]),
}


def main():
    mul, _ = metacyclic(19, 18, 2)
    groups = next(g for label, g in probe.proposals(probe.SPEC, 684) if label == probe.LABEL)
    records = []
    for name, (a, b) in SUPPORTS.items():
        path = probe.REPO / "codes" / f"{name}.json"
        doc = json.loads(path.read_text())
        hx, hz = [probe.supports_matrix(doc["checks"][s], doc["n"]) for s in ("X", "Z")]
        rx, rz = build_2bga(mul, a, b)
        assert probe.np.array_equal(rx, hx) and probe.np.array_equal(rz, hz), name
        dimensions = []
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            prepared = probe.ris_native.Prepared(own, opposite)
            encoders = [("both_orbit", groups)]
            for half in (0, 1):
                encoders.append((f"half_{half}_orbit", [g for g in groups if g[0] // 342 == half]))
                encoders.append((f"half_{half}_unrestricted", [[q] for q in range(half * 342, (half + 1) * 342)]))
            for label, encoder in encoders:
                space = probe.native.Space(opposite, prepared.logicals, encoder)
                dimensions.append(
                    dict(
                        side=side,
                        encoder=label,
                        columns=len(encoder),
                        dimension=space.dimension,
                        logical_rank=space.logical_rank,
                    )
                )
        records.append(
            dict(case=name, source_sha256=probe.sha256(path), a=a, b=b, exact_matrix_match=True, dimensions=dimensions)
        )
    probe.atomic_json(
        probe.REPO / "benchmarks/distance/results/large-structure-study/finite-diagnostics.json",
        dict(construction="HX=[L(a)|R(b)], HZ=[R(b).T|L(a).T]", group=probe.SPEC, cases=records),
    )
    print("All three affine inputs reconstructed exactly; dimensions recorded without witness search.")


if __name__ == "__main__":
    main()
