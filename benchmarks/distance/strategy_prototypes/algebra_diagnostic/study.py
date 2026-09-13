"""Run a bounded matrix-only algebraic audit on the frozen six-code corpus."""

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from run import gf2, np

# isort: split
from common import ROOT, atomic_json, matrix_hash, sha256
from strategy_prototypes.algebra_diagnostic.algebra import (
    balanced_profiles,
    components,
    decomposition,
    logical_rank,
    polynomial_model,
    preserves_rowspace,
    row_mix,
)


def preserved(root):
    count = 0
    for name in ("component-study", "block-collision-study", "large-structure-study"):
        previous = HERE / "results" / name
        for line in (previous / "SHA256SUMS").read_text().splitlines():
            digest, path = line.split("  ", 1)
            assert sha256(previous / path) == digest, path
            count += 1
    return count


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    source_dir = Path(__file__).resolve().parent
    sources = list(source_dir.glob("*.py")) + [
        source_dir / "PLAN.md",
        ROOT / "verify/gf2.py",
        HERE / "run.py",
        HERE / "common.py",
    ]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    atomic_json(output / "environment.json", dict(source_hashes=hashes, python=sys.version, search_performed=False))
    prior = preserved(output)
    corpus = HERE / "results/component-study/corpus"
    manifest = json.loads((corpus / "manifest.json").read_text())
    transforms = json.loads((HERE / "results/component-study/permuted-corpus/transforms.json").read_text())
    atomic_json(output / "manifest.json", manifest)
    all_rows = []
    cuts = []
    symmetries = []
    polynomials = []
    invariance = []
    total = time.perf_counter()
    for case in manifest["cases"]:
        begin = time.perf_counter()
        with np.load(corpus / case["file"]) as data:
            hx, hz = data["hx"], data["hz"]
        assert matrix_hash(hx, hz) == case["matrix_sha256"]
        (output / "matrices").mkdir(exist_ok=True)
        np.savez_compressed(output / "matrices" / case["file"], hx=hx, hz=hz)
        permutation = np.array(transforms[case["id"] + "-permuted"]["columns"])
        inverse = np.argsort(permutation)
        rx = np.array(transforms[case["id"] + "-permuted"]["rows_x"])
        rz = np.array(transforms[case["id"] + "-permuted"]["rows_z"])
        mixed_x, mixed_z = row_mix(hx, 1910), row_mix(hz, 1911)
        for side, own, opposite, mixed, row_order in [("X", hx, hz, mixed_z, rz), ("Z", hz, hx, mixed_x, rx)]:
            r_own = gf2.rank(own)
            n = case["n"]
            seen = set()
            rank_cache = {}
            proposals = [("whole", list(range(n)))] + [(f"metadata{i}", b) for i, b in enumerate(case["blocks"])]
            for label, proposed_coords in proposals:
                key = tuple(sorted(proposed_coords))
                if key in seen:
                    continue
                seen.add(key)
                coords = list(key)
                h = opposite[:, coords]
                tick = time.perf_counter()
                groups, reduced, pivots = decomposition(h)
                graph = components(h)
                exact = []
                for group in groups:
                    support = [coords[q] for q in group["coordinates"]]
                    signature = tuple(support)
                    if signature not in rank_cache:
                        rank_cache[signature] = logical_rank(own, opposite, support, r_own) if group["dimension"] else 0
                    exact.append(
                        dict(coordinates=support, dimension=group["dimension"], logical_rank=rank_cache[signature])
                    )
                assert sum(g["dimension"] for g in exact) == len(coords) - len(pivots)
                # Every kernel vector can split over these masks: no reduced check crosses groups.
                ownership = {q: i for i, g in enumerate(groups) for q in g["coordinates"]}
                for row in reduced:
                    assert len({ownership[int(q)] for q in np.flatnonzero(row)}) <= 1
                expected = {frozenset(g["coordinates"]) for g in exact}
                mixed_groups = decomposition(mixed[:, coords])[0]
                assert {frozenset(coords[q] for q in g["coordinates"]) for g in mixed_groups} == expected
                transformed_coords = sorted(int(inverse[q]) for q in coords)
                transformed = opposite[row_order][:, permutation][:, transformed_coords]
                recovered = decomposition(transformed)[0]
                assert {
                    frozenset(int(permutation[transformed_coords[q]]) for q in g["coordinates"]) for g in recovered
                } == expected
                all_rows.append(
                    dict(
                        case=case["id"],
                        side=side,
                        proposal=label,
                        coordinates=coords,
                        rank=len(pivots),
                        graph_components=[[coords[q] for q in g] for g in graph],
                        exact_components=exact,
                        seconds=time.perf_counter() - tick,
                    )
                )
                if label == "whole":
                    domains = [("whole", list(range(n)))] + [
                        (f"exact{i}", g["coordinates"])
                        for i, g in enumerate(exact)
                        if g["dimension"] and len(g["coordinates"]) >= 32 and len(g["coordinates"]) < n
                    ]
                    for domain, support in domains:
                        profiles = balanced_profiles(opposite[:, support])
                        for p in profiles:
                            p["coordinates"] = [support[q] for q in p["coordinates"]]
                            # Independent trusted rank verifies the selected profile minimum.
                            s = set(p["coordinates"])
                            t = [q for q in support if q not in s]
                            assert p["coupling"] == gf2.rank(opposite[:, sorted(s)]) + gf2.rank(
                                opposite[:, t]
                            ) - gf2.rank(opposite[:, support])
                            p["logical_rank_left"] = logical_rank(own, opposite, sorted(s), r_own)
                            p["logical_rank_right"] = logical_rank(own, opposite, t, r_own)
                        cuts.append(
                            dict(
                                case=case["id"], side=side, domain=domain, domain_coordinates=support, profiles=profiles
                            )
                        )
                elif 0 < len(coords) < n:
                    complement = sorted(set(range(n)) - set(coords))
                    coupling = len(pivots) + gf2.rank(opposite[:, complement]) - gf2.rank(opposite)
                    cuts.append(
                        dict(
                            case=case["id"],
                            side=side,
                            domain=label,
                            coordinates=coords,
                            coupling=coupling,
                            logical_rank_left=logical_rank(own, opposite, coords, r_own),
                        )
                    )
                invariance.append(
                    dict(case=case["id"], side=side, proposal=label, row_mixing=True, joint_permutation=True)
                )
        accepted = []
        for length in range(3, case["n"] + 1):
            if case["n"] % length:
                continue
            p = [(q // length) * length + (q + 1) % length for q in range(case["n"])]
            x = preserves_rowspace(hx, p)
            z = preserves_rowspace(hz, p)
            entry = dict(case=case["id"], block_length=length, blocks=case["n"] // length, X=x, Z=z, accepted=x and z)
            if x and z:
                assert preserves_rowspace(mixed_x, p) and preserves_rowspace(mixed_z, p)
                conjugate = [int(inverse[p[int(permutation[q])]]) for q in range(case["n"])]
                assert preserves_rowspace(hx[rx][:, permutation], conjugate) and preserves_rowspace(
                    hz[rz][:, permutation], conjugate
                )
                entry["permutation"] = p
                entry["conjugated_permutation"] = conjugate
                accepted.append(length)
            symmetries.append(entry)
        model = polynomial_model(hx, hz) if case["n"] // 2 in accepted else dict(status="no_verified_half_shift")
        polynomials.append(dict(case=case["id"], **model))
        for name, value in [
            ("decompositions.json", all_rows),
            ("cuts.json", cuts),
            ("symmetries.json", symmetries),
            ("polynomials.json", polynomials),
            ("invariance.json", invariance),
        ]:
            atomic_json(output / name, value)
        print(
            json.dumps(
                dict(
                    case=case["id"],
                    seconds=time.perf_counter() - begin,
                    accepted_shift_lengths=accepted,
                    polynomial=model["status"],
                )
            ),
            flush=True,
        )
    assert preserved(output) == prior
    for name, digest in hashes.items():
        assert sha256(ROOT / name) == digest, name
    atomic_json(
        output / "completed.json",
        dict(
            cases=len(manifest["cases"]),
            seconds=time.perf_counter() - total,
            decomposition_proposals=len(all_rows),
            cut_domains=len(cuts),
            symmetry_proposals=len(symmetries),
            invariance_checks=len(invariance),
            prior_artifacts_unchanged=prior,
            search_performed=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
