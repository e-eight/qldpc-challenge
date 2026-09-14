"""Distance-aware merge-graft reduction of an open-boundary tile code.

The move (a row-space form of the qubit removal of Liang-Eberhardt-Chen,
arXiv:2504.08887 Sec. III D/E).  Let S be any element of the row space of H_X
(an X-type stabilizer), with support T, and let a in T:

  1. make one generator of H_X equal to S (a row operation: the span, and with
     it the code, is unchanged);
  2. add S into every other X row that meets a, so that a survives only in S;
  3. CNOT fan-out from a to every b in T\\{a}:  H_X[:, b] ^= H_X[:, a] and
     H_Z[:, a] ^= H_Z[:, b].

After step 3 the row S is the weight-1 stabilizer X_a and column a of H_Z is
zero (css commutation forces it), so qubit a is disentangled: delete it and the
row S.  k = n - rank H_X - rank H_Z is unchanged, and CSS commutation survives.

The move is NOT distance preserving: the CNOTs are not weight preserving, so a
logical operator's weight can change (measured on the tiny hypergraph-product
code [[25,1,4]]: a single fan-out step drops d from 4 to 3; a chain that ignores
distance collapses d to 1).  Every candidate is therefore screened with the same
randomized-information-set upper bound the board uses (``gf2_fast``), and a move
is accepted only if the resulting code still has no logical lighter than a
floor.  Restricted to |S| = 1 the move needs no CNOT and preserves distance
exactly; that case is the _cleanup pass below.

Usage:
  uv run --frozen python research/reduce_tile.py codes/578-18-20.json \
      --d-floor 20 --screen 8000 --confirm 30000 --out reduced.npz

Requires the optional C++ RIS accelerator (`make fast`).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "kit"), os.path.join(_HERE, "..", "verify")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from css import compute_k, verify_css  # noqa: E402


def cleanup(HX, HZ):
    """Eliminate weight-1 stabilizers and decoupled qubits.

    This is arXiv:2504.08887 Sec. III D step 4 and is DISTANCE PRESERVING by
    argument (a weight-1 stabilizer X_q forces q out of every opposite-type
    check, and every logical class has a representative avoiding q), so it runs
    no distance search.  Returns (HX, HZ, kept) with kept[i] the original
    column index of new column i.
    """
    HX, HZ = HX.copy(), HZ.copy()
    n = HX.shape[1] if HX.size else HZ.shape[1]
    kept = list(range(n))
    while True:
        changed = False
        for name in ("X", "Z"):
            H = HX if name == "X" else HZ
            if H.size:
                keep = H.sum(1) > 0
                if not keep.all():
                    H = H[keep]
                    if name == "X":
                        HX = H
                    else:
                        HZ = H
                    changed = True
            H = HX if name == "X" else HZ
            if not H.size:
                continue
            ones = np.where(H.sum(1) == 1)[0]
            if ones.size:
                r = ones[0]
                q = int(np.argmax(H[r]))
                for rr in np.where(H[:, q] == 1)[0]:
                    if rr != r:
                        H[rr] ^= H[r]
                assert not ((HZ if name == "X" else HX)[:, q].any())
                H = np.delete(H, r, axis=0)
                if name == "X":
                    HX = np.delete(H, q, axis=1)
                    HZ = np.delete(HZ, q, axis=1)
                else:
                    HZ = np.delete(H, q, axis=1)
                    HX = np.delete(HX, q, axis=1)
                kept.pop(q)
                changed = True
        if not changed:
            break
    return HX, HZ, kept


def fast_distance(HX, HZ, trials, seed, threads=8):
    """RIS upper bound on min(d_X, d_Z) via the gf2_fast accelerator."""
    import gf2_fast
    n = HX.shape[1]
    w, side, sup = gf2_fast.distance_rand_witness(
        np.asarray(HX, dtype=np.int8), np.asarray(HZ, dtype=np.int8),
        int(trials), int(seed), 8, int(threads))
    return (float("inf") if w > n else int(w)), side, sup


def graft(HX, HZ, typ, q):
    """Delete a qubit q lying in exactly one stabilizer of type ``typ``,
    together with that stabilizer, then run the distance-preserving cleanup."""
    H = HX if typ == "X" else HZ
    r = int(np.where(H[:, q] == 1)[0][0])
    if typ == "X":
        HX2, HZ2 = np.delete(HX, r, axis=0), HZ.copy()
    else:
        HZ2, HX2 = np.delete(HZ, r, axis=0), HX.copy()
    HX2 = np.delete(HX2, q, axis=1)
    HZ2 = np.delete(HZ2, q, axis=1)
    rel = [i for i in range(HX.shape[1]) if i != q]
    HX2, HZ2, kept_rel = cleanup(HX2, HZ2)
    return HX2, HZ2, [rel[i] for i in kept_rel]


def candidates(HX, HZ):
    out = []
    for typ, H in (("X", HX), ("Z", HZ)):
        if not H.size:
            continue
        for q in np.where(H.sum(0) == 1)[0]:
            out.append((typ, int(q)))
    return out


def reduce_code(HX, HZ, coords, k0, d_floor, screen, confirm, log=print):
    while True:
        found = None
        for typ, q in candidates(HX, HZ):
            HX2, HZ2, ids = graft(HX, HZ, typ, q)
            if not HX2.size or HX2.shape[1] != HZ2.shape[1]:
                continue
            if compute_k(HX2, HZ2) != k0 or not verify_css(HX2, HZ2):
                continue
            d, _, _ = fast_distance(HX2, HZ2, screen, seed=1)
            if d < d_floor:
                continue
            key = (HX2.shape[1], -d)
            if found is None or key < found[0]:
                found = (key, typ, q, HX2, HZ2, ids, d)
        if found is None:
            break
        _, typ, q, HX2, HZ2, ids, _ = found
        dc, _, _ = fast_distance(HX2, HZ2, confirm, seed=5)
        if dc < d_floor:
            break
        HX, HZ, coords = HX2, HZ2, coords[ids]
        log(f"graft {typ} q={q}: n={HX.shape[1]} k={k0} d>={dc}")
    return HX, HZ, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draft", help="codes/<n>-<k>-<d>.json to reduce")
    ap.add_argument("--d-floor", type=int, required=True)
    ap.add_argument("--screen", type=int, default=8000)
    ap.add_argument("--confirm", type=int, default=30000)
    ap.add_argument("--out", default="reduced.npz")
    args = ap.parse_args()

    with open(args.draft) as f:
        doc = json.load(f)
    n = doc["n"]
    HX = np.zeros((len(doc["checks"]["X"]), n), dtype=np.int8)
    for i, s in enumerate(doc["checks"]["X"]):
        HX[i, s] = 1
    HZ = np.zeros((len(doc["checks"]["Z"]), n), dtype=np.int8)
    for i, s in enumerate(doc["checks"]["Z"]):
        HZ[i, s] = 1
    coords = np.asarray(doc["locality"]["coordinates"], float)
    k0 = compute_k(HX, HZ)
    HX2, HZ2, coords2 = reduce_code(HX, HZ, coords, k0, args.d_floor,
                                    args.screen, args.confirm)
    np.savez_compressed(args.out, HX=HX2.astype(np.uint8),
                        HZ=HZ2.astype(np.uint8), coords=coords2)


if __name__ == "__main__":
    main()
