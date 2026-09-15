import sys, time, math, random, json
import numpy as np

BASE = '/home/soham/Projects/unitaryfoundation/qldpc-challenge'
sys.path[:0] = [BASE + '/research/kit', BASE + '/research/local2d', BASE + '/verify']
from boundary_engine import build_planar
from planar import grid_coordinates
from css import compute_k, verify_css
from surrogate import distance_rand


def _diam(pts):
    return max(math.dist(pts[i], pts[j])
               for i in range(len(pts)) for j in range(i + 1, len(pts)))


def max_radius(HX, HZ, coords):
    best = 0.0
    for H in (HX, HZ):
        for r in H:
            s = np.nonzero(r)[0].tolist()
            if len(s) > 1:
                best = max(best, _diam([coords[q] for q in s]))
    return best


def probe(Sf, Sg, L=9, trials=3000, seed=0, threads=2):
    t = time.time()
    HX, HZ, info = build_planar(L, L, [tuple(x) for x in Sf], [tuple(x) for x in Sg])
    if HX.size == 0 or HZ.size == 0:
        return None
    n = int(HX.shape[1])
    k = int(compute_k(HX, HZ))
    w = int(max(max((int(r.sum()) for r in HX), default=0),
                max((int(r.sum()) for r in HZ), default=0)))
    kept = info.get('kept_qubits')
    coords = grid_coordinates(L, L, kept=kept)
    r = max_radius(HX, HZ, coords)
    d = distance_rand(HX, HZ, trials=trials, seed=seed, backend='auto', threads=threads)
    return dict(n=n, k=k, w=w, r=round(r, 4), d=(None if d == float('inf') else int(d)),
                L=L, dt=round(time.time() - t, 1),
                Sf=[list(x) for x in Sf], Sg=[list(x) for x in Sg], css=verify_css(HX, HZ))


if __name__ == '__main__':
    f8 = [(0, 0), (0, 3), (2, 2), (3, 0)]
    g8 = [(0, 1), (1, 1), (2, 0), (3, 3)]
    print('baseline weight-8 tile @L=9:', probe(f8, g8, L=9, trials=3000), flush=True)
