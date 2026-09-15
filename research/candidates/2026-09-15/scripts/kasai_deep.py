import sys, json, time
import numpy as np
BASE = '/home/soham/Projects/unitaryfoundation/qldpc-challenge'
sys.path[:0] = [BASE + '/research/kit', BASE + '/verify', '/tmp']
from group_algebra import build_2bga, metacyclic
from search import _primitive_root
from css import compute_k, verify_css
import gf2_fast
from boardlib import dominators

data = json.load(open('/tmp/sweep_kasai_w4_505.json'))
nd = [r for r in data['records'] if not dominators(r['n'], r['k'], r['d'], r['w'], 'unrestricted')]
seen, uniq = set(), []
for r in sorted(nd, key=lambda r: -r['d']):
    if r['fingerprint'] in seen:
        continue
    seen.add(r['fingerprint'])
    uniq.append(r)
print(f'non-dominated kasai w4: {len(uniq)}', flush=True)

for r in uniq[:3]:
    s = r['spec']
    q = int(s['q'])
    mul, _ = metacyclic(q, q - 1, _primitive_root(q))
    HX, HZ = build_2bga(mul, [int(x) for x in s['a']], [int(x) for x in s['b']])
    n = int(HX.shape[1]); k = int(compute_k(HX, HZ))
    w = int(max(max((int(x.sum()) for x in HX), default=0), max((int(x.sum()) for x in HZ), default=0)))
    assert verify_css(HX, HZ)
    best, sides = n + 1, {}
    t = time.time()
    for seed in (7501, 7502):
        ww, side, sup = gf2_fast.distance_rand_witness(HX, HZ, 1_000_000, seed, 8, 8)
        if ww <= n:
            best = min(best, ww); sides.setdefault(side, []).append(ww)
    print(f"q={q} [[{n},{k},<={r['d']}]] w{w} screen_d={r['d']} a={s['a']} b={s['b']} | "
          f"DEEP 1M x2 -> lightest {best} sides={[(x, min(v)) for x, v in sides.items()]} "
          f"[{time.time()-t:.0f}s]", flush=True)
    np.savez(f'/tmp/kasai_{r["fingerprint"]}.npz', HX=HX, HZ=HZ)
