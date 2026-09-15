"""Rebuild the LP sweep's non-dominated candidates and deep-verify distance."""
import sys, json, time
import numpy as np

BASE = '/home/soham/Projects/unitaryfoundation/qldpc-challenge'
sys.path[:0] = [BASE + '/research/kit', BASE + '/verify', '/tmp']
from products import sample_lifted_product
from search import fingerprint
from css import compute_k, verify_css
import gf2_fast
from boardlib import dominators

data = json.load(open('/tmp/sweep_lp_w3_606.json'))
recs = data['records']
nd = [r for r in recs if not dominators(r['n'], r['k'], r['d'], r['w'], 'unrestricted')]
nd.sort(key=lambda r: -r['efficiency'])
print(f'non-dominated LP candidates: {len(nd)}', flush=True)
want = {r['fingerprint']: r for r in nd}

gen = sample_lifted_product(1500, order_range=(30, 350), weight_a=3, weight_b=3, seed=606)
found = {}
for spec, HX, HZ in gen:
    fp = fingerprint(HX, HZ)
    if fp in want and fp not in found:
        found[fp] = (spec, HX, HZ)
    if len(found) == len(want):
        break
print(f'rebuilt {len(found)}/{len(want)}', flush=True)

for fp, r in [(r['fingerprint'], r) for r in nd[:8]]:
    if fp not in found:
        print(f'{fp}: could not rebuild', flush=True)
        continue
    spec, HX, HZ = found[fp]
    n = int(HX.shape[1])
    k = int(compute_k(HX, HZ))
    w = int(max(max((int(x.sum()) for x in HX), default=0),
                max((int(x.sum()) for x in HZ), default=0)))
    assert verify_css(HX, HZ)
    tag = f"[[{n},{k},<={r['d']}]] w{w} screen_d={r['d']} K={r['efficiency']}"
    best = n + 1
    sides = {}
    for trials in (300_000,):
        t = time.time()
        for seed in (5101, 5102):
            ww, side, sup = gf2_fast.distance_rand_witness(HX, HZ, int(trials), int(seed), 8, 8)
            if ww <= n:
                best = min(best, ww)
                sides.setdefault(side, []).append(ww)
        print(f'{tag} | DEEP trials={trials} x2 seeds -> lightest {best} '
              f'sides={[(s, min(v)) for s, v in sides.items()]} [{time.time()-t:.0f}s]', flush=True)
    json.dump({'spec': spec, 'n': n, 'k': k, 'screen_d': r['d'],
               'deep_lightest': None if best > n else int(best), 'w': w,
               'fingerprint': fp}, open(f'/tmp/lp_{fp}.json', 'w'))
