"""Very deep RIS confirmation of the leading non-dominated candidates."""
import sys, json, time
import numpy as np

BASE = '/home/soham/Projects/unitaryfoundation/qldpc-challenge'
sys.path[:0] = [BASE + '/research/kit', BASE + '/verify', '/tmp']
from products import sample_lifted_product
from group_algebra import build_2bga, metacyclic
from search import fingerprint
from css import compute_k, verify_css
import gf2_fast

LP_SEED = 606
TARGETS = {
    'lp672': '670ecbf0cc5275d4',    # [[672,6,<=60]] deep 44
    'lp684': 'd726ee4ea3884b01',    # [[684,6,<=68]] deep 42
    'lp648': 'd16f3f5fd4f0648d',    # [[648,4,<=54]] deep 46
    '2bga648': '542535054cfcabd3',  # metacyclic [[648,4,<=56]] deep 38
}


def rebuild(fp):
    if fp in ('670ecbf0cc5275d4', 'd726ee4ea3884b01', 'd16f3f5fd4f0648d'):
        for spec, HX, HZ in sample_lifted_product(1500, order_range=(30, 350),
                                                  weight_a=3, weight_b=3, seed=LP_SEED):
            if fingerprint(HX, HZ) == fp:
                return spec, HX, HZ
        raise SystemExit('not found')
    # metacyclic n=36,k_m=9,r=13 in the sweep record
    data = json.load(open('/tmp/sweep_metacyclic_w3_303.json'))
    rec = next(r for r in data['records'] if r['fingerprint'] == fp)
    s = rec['spec']
    mul, _ = metacyclic(int(s['n']), int(s['k_m']), int(s['r']))
    return s, *build_2bga(mul, [int(x) for x in s['a']], [int(x) for x in s['b']])


name = sys.argv[1]
tag, trials, nseeds = name, 2_000_000, 2
spec, HX, HZ = rebuild(TARGETS[name])
n = int(HX.shape[1]); k = int(compute_k(HX, HZ))
w = int(max(max((int(x.sum()) for x in HX), default=0), max((int(x.sum()) for x in HZ), default=0)))
assert verify_css(HX, HZ)
np.savez(f'/tmp/final_{name}.npz', HX=HX, HZ=HZ)
print(f'{name}: spec={json.dumps(spec)[:110]}', flush=True)
print(f'{name}: n={n} k={k} w={w} fingerprint={fingerprint(HX,HZ)}', flush=True)
t = time.time(); best = n + 1; sides = {}
for seed in (8801, 8802)[:nseeds]:
    ww, side, sup = gf2_fast.distance_rand_witness(HX, HZ, int(trials), int(seed), 8, 8)
    if ww <= n:
        best = min(best, ww); sides.setdefault(side, []).append(ww)
print(f'{name}: DEEP trials={trials} x{nseeds} -> lightest {best} '
      f'sides={[(s, min(v)) for s, v in sides.items()]} [{time.time()-t:.0f}s]', flush=True)
