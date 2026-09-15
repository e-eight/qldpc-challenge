"""Rebuild 2BGA sweep candidates and deep-verify distance."""
import sys, json, time
import numpy as np

BASE = '/home/soham/Projects/unitaryfoundation/qldpc-challenge'
sys.path[:0] = [BASE + '/research/kit', BASE + '/verify', '/tmp']
from group_algebra import build_2bga, metacyclic
from search import _primitive_root
from css import compute_k, verify_css
import gf2_fast
from boardlib import dominators


def rebuild(spec):
    fam = spec['family']
    if fam == '2bga-metacyclic':
        mul, _ = metacyclic(int(spec['n']), int(spec['k_m']), int(spec['r']))
    elif fam == '2bga-affine':
        q = int(spec['q'])
        mul, _ = metacyclic(q, q - 1, _primitive_root(q))
    else:
        raise ValueError(fam)
    return build_2bga(mul, [int(x) for x in spec['a']], [int(x) for x in spec['b']])


cands = []
for f in ('/tmp/sweep_kasai_w3_404.json', '/tmp/sweep_metacyclic_w3_303.json'):
    data = json.load(open(f))
    for r in data['records']:
        if not dominators(r['n'], r['k'], r['d'], r['w'], 'unrestricted'):
            cands.append(r)
# dedup by fingerprint, sort by screening d desc
seen = set()
uniq = []
for r in sorted(cands, key=lambda r: -r['d']):
    if r['fingerprint'] in seen:
        continue
    seen.add(r['fingerprint'])
    uniq.append(r)
print(f'non-dominated 2BGA candidates: {len(uniq)}', flush=True)

saved = []
for r in uniq[:6]:
    try:
        HX, HZ = rebuild(r['spec'])
    except Exception as e:
        print(f"{r['fingerprint']}: rebuild failed {e}", flush=True)
        continue
    n = int(HX.shape[1])
    k = int(compute_k(HX, HZ))
    w = int(max(max((int(x.sum()) for x in HX), default=0),
                max((int(x.sum()) for x in HZ), default=0)))
    assert verify_css(HX, HZ)
    best, sides = n + 1, {}
    t = time.time()
    for trials in (300_000,):
        for seed in (6101, 6102):
            ww, side, sup = gf2_fast.distance_rand_witness(HX, HZ, int(trials), int(seed), 8, 8)
            if ww <= n:
                best = min(best, ww)
                sides.setdefault(side, []).append(ww)
    print(f"[[{n},{k},<={r['d']}]] w{w} screen_d={r['d']} K={r['efficiency']:.1f} "
          f"spec={json.dumps(r['spec'])[:70]} | DEEP -> lightest {best} "
          f"sides={[(s, min(v)) for s, v in sides.items()]} [{time.time()-t:.0f}s]", flush=True)
    if best <= n:
        np.savez(f'/tmp/cand_{r["fingerprint"]}.npz', HX=HX, HZ=HZ)
        saved.append({'fingerprint': r['fingerprint'], 'spec': r['spec'], 'n': n, 'k': k,
                      'screen_d': r['d'], 'deep_lightest': int(best), 'w': w})
json.dump(saved, open('/tmp/twobga_deep.json', 'w'))
print('saved', len(saved), flush=True)
