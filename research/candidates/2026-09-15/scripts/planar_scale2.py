import sys, time
sys.path.insert(0, '/tmp')
from planar_probe import probe
from boardlib import report

CAND = {
    'B_w11': ([[0, 0], [1, 3], [3, 3], [1, 0], [2, 3], [2, 1]],
              [[3, 0], [2, 2], [0, 2]]),
    'C_w11': ([[0, 3], [3, 2], [3, 0]],
              [[0, 0], [0, 1], [2, 1], [2, 3], [3, 3], [2, 0]]),
    'D_w10': ([[1, 2], [3, 3], [0, 2]],
              [[1, 0], [3, 0], [0, 0], [3, 1], [1, 2], [0, 2]]),
}

results = []
for name, (Sf, Sg) in CAND.items():
    for L in (17, 18, 19):
        t = time.time()
        try:
            rec = probe(Sf, Sg, L=L, trials=40000, seed=11, threads=2)
        except Exception as e:
            print(f'{name} L={L}: ERROR {e}', flush=True)
            continue
        if rec is None or rec['k'] == 0:
            print(f'{name} L={L}: empty/k0', flush=True)
            continue
        loc = 'local-2d-bilayer' if rec['r'] <= 7.0 else 'unrestricted'
        rec.update(name=name, loc=loc)
        results.append(rec)
        print(f"{name} L={L}: n={rec['n']} k={rec['k']} d<={rec['d']} w={rec['w']} "
              f"r={rec['r']} [{time.time()-t:.0f}s]", flush=True)
        if loc == 'local-2d-bilayer' and rec['d']:
            report(f'{name} L={L}', rec['n'], rec['k'], rec['d'], rec['w'], loc)

import json
json.dump(results, open('/tmp/planar_candidates.json', 'w'))
