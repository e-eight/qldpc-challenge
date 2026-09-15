import sys, time
import numpy as np
sys.path[:0] = ['/home/soham/Projects/unitaryfoundation/qldpc-challenge/research/kit',
                '/home/soham/Projects/unitaryfoundation/qldpc-challenge/verify']
from css import compute_k, verify_css
import gf2_fast

name, trials, seedbase = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
z = np.load(f'/tmp/final_{name}.npz')
HX, HZ = z['HX'], z['HZ']
n = int(HX.shape[1]); k = int(compute_k(HX, HZ))
assert verify_css(HX, HZ)
print(f'{name}: n={n} k={k} css ok', flush=True)
best, sides = n + 1, {}
t = time.time()
for seed in (seedbase, seedbase + 1):
    ww, side, sup = gf2_fast.distance_rand_witness(HX, HZ, trials, seed, 8, 8)
    if ww <= n:
        best = min(best, ww); sides.setdefault(side, []).append(ww)
print(f'{name}: confirm trials={trials} x2 seeds={seedbase},{seedbase+1} -> lightest {best} '
      f'sides={[(s, min(v)) for s, v in sides.items()]} [{time.time()-t:.0f}s]', flush=True)
