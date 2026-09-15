import sys, time, json
import numpy as np
sys.path.insert(0, "/home/soham/Projects/unitaryfoundation/qldpc-challenge/research/kit")
sys.path.insert(0, "/home/soham/Projects/unitaryfoundation/qldpc-challenge/verify")
import gf2_fast
from css import compute_k, verify_css

doc = json.load(open("/home/soham/Projects/unitaryfoundation/qldpc-challenge/codes/566-18-20.json"))
n = doc["n"]
HX = np.zeros((len(doc["checks"]["X"]), n), dtype=np.int8)
for i, s in enumerate(doc["checks"]["X"]): HX[i, s] = 1
HZ = np.zeros((len(doc["checks"]["Z"]), n), dtype=np.int8)
for i, s in enumerate(doc["checks"]["Z"]): HZ[i, s] = 1
print("reloaded n", n, "k", compute_k(HX, HZ), "css", verify_css(HX, HZ), flush=True)

# Independent deep re-verification: fresh seed range (far from prior runs), high trials.
for trials, seeds in ((300_000, (9001, 9002, 9003, 9004)), (2_000_000, (9101, 9102, 9103))):
    t = time.time(); best = n + 1; sides = {}
    for s in seeds:
        w, side, sup = gf2_fast.distance_rand_witness(HX, HZ, int(trials), int(s), 8, 8)
        if w <= n:
            best = min(best, w)
            sides.setdefault(side, []).append(w)
    print(f"DEEP trials={trials} seeds={list(seeds)}: lightest logical found = {best} "
          f"sides={[(k2, min(v)) for k2, v in sides.items()]} [{time.time()-t:.0f}s]", flush=True)
