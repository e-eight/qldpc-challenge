import sys, time, json
sys.path.insert(0, '/tmp')
from planar_probe import probe

CAND = {
    'C_w11_k18': ([[0, 3], [3, 2], [3, 0]],
                  [[0, 0], [0, 1], [2, 1], [2, 3], [3, 3], [2, 0]]),
    'B_w11_k15': ([[0, 0], [1, 3], [3, 3], [1, 0], [2, 3], [2, 1]],
                  [[3, 0], [2, 2], [0, 2]]),
    'A_w11_k12': ([[0, 3], [0, 2], [0, 0], [3, 1], [2, 1], [3, 0]],
                  [[1, 2], [2, 2], [0, 0]]),
    'D_w10_k9': ([[1, 2], [3, 3], [0, 2]],
                 [[1, 0], [3, 0], [0, 0], [3, 1], [1, 2], [0, 2]]),
}

for name, (Sf, Sg) in CAND.items():
    for L in (11, 13, 15):
        t = time.time()
        try:
            rec = probe(Sf, Sg, L=L, trials=20000, seed=7, threads=2)
        except Exception as e:
            print(f'{name} L={L}: ERROR {type(e).__name__}: {e}', flush=True)
            continue
        if rec is None:
            print(f'{name} L={L}: empty', flush=True)
            continue
        rec['eff'] = round(rec['k'] * (rec['d'] or 0) ** 2 / rec['n'], 3)
        print(f"{name} L={L}: n={rec['n']} k={rec['k']} d<={rec['d']} w={rec['w']} "
              f"r={rec['r']} K={rec['eff']} [{time.time()-t:.0f}s]", flush=True)
