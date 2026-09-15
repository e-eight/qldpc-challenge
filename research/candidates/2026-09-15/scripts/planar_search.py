import sys, time, random, json
sys.path.insert(0, '/tmp')
from planar_probe import probe

BOX = [(i, j) for i in range(4) for j in range(4)]


def run(sizes, N, L=9, trials=2000, seed=0):
    rng = random.Random(seed)
    out = []
    t0 = time.time()
    for it in range(N):
        kf, kg = rng.choice(sizes)
        Sf = rng.sample(BOX, kf)
        Sg = rng.sample(BOX, kg)
        try:
            rec = probe(Sf, Sg, L=L, trials=trials, seed=seed + it, threads=2)
        except Exception:
            rec = None
        if not rec or rec['k'] < 4 or rec['css'] is False:
            continue
        if rec['r'] > 7.0:
            continue
        rec['eff'] = round(rec['k'] * rec['d'] ** 2 / rec['n'], 3) if rec['d'] else 0
        out.append(rec)
    # dedup by (n,k,d,w)
    seen, uniq = set(), []
    for r in out:
        key = (r['n'], r['k'], r['d'], r['w'])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    uniq.sort(key=lambda r: (-r['k'], -(r['d'] or 0), r['n']))
    print(f'sizes={sizes} N={N} L={L}: {len(uniq)} usable in {time.time()-t0:.0f}s', flush=True)
    for r in uniq[:25]:
        print(f"  [[{r['n']},{r['k']},{r['d']}]] w{r['w']} r={r['r']} eff={r['eff']} "
              f"Sf={r['Sf']} Sg={r['Sg']}", flush=True)
    return uniq


if __name__ == '__main__':
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run([(4, 5), (5, 4), (3, 6), (6, 3)], N, L=L, trials=2000, seed=seed)
