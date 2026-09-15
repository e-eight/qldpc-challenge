"""Parallel family sweep for the qLDPC challenge (autoresearch loop step 3-4).

Screens a family with the kit funnel, writes every survivor to a JSON file
(spec + (n,k,d,w) + fingerprint) so a later stage can rebuild winners.
"""
import sys, json, time, argparse

KIT = '/home/soham/Projects/unitaryfoundation/qldpc-challenge/research/kit'
VER = '/home/soham/Projects/unitaryfoundation/qldpc-challenge/verify'
sys.path[:0] = [KIT, VER]

from search import (screen, sample_dihedral, sample_metacyclic,
                    sample_kasai_affine, sample_bb)
from products import sample_lifted_product, sample_balanced_product


def make_gen(family, num, weight, seed):
    if family == 'dihedral':
        return sample_dihedral(num, m_range=(30, 175), weight=weight, seed=seed)
    if family == 'metacyclic':
        return sample_metacyclic(num, order_range=(120, 350), weight=weight, seed=seed)
    if family == 'kasai':
        return sample_kasai_affine(num, q_range=(7, 19), weight=weight, seed=seed)
    if family == 'lp':
        return sample_lifted_product(num, order_range=(30, 350),
                                     weight_a=weight, weight_b=weight, seed=seed)
    if family == 'balanced':
        return sample_balanced_product(num, order_range=(4, 16), seed=seed)
    if family == 'bb':
        return sample_bb(num, weight=weight, seed=seed)
    raise SystemExit(f'unknown family {family}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True)
    ap.add_argument('--num', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trials', type=int, default=4000)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--weight', type=int, default=3)
    ap.add_argument('--min-k', type=int, default=6)
    ap.add_argument('--min-d', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    gen = make_gen(args.family, args.num, args.weight, args.seed)
    t = time.time()
    recs = screen(gen, min_k=args.min_k, min_d=args.min_d, trials=args.trials,
                  seed=args.seed, backend='auto', threads=args.threads)
    json.dump({'family': args.family, 'num': args.num, 'trials': args.trials,
               'seed': args.seed, 'weight': args.weight, 'records': recs},
              open(args.out, 'w'))
    print(f'{args.family} w{args.weight} num={args.num} seed={args.seed} '
          f'-> {len(recs)} survivors in {time.time()-t:.0f}s', flush=True)
    for r in recs[:20]:
        print(f"  [[{r['n']},{r['k']},{r['d']}]] w{r['w']} eff={r['efficiency']:.2f} "
              f"{json.dumps(r['spec'])[:90]}", flush=True)


if __name__ == '__main__':
    main()
