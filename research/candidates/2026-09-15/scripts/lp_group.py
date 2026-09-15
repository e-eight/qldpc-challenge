import sys, json
import numpy as np
BASE = '/home/soham/Projects/unitaryfoundation/qldpc-challenge'
sys.path[:0] = [BASE + '/research/kit', BASE + '/verify']
import products
from search import fingerprint
from group_algebra import metacyclic, dihedral, cyclic_product

orig = products._build_group
state = {}


def wrapped(order, rng):
    g = orig(order, rng)
    state['last'] = (order, g)
    return g


products._build_group = wrapped
TARGETS = {'670ecbf0cc5275d4': 'lp672', 'd16f3f5fd4f0648d': 'lp648'}


def is_abelian(mul):
    return np.array_equal(mul, mul.T)


def identify(order, mul):
    if order <= 24:
        for m in range(2, 13):
            if 2 * m == order:
                d, _ = dihedral(m)
                if np.array_equal(d, mul):
                    return f'dihedral D_{m}'
    for n in range(3, order + 1):
        if order % n:
            continue
        k = order // n
        if k < 2:
            continue
        for r in range(2, n):
            if pow(r, k, n) == 1:
                mm, _ = metacyclic(n, k, r)
                if mm.shape == mul.shape and np.array_equal(mm, mul):
                    return f'metacyclic Z_{n} x| Z_{k} (r={r})'
    c, _ = cyclic_product(order)
    if np.array_equal(c, mul):
        return f'cyclic Z_{order}'
    return f'group of order {order} (unidentified){" [abelian]" if is_abelian(mul) else " [non-abelian]"}'


for spec, HX, HZ in products.sample_lifted_product(1500, order_range=(30, 350),
                                                  weight_a=3, weight_b=3, seed=606):
    fp = fingerprint(HX, HZ)
    if fp in TARGETS:
        order, mul = state['last']
        print(f"{TARGETS[fp]} fp={fp}: order={order} abelian={is_abelian(mul)} -> "
              f"{identify(order, mul)}", flush=True)
        print(f"   spec={json.dumps(spec)}", flush=True)
