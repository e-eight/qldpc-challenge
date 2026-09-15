import sys, os, json
import numpy as np
BASE = '/home/soham/Projects/unitaryfoundation/qldpc-challenge'
sys.path[:0] = [BASE + '/research/kit', BASE + '/verify']
from submit import make_submission, save_submission
from css import compute_k

os.makedirs(BASE + '/research/candidates', exist_ok=True)

CANDS = [
    ('/tmp/final_lp672.npz', dict(
        name='[[672,6,42]] lifted product on the metacyclic group Z_3 x| Z_112',
        construction=('Lifted product (two-block group algebra) on the metacyclic group '
                      'Z_3 x| Z_112 (r=2, order 336; Cayley table built by '
                      'research/kit/group_algebra.py:metacyclic(3,112,2)), support_a = '
                      '{128, 211, 257} on the left-regular block and support_b = '
                      '{92, 130, 269} on the right-regular block. Max check weight 6.'),
        family='lifted-product',
        references=['arXiv:1904.02703', 'arXiv:2306.16400', 'arXiv:2111.03654'])),
    ('/tmp/final_lp648.npz', dict(
        name='[[648,4,40]] lifted product on the metacyclic group Z_3 x| Z_108',
        construction=('Lifted product (two-block group algebra) on the metacyclic group '
                      'Z_3 x| Z_108 (r=2, order 324; group_algebra.metacyclic(3,108,2)), '
                      'support_a = {4, 123, 250}, support_b = {64, 210, 238}. '
                      'Max check weight 6.'),
        family='lifted-product',
        references=['arXiv:1904.02703', 'arXiv:2306.16400', 'arXiv:2111.03654'])),
    ('/tmp/kasai_1a1cc63eb8604a10.npz', dict(
        name='[[684,4,88]] two-block group algebra on the affine group Aff(F_19)',
        construction=('Two-block group algebra (affine generalized bicycle) on the affine '
                      'group Aff(F_19) = Z_19 x| Z_18 (order 342, n = 684), support_a = '
                      '{234, 269, 233, 249}, support_b = {28, 205, 318, 53}. '
                      'Max check weight 8.'),
        family='generalized-bicycle',
        references=['arXiv:1904.02703', 'arXiv:2111.03654'])),
]

for path, kw in CANDS:
    z = np.load(path)
    HX, HZ = z['HX'], z['HZ']
    doc = make_submission(HX, HZ, authors=['@e-eight'], confidence='upper_bound',
                          date='2026-09-15', **kw)
    n, k = doc['n'], doc['k']
    d = doc['distance']['d']
    out = f'{BASE}/research/candidates/{n}-{k}-{d}.json'
    save_submission(doc, out)
    print(f'staged [[{n},{k},{d}]] -> {out}', flush=True)
