import json

W = {'weight-4': 0, 'weight-6': 1, 'weight-8': 2, 'weight-9plus': 3}
L = {'local-2d-single': 0, 'local-2d-bilayer': 1, 'unrestricted': 2}
BOARD = json.load(open('/tmp/board_index.json'))


def wrank(w):
    return 0 if w <= 4 else 1 if w <= 6 else 2 if w <= 8 else 3


def dominators(n, k, d, w, locality_class):
    """Board entries that dominate (n,k,d,w) in the candidate's own cell."""
    wr = wrank(w)
    lr = L[locality_class]
    out = []
    for b in BOARD:
        if W.get(b['weight_class'], 9) <= wr and L.get(b['locality_class'], 9) <= lr:
            if (b['n'] <= n and b['k'] >= k and b['d'] >= d and b['w'] <= w
                    and (b['n'] < n or b['k'] > k or b['d'] > d or b['w'] < w)):
                out.append(b)
    return out


def report(tag, n, k, d, w, locality_class='local-2d-bilayer'):
    dom = dominators(n, k, d, w, locality_class)
    verdict = 'NON-DOMINATED' if not dom else f'dominated by {len(dom)}'
    print(f'{tag}: [[{n},{k},{d}]] w{w} {locality_class} K={k*d*d/n:.3f} -> {verdict}')
    for b in dom[:5]:
        print(f'    by [[{b["n"]},{b["k"]},{b["d"]}]] w{b["w"]} {b["weight_class"]}/{b["locality_class"]} {b["file"]}')
    return not dom
