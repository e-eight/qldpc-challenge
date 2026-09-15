import json, sys

W = {'weight-4': 0, 'weight-6': 1, 'weight-8': 2, 'weight-9plus': 3}
board = json.load(open('/tmp/board_index.json'))

for cell_wc, name in ((1, 'weight-6 (~all locality)'), (2, 'weight-8 (~all locality)')):
    rows = [b for b in board if W.get(b['weight_class'], 9) <= cell_wc]
    front = []
    for a in rows:
        dominated = any(
            b is not a and b['n'] <= a['n'] and b['k'] >= a['k'] and b['d'] >= a['d']
            and b['w'] <= a['w']
            and (b['n'] < a['n'] or b['k'] > a['k'] or b['d'] > a['d'] or b['w'] < a['w'])
            for b in rows)
        if not dominated:
            front.append(a)
    front.sort(key=lambda r: (r['n'], -r['k']))
    print(f'=== {name}: {len(rows)} entries, {len(front)} frontier points ===')
    for r in front:
        print(f"  [[{r['n']},{r['k']},{r['d']}]] w{r['w']} K={r['k']*r['d']**2/r['n']:.1f} "
              f"{r['weight_class']}/{r['locality_class']}  {r['file']}")
    print()
