"""Read-only component dimensions and localization of already saved search exports."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'benchmarks/distance'))
from run import np
from strategy_prototypes.large_structure.adapter import native
from study_strategies import ris_native

HERE = Path(__file__).resolve().parent
rows = []
for file in sorted((HERE / 'corpus').glob('*.npz')):
    with np.load(file) as data:
        hx, hz = data['hx'], data['hz']
    for side, own, opposite in [('X', hx, hz), ('Z', hz, hx)]:
        prepared = ris_native.Prepared(own, opposite)
        for half in range(2):
            coords = list(range(half * 342, (half + 1) * 342))
            parent = {q: q for q in coords}
            def find(q):
                while parent[q] != q:
                    parent[q] = parent[parent[q]]
                    q = parent[q]
                return q
            for check in opposite[:, coords]:
                support = [coords[i] for i in np.flatnonzero(check)]
                for q in support[1:]:
                    parent[find(q)] = find(support[0])
            groups = {}
            for q in coords:
                groups.setdefault(find(q), []).append(q)
            for component in sorted(groups.values(), key=lambda x: x[0]):
                space = native.Space(opposite, prepared.logicals, [[q] for q in component])
                events = []
                for result in (HERE / 'screen-2s' / file.stem).glob('block-*/result.json'):
                    r = json.loads(result.read_text())
                    for e in r['workers'][side][0]['events']:
                        if e['seconds'] <= 1 and set(e['support']) <= set(component):
                            events.append(e['weight'])
                rows.append(dict(case=file.stem, side=side, half=half, coordinates=component,
                    columns=len(component), dimension=space.dimension, logical_rank=space.logical_rank,
                    best_existing_export=min(events, default=None)))
(HERE / 'component-diagnostic.json').write_text(json.dumps(rows, indent=2) + '\n')
for row in rows:
    print({k: v for k, v in row.items() if k != 'coordinates'})
