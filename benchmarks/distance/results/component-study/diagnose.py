"""Inspect every documented proposal algebraically; do not search for witnesses."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'benchmarks/distance'))
from run import np
from strategy_prototypes.component_search.adapter import components,restriction,ris_native
rows=[]
for case in json.loads((HERE/'corpus/manifest.json').read_text())['cases']:
    with np.load(HERE/'corpus'/case['file']) as d:hx,hz=d['hx'],d['hz']
    for side,own,opposite in [('X',hx,hz),('Z',hz,hx)]:
        duals=ris_native.Prepared(own,opposite).logicals
        seen=set()
        for label,coords in [('whole',list(range(case['n'])))]+[(f'metadata{i}',b) for i,b in enumerate(case['blocks'])]:
            for group in components(opposite,coords):
                if tuple(group) in seen:continue
                seen.add(tuple(group))
                space=restriction.Space(opposite,duals,[[q] for q in group])
                rows.append(dict(case=case['id'],side=side,proposal=label,coordinates=group,
                    dimension=space.dimension,logical_rank=space.logical_rank,
                    reference_contained=set(case['reference'][side])<=set(group)))
(HERE/'geometry-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
for case in sorted({r['case'] for r in rows}):
    rs=[r for r in rows if r['case']==case]
    eligible=[r for r in rs if 0<r['logical_rank'] and r['dimension']<=64]
    print(case,'spaces',len(rs),'eligible',len(eligible),'dimensions',sorted({r['dimension'] for r in rs if r['logical_rank']}))
