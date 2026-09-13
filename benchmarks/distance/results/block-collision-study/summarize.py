"""Aggregate audited grids; copy already persisted witnesses, without searching."""
import hashlib
import json
import shutil
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GRIDS = ('screen-2s', 'confirmation-10s', 'exact-blocks')

def read(path):
    return json.loads(path.read_text())

def write(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2) + '\n')

all_records = []
accounting = []
cells = []
winners = {}
for grid in GRIDS:
    directory = HERE / grid
    audit = read(directory / 'audit-block-collision.json')
    assert audit['status'] == 'passed'
    env = read(directory / 'environment.json')
    records = [read(p) for p in sorted(directory.glob('*/*/result.json'))]
    assert len(records) == audit['configurations']
    accounting.append(dict(grid=grid, configurations=len(records),
        allocated_search_seconds=len(records)*env['seconds_per_code'],
        actual_search_seconds=sum(w['search_seconds'] for r in records for ws in r['workers'].values() for w in ws),
        validation_seconds=sum(r['validation_seconds'] for r in records),
        saved_documents=audit['saved_witness_documents_checked']))
    for kind in ('source_hashes', 'binary_hashes'):
        for path, digest in env[kind].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    for r in records:
        all_records.append((grid, r))
        sequence = 0
        for side in ('X', 'Z'):
            seen = set()
            for w in r['workers'][side]:
                for event in w['events']:
                    support = tuple(event['support'])
                    if support in seen:
                        continue
                    seen.add(support)
                    if event['seconds'] <= r['budget_seconds']/2:
                        key = (r['case'], side)
                        if key not in winners or event['weight'] < winners[key]['weight']:
                            winners[key] = dict(grid=grid, case=r['case'], side=side, method=r['method'],
                                seed=r['seed'], sequence=sequence, **event)
                    sequence += 1
        assert sequence == r['saved_candidates']
    for c in read(directory / 'summary.json')['cells']:
        rs = [r for r in records if r['case']==c['case'] and r['method']==c['method']]
        global_collision_improvements = 0
        first_final = []
        for r in rs:
            final = min(v['best_in_budget'] for v in r['sides'].values())
            times = []
            for side in ('X', 'Z'):
                incumbent = 1000000
                for e in r['workers'][side][0]['events']:
                    if e['seconds'] > r['budget_seconds']/2:
                        continue
                    if e['weight'] < incumbent:
                        global_collision_improvements += e['stage'].endswith(':collision4')
                        incumbent = e['weight']
                    if e['weight'] == final:
                        times.append(e['seconds'] + (r['budget_seconds']/2 if side=='Z' else 0))
            first_final.append(min(times))
        sessions = [s for s in audit['sessions'] if s['case']==c['case'] and s['method']==c['method']]
        cells.append(dict(grid=grid, **c, first_final_scheduled_seconds=first_final,
            sector_incumbent_collision_improvements=global_collision_improvements,
            reductions=sum(s['reductions'] for s in sessions),
            collision_matches=sum(s['collisions'] for s in sessions),
            workspace_bytes_range=[min(s['workspace_bytes'] for s in sessions),max(s['workspace_bytes'] for s in sessions)] if sessions else None))
portable=[]
for (case, side), winner in sorted(winners.items()):
    source_dir = ROOT / 'research/candidates/distance-benchmark' / case / f"{winner['grid']}-{winner['method']}-s{winner['seed']}" / str(winner['sequence'])
    sources = list(source_dir.glob('*.json'))
    assert len(sources) == 1
    document = read(sources[0])
    assert document['distance'][side]['witness'] == sorted(winner['support'])
    name = f"{case}-{side}-w{winner['weight']}.json"
    shutil.copyfile(sources[0], HERE/name)
    portable.append(dict(winner, artifact=name, sha256=hashlib.sha256((HERE/name).read_bytes()).hexdigest()))
write('witness-index.json', portable)
write('accounting.json', accounting)
write('effects.json', cells)
new_bounds=[]
for c in read(HERE/'corpus/manifest.json')['cases']:
    target=min(map(len,c['reference'].values()))
    found=min(w['weight'] for w in portable if w['case']==c['id'])
    if found<target:
        new_bounds.append(dict(case=c['id'],reference=target,found=found))
second=[]
for c in cells:
    if c['grid']=='screen-2s' and c['case']!='684-10-101' and c['method'] in ('block-pairs','block-stern') and c['hits']==3:
        control=next(x for x in cells if x['grid']=='confirmation-10s' and x['case']==c['case'] and x['method']=='guided')
        if control['hits']<=1:
            second.append(dict(case=c['case'],method=c['method'],control_hits=control['hits']))
write('decision.json',dict(continuation_criterion_met=bool(new_bounds or second),new_lower_bounds=new_bounds,
    replicated_short_budget_advantage=second, source_and_binary_hashes_match=True,
    scope='Restricted-block progress on three affine codes; no claim of full-code exact distance or general scaling.'))
print(json.dumps(dict(accounting=accounting, new_bounds=new_bounds),indent=2))
