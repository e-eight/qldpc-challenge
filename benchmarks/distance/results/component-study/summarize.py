"""Summarize audited runs and preserve portable copies; no witness search."""
import hashlib,json,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'benchmarks/distance'))
from run import gf2,np
from common import matrix_hash
from corpus import supports_matrix

def read(p):return json.loads(p.read_text())
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
GRIDS=('exact-affine','screen-2s','confirmation-10s','permuted-2s')
cells=[];accounts=[];winners={};initial={};all_records=[]
for grid in GRIDS:
    directory=HERE/grid
    audit=read(directory/'audit-components.json');assert audit['status']=='passed'
    env=read(directory/'environment.json')
    assert env['threads']==1 and all(p['num_threads']==1 for p in env['threadpools'])
    for source in ('environment.json','audit.json'):
        metadata=read(directory/source)
        for kind in ('source_hashes','binary_hashes','supplemental_source_hashes'):
            for name,sha in metadata.get(kind,{}).items():assert digest(ROOT/name)==sha,name
    for entry in audit['initialization']:
        key=entry['case'],entry['side'];signature=entry['signature_sha256']
        assert initial.setdefault(key,signature)==signature
    records=[read(p) for p in sorted(directory.glob('*/*/result.json'))]
    assert len(records)==audit['configurations']
    accounts.append(dict(grid=grid,configurations=len(records),allocated_search_seconds=len(records)*env['seconds_per_code'],
        actual_search_seconds=sum(w['search_seconds'] for r in records for ws in r['workers'].values() for w in ws),
        validation_seconds=sum(r['validation_seconds'] for r in records),saved_documents=audit['saved_documents']))
    for r in records:
        all_records.append((grid,r));sequence=0
        for side in ('X','Z'):
            seen=set()
            for w in r['workers'][side]:
                for e in w['events']:
                    key=tuple(e['support'])
                    if key in seen:continue
                    seen.add(key)
                    index=(r['case'],side)
                    if e['seconds']<=r['budget_seconds']/2 and (index not in winners or e['weight']<winners[index]['weight']):
                        winners[index]=dict(grid=grid,case=r['case'],side=side,method=r['method'],seed=r['seed'],sequence=sequence,**e)
                    sequence+=1
        assert sequence==r['saved_candidates']
    for cell in read(directory/'summary.json')['cells']:
        rs=sorted([r for r in records if r['case']==cell['case'] and r['method']==cell['method']],key=lambda r:r['seed'])
        component_bests=[];initial_bests=[];fallback_bests=[];eligible=[];setup=[];stage_wins=[]
        for r in rs:
            grouped={'component':[],'initial':[],'fallback':[]};spaces=[]
            for side,ws in r['workers'].items():
                c=ws[0]['counters'];spaces.extend(c.get('spaces',[]));setup.append(c.get('setup_seconds',0))
                for e in ws[0]['events']:
                    if e['seconds']<=r['budget_seconds']/2:
                        group='component' if e['stage'].startswith('component') else ('fallback' if e['stage']=='guided' else 'initial')
                        grouped[group].append(e['weight'])
            bs={k:min(v,default=None) for k,v in grouped.items()}
            component_bests.append(bs['component']);initial_bests.append(bs['initial']);fallback_bests.append(bs['fallback'])
            eligible.append(sum('stats' in s for s in spaces))
            stage_wins.append(bs['component'] is not None and bs['component']<min(bs['initial'] or 100000,bs['fallback'] or 100000))
        cells.append(dict(grid=grid,**cell,component_weights=component_bests,initial_weights=initial_bests,
            fallback_weights=fallback_bests,eligible_space_counts=eligible,component_unique_best=stage_wins,
            setup_seconds_range=[min(setup),max(setup)]))
portable=[]
for (case,side),w in sorted(winners.items()):
    source=ROOT/'research/candidates/distance-benchmark'/case/f"{w['grid']}-{w['method']}-s{w['seed']}"/str(w['sequence'])
    files=list(source.glob('*.json'));assert len(files)==1
    doc=read(files[0]);assert doc['distance'][side]['witness']==w['support']
    own=supports_matrix(doc['checks'][side],doc['n']);opp=supports_matrix(doc['checks']['Z' if side=='X' else 'X'],doc['n'])
    v=np.zeros(doc['n'],dtype=np.int8);v[w['support']]=1
    assert gf2.commutes(v,opp) and not gf2.in_rowspace(v,own)
    name=f"{case}-{side}-w{w['weight']}.json";shutil.copyfile(files[0],HERE/name)
    portable.append(dict(w,artifact=name,sha256=digest(HERE/name)))
# Independently verify every permutation and remapped reference.
base={c['id']:c for c in read(HERE/'corpus/manifest.json')['cases']}
variants=read(HERE/'permuted-corpus/manifest.json')['cases'];transforms=read(HERE/'permuted-corpus/transforms.json')
for c in variants:
    t=transforms[c['id']];origin=base[t['source_case']]
    with np.load(HERE/'corpus'/origin['file']) as d:hx,hz=d['hx'],d['hz']
    cols=np.array(t['columns']);rx=np.array(t['rows_x']);rz=np.array(t['rows_z'])
    for p,n in [(cols,hx.shape[1]),(rx,len(hx)),(rz,len(hz))]:assert np.array_equal(np.sort(p),np.arange(n))
    with np.load(HERE/'permuted-corpus'/c['file']) as d:
        assert np.array_equal(d['hx'],hx[rx][:,cols]) and np.array_equal(d['hz'],hz[rz][:,cols])
        assert matrix_hash(d['hx'],d['hz'])==c['matrix_sha256']
    for side in ('X','Z'):assert sorted(int(cols[q]) for q in c['reference'][side])==sorted(origin['reference'][side])
geometry=read(HERE/'geometry-diagnostic.json')
exact_sessions=read(HERE/'exact-affine/audit-components.json')['sessions']
for case in ('684-12-73','684-8-85'):
    for side in ('X','Z'):
        expected={tuple(s['coordinates']) for s in geometry if s['case']==case and s['side']==side and 0<s['logical_rank'] and s['dimension']<=20}
        actual={tuple(s['coordinates']) for s in exact_sessions if s['case']==case and s['side']==side and s['stats']['enumeration_complete']}
        assert actual==expected,(case,side,'Incomplete component coverage')
prior_count=0
for previous in ('block-collision-study','large-structure-study'):
    root=HERE.parent/previous
    for line in (root/'SHA256SUMS').read_text().splitlines():
        sha,name=line.split('  ',1);assert digest(root/name)==sha,name;prior_count+=1
success=[]
for c in cells:
    if c['grid']!='screen-2s' or c['method']=='guided' or c['case'].startswith('684'):continue
    control=next(x for x in cells if x['grid']=='confirmation-10s' and x['method']=='guided' and x['case']==c['case'])
    if all(w<=.9*min(control['weights']) for w in c['weights']) and all(w is not None and w<=.9*min(control['weights']) for w in c['component_weights']):
        success.append(dict(case=c['case'],method=c['method'],weights=c['weights'],guided10=control['weights']))
new=[]
for case in base.values():
    target=min(map(len,case['reference'].values()))
    weight=min(w['weight'] for w in portable if w['case']==case['id'])
    if weight<target:new.append(dict(case=case['id'],reference=target,weight=weight))
write('effects.json',cells);write('accounting.json',accounts);write('witness-index.json',portable)
write('decision.json',dict(continuation_criterion_met=bool(success),non_affine_successes=success,new_witnessed_upper_bounds=new,
    scope='Fixed detector/proposals and dimension64 policy only; caps and missing proposals are coverage limitations.'))
write('final-audit.json',dict(status='passed',configurations=sum(a['configurations'] for a in accounts),saved_documents=sum(a['saved_documents'] for a in accounts),
    portable_witnesses=len(portable),permuted_cases_checked=len(variants),initial_signatures=len(initial),prior_artifacts_unchanged=prior_count,
    all_measured_sources_binaries_and_supplemental_sources_unchanged=True))
print(json.dumps(dict(accounting=accounts,successes=success,new_bounds=new),indent=2))
