"""Independently verify algebraic certificates without distance search."""
import hashlib,json,sys,tarfile,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'benchmarks/distance'))
from run import gf2,np
from common import matrix_hash,sha256
from strategy_prototypes.algebra_diagnostic.algebra import packed_rows,poly_div,poly_mul,irreducible


def read(path):return json.loads(path.read_text())

def main():
    root=HERE/'analysis';env=read(root/'environment.json');complete=read(root/'completed.json')
    archived={}
    with tarfile.open(root/'sources.tar.gz') as archive:
        for member in archive:
            if member.isfile():archived[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert archived==env['source_hashes']
    for name,digest in archived.items():assert sha256(ROOT/name)==digest,name
    manifest={c['id']:c for c in read(root/'manifest.json')['cases']};matrices={}
    for case in manifest.values():
        with np.load(root/'matrices'/case['file']) as data:hx,hz=data['hx'],data['hz']
        assert matrix_hash(hx,hz)==case['matrix_sha256'];matrices[case['id']]=(hx,hz)
    equation_count=0;spaces=0;cached={}
    for row in read(root/'decompositions.json'):
        hx,hz=matrices[row['case']];own,opposite=(hx,hz) if row['side']=='X' else (hz,hx)
        coords=row['coordinates'];h=opposite[:,coords];g=gf2.kernel_basis(h)
        index={q:i for i,q in enumerate(coords)}
        masks=[sum(1<<index[q] for q in s['coordinates']) for s in row['exact_components']]
        assert sorted(q for s in row['exact_components'] for q in s['coordinates'])==coords
        assert sum(masks)==(1<<len(coords))-1
        basis={};packed_g=packed_rows(g)
        for check in packed_rows(h):
            for generator in packed_g:
                constraint=check&generator
                if not constraint:continue
                equation_count+=1
                assert all((constraint&mask).bit_count()%2==0 for mask in masks)
                while constraint:
                    pivot=(constraint&-constraint).bit_length()-1
                    if pivot not in basis:basis[pivot]=constraint;break
                    constraint^=basis[pivot]
        assert len(coords)-len(basis)==len(masks),'Mask space has a finer split'
        for component in row['exact_components']:
            signature=(row['case'],row['side'],tuple(component['coordinates']))
            if signature in cached:assert cached[signature]==(component['dimension'],component['logical_rank']);continue
            support=component['coordinates'];small=gf2.kernel_basis(opposite[:,support])
            lifted=np.zeros((len(small),own.shape[1]),dtype=np.int8);lifted[:,support]=small
            logical=gf2.rank(np.vstack([own,lifted]))-gf2.rank(own)
            assert (len(small),logical)==(component['dimension'],component['logical_rank'])
            cached[signature]=(len(small),logical);spaces+=1
    checked_cuts=0
    for entry in read(root/'cuts.json'):
        hx,hz=matrices[entry['case']];h=hz if entry['side']=='X' else hx
        if 'profiles' in entry:
            domain=entry['domain_coordinates'];rank=gf2.rank(h[:,domain]);n=len(domain)
            for p in entry['profiles']:
                order=[domain[q] for q in p['order']]
                assert sorted(order)==sorted(domain)
                assert p['coordinates']==order[:p['cut']]
                for pos in {0,n//4,n//2,3*n//4,n,p['cut']}:
                    actual=gf2.rank(h[:,order[:pos]])+gf2.rank(h[:,order[pos:]])-rank
                    assert actual==p['profile'][pos];checked_cuts+=1
                edge=(n+3)//4
                assert p['coupling']==min(p['profile'][edge:n-edge+1])
        else:
            s=entry['coordinates'];t=sorted(set(range(h.shape[1]))-set(s))
            assert entry['coupling']==gf2.rank(h[:,s])+gf2.rank(h[:,t])-gf2.rank(h);checked_cuts+=1
    symmetries=read(root/'symmetries.json');accepted=0
    for case in manifest.values():
        proposals=[s for s in symmetries if s['case']==case['id']]
        assert {s['block_length'] for s in proposals}=={l for l in range(3,case['n']+1) if case['n']%l==0}
        hx,hz=matrices[case['id']]
        for s in proposals:
            l=s['block_length'];p=[(q//l)*l+(q+1)%l for q in range(case['n'])]
            x=np.array_equal(gf2.rref(hx)[0],gf2.rref(hx[:,p])[0]);z=np.array_equal(gf2.rref(hz)[0],gf2.rref(hz[:,p])[0])
            assert (x,z,x and z)==(s['X'],s['Z'],s['accepted']);accepted+=int(x and z)
    for m in read(root/'polynomials.json'):
        if m['status']!='verified':continue
        a,b,d,u,v=[int(m[k],16) for k in ('a','b','gcd_ab','bezout_u','bezout_v')]
        assert poly_mul(u,a)^poly_mul(v,b)==d
        product=1
        for f in m['factors']:
            polynomial=int(f['polynomial'],16);assert irreducible(polynomial)
            product=poly_mul(product,polynomial)
            assert (poly_div(int(m['gcd_common'],16),polynomial)[1]==0)==f['divides_common']
        assert product==int(m['modulus'],16)
    shutil.copyfile(ROOT/'benchmarks/distance/results/component-study/permuted-corpus/transforms.json',HERE/'transforms.json')
    supplementary=[HERE/'qc_module.py',HERE/'test_qc.py',Path(__file__).resolve()]
    supplement_hashes={str(p.relative_to(ROOT)):sha256(p) for p in supplementary}
    with tarfile.open(HERE/'supplementary-sources.tar.gz','w:gz') as archive:
        for p in supplementary:archive.add(p,arcname=str(p.relative_to(ROOT)))
    prior=0
    for name in ('component-study','block-collision-study','large-structure-study'):
        previous=ROOT/'benchmarks/distance/results'/name
        for line in (previous/'SHA256SUMS').read_text().splitlines():
            digest,path=line.split('  ',1);assert sha256(previous/path)==digest;prior+=1
        for envfile in previous.glob('*/environment.json'):
            e=read(envfile)
            for category in ('source_hashes','binary_hashes'):
                for name,digest in e.get(category,{}).items():assert sha256(ROOT/name)==digest,name
    result=dict(status='passed',mask_equations_checked=equation_count,component_spaces_checked=spaces,selected_cut_ranks_checked=checked_cuts,
        symmetry_proposals_checked=len(symmetries),accepted_symmetries=accepted,prior_artifacts_unchanged=prior,
        no_distance_search=True,supplementary_source_hashes=supplement_hashes,completed_diagnostic=complete)
    (HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
