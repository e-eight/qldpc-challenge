"""Supplemental polynomial-matrix check for verified multi-block cyclic actions."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'benchmarks/distance'))
from run import gf2,np
from strategy_prototypes.algebra_diagnostic.algebra import bezout,factor,irreducible,packed_rows,poly_div,poly_mul


def field_rank(matrix,modulus):
    if not irreducible(modulus):raise ValueError('Field modulus must be irreducible')
    a=[[poly_div(v,modulus)[1] for v in row] for row in matrix];rank=0
    if not a:return 0
    for column in range(len(a[0])):
        pivot=next((r for r in range(rank,len(a)) if a[r][column]),None)
        if pivot is None:continue
        a[rank],a[pivot]=a[pivot],a[rank]
        d,inverse,_=bezout(a[rank][column],modulus);assert d==1
        a[rank]=[poly_div(poly_mul(x,inverse),modulus)[1] for x in a[rank]]
        for r in range(len(a)):
            if r==rank or not a[r][column]:continue
            multiple=a[r][column]
            a[r]=[x^poly_div(poly_mul(multiple,y),modulus)[1] for x,y in zip(a[r],a[rank])]
        rank+=1
        if rank==len(a):break
    return rank


def row_orbits(h,length):
    n=h.shape[1];mask=(1<<length)-1;rows=set(packed_rows(h));unused=set(rows);base=[];sizes=[]
    while unused:
        seed=min(unused);word=seed;orbit=set()
        for _ in range(length):
            orbit.add(word);rotated=0
            for first in range(0,n,length):
                block=(word>>first)&mask
                rotated|=(((block<<1)&mask)|(block>>(length-1)))<<first
            word=rotated
        if not orbit<=rows:return None
        unused-=orbit;base.append([(seed>>first)&mask for first in range(0,n,length)]);sizes.append(len(orbit))
    return base,sizes


def main():
    root=HERE/'analysis'
    manifest={c['id']:c for c in json.loads((root/'manifest.json').read_text())['cases']}
    results=[]
    for symmetry in json.loads((root/'symmetries.json').read_text()):
        if not symmetry['accepted']:continue
        case=manifest[symmetry['case']];length=symmetry['block_length']
        with np.load(root/'matrices'/case['file']) as data:hx,hz=data['hx'],data['hz']
        ox,oz=row_orbits(hx,length),row_orbits(hz,length)
        if ox is None or oz is None:
            results.append(dict(case=case['id'],length=length,status='rowspace_symmetry_without_complete_row_orbits'));continue
        if length%2==0:
            results.append(dict(case=case['id'],length=length,status='repeated_root_ring_not_field_product'));continue
        bx,sx=ox;bz,sz=oz;modulus=(1<<length)|1
        factors=factor(modulus);degrees=[f.bit_length()-1 for f in factors]
        assert len(factors)==len(set(factors))
        sectors=[]
        for f,d in zip(factors,degrees):
            sectors.append(dict(factor=hex(f),degree=d,X_rank=field_rank(bx,f),Z_rank=field_rank(bz,f)))
        xrank=sum(s['degree']*s['X_rank'] for s in sectors);zrank=sum(s['degree']*s['Z_rank'] for s in sectors)
        assert xrank==gf2.rank(hx) and zrank==gf2.rank(hz)
        results.append(dict(case=case['id'],length=length,blocks=case['n']//length,status='verified',
            X_base=[[hex(p) for p in row] for row in bx],Z_base=[[hex(p) for p in row] for row in bz],
            X_orbit_sizes=sx,Z_orbit_sizes=sz,factors=sectors,X_binary_rank=xrank,Z_binary_rank=zrank,
            quantum_dimension=case['n']-xrank-zrank))
    (HERE/'qc-modules.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        print(r['case'],r['status'],[(s['degree'],s['X_rank'],s['Z_rank']) for s in r.get('factors',[])])
if __name__=='__main__':main()
