import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
python=str(ROOT/'.venv-benchmark/bin/python')
for grid in ('screen-2s','confirmation-10s','exact-blocks'):
    env=json.loads((HERE.parent/'block-collision-study'/grid/'environment.json').read_text())
    for kind in ('source_hashes','binary_hashes'):
        for name,digest in env[kind].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for grid,seconds,seed,seeds,extra in [
    ('exact-affine',4,1830,1,['--methods','exact-components','--cases','684-12-73','684-8-85']),
    ('screen-2s',2,1800,3,[]),
    ('confirmation-10s',10,1810,3,[]),
    ('permuted-2s',2,1820,1,['--corpus',str(HERE/'permuted-corpus'),'--methods','guided','auto-components'])]:
    with (HERE/f'{grid}-runner.log').open('w') as log:
        subprocess.run([python,'benchmarks/distance/study_components.py','--output',str(HERE/grid),'--seconds',str(seconds),'--seed-start',str(seed),'--seeds',str(seeds),*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (HERE/f'{grid}-audit.log').open('w') as log:
        subprocess.run(['taskset','-c','7',python,'benchmarks/distance/audit_components.py',str(HERE/grid)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(grid,'complete and audited',flush=True)
