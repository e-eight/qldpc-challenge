import hashlib,json,subprocess
from pathlib import Path
repo=Path('/home/exedev/qldpc-challenge')
root=repo/'benchmarks/distance/results/block-collision-study'
python=str(repo/'.venv-benchmark/bin/python')
for grid in ('screen-2s','confirmation-20s','finite-order6'):
 env=json.loads((repo/'benchmarks/distance/results/large-structure-study'/grid/'environment.json').read_text())
 for kind in ('source_hashes','binary_hashes'):
  for name,digest in env[kind].items():
   assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
for grid,seconds,seed,seeds,methods in [('screen-2s',2,1700,3,[]),('confirmation-10s',10,1710,3,[]),('exact-blocks',4,1720,1,['--methods','block-exact'])]:
 with (root/f'{grid}-runner.log').open('w') as log:
  subprocess.run([python,'benchmarks/distance/study_block_collision.py','--output',str(root/grid),'--seconds',str(seconds),'--seed-start',str(seed),'--seeds',str(seeds),*methods],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (root/f'{grid}-audit.log').open('w') as log:
  subprocess.run(['taskset','-c','7',python,'benchmarks/distance/audit_block_collision.py',str(root/grid)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(grid,'complete and audited',flush=True)
print('Bounded block/collision experiment complete',flush=True)
