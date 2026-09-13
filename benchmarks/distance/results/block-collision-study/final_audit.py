"""Cross-grid preservation and portable-witness verification, without searching."""
import hashlib
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'benchmarks/distance'))
from run import gf2,np
from corpus import supports_matrix

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

signatures={}
for grid in ('screen-2s','confirmation-10s','exact-blocks'):
    audit=read(HERE/grid/'audit-block-collision.json')
    assert audit['status']=='passed'
    for entry in audit['initialization']:
        key=entry['case'],entry['side']
        signature=entry['signature_sha256']
        assert signatures.setdefault(key,signature)==signature
    for source in ('environment.json','audit.json'):
        metadata=read(HERE/grid/source)
        for kind in ('source_hashes','binary_hashes','supplemental_source_hashes'):
            for name,sha in metadata.get(kind,{}).items():
                assert digest(ROOT/name)==sha,name
for winner in read(HERE/'witness-index.json'):
    file=HERE/winner['artifact']
    assert digest(file)==winner['sha256']
    doc=read(file)
    side=winner['side']
    own=supports_matrix(doc['checks'][side],doc['n'])
    opposite=supports_matrix(doc['checks']['Z' if side=='X' else 'X'],doc['n'])
    with np.load(HERE/'corpus'/f"{winner['case']}.npz") as data:
        assert np.array_equal(own,data['hx' if side=='X' else 'hz'])
        assert np.array_equal(opposite,data['hz' if side=='X' else 'hx'])
    vector=np.zeros(doc['n'],dtype=np.int8)
    vector[winner['support']]=1
    assert gf2.commutes(vector,opposite)
    assert not gf2.in_rowspace(vector,own)
    assert int(vector.sum())==winner['weight']
previous=HERE.parent/'large-structure-study'
old_files=0
for line in (previous/'SHA256SUMS').read_text().splitlines():
    sha,name=line.split('  ',1)
    assert digest(previous/name)==sha,name
    old_files+=1
result=dict(status='passed',cross_grid_initializations=len(signatures),portable_witnesses=6,
    prior_study_files_unchanged=old_files,all_measured_sources_binaries_and_supplemental_sources_unchanged=True)
(HERE/'final-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
