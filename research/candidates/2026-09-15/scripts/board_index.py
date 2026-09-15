import sys, json, glob, os
sys.path.insert(0, '/home/soham/Projects/unitaryfoundation/qldpc-challenge/verify')
import qldpc_verify

out = []
for p in sorted(glob.glob('/tmp/wt-main/codes/*.json')):
    try:
        d = json.load(open(p))
        rep = qldpc_verify.verify(d, refute=False)
        comp = rep.get('computed', {})
        w = max(len(s) for s in d['checks']['X'] + d['checks']['Z'])
        out.append({'file': os.path.basename(p), 'n': d['n'], 'k': d['k'],
                    'd': d['distance']['d'], 'w': w,
                    'weight_class': comp.get('weight_class'),
                    'locality_class': comp.get('locality_class'),
                    'fingerprint': rep.get('fingerprint'),
                    'signature': rep.get('signature', {}).get('hash')})
    except Exception:
        pass
json.dump(out, open('/tmp/board_index.json', 'w'))
print('indexed', len(out), flush=True)
