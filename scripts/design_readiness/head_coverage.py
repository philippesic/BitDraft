"""Current fixed-head label coverage on already-consumed data; no shortlist search."""
from pathlib import Path
import collections
import hashlib
import json
import sys
import numpy as np
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'third_party/llama.cpp/gguf-py'))
from gguf import GGUFReader


def pin(path):
    path=Path(path).resolve()
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha)
def read(path):return json.loads(Path(path).read_text())

def main():
    out=P/'experiments/design-readiness-20261104/head-coverage.json'
    assert not out.exists()
    model=P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf'
    ref=pin(model);assert ref['sha256']=='6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048'
    reader=GGUFReader(str(model));mapping=next(t for t in reader.tensors if t.name=='d2t').data.reshape(-1)
    supported=set(int(i) for i in mapping);assert len(supported)==50000 and min(supported)>=0 and max(supported)<248320
    union=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01/union01/manifest.json');u=read(union)
    windows=collections.defaultdict(collections.Counter)
    for w in u['windows']:
        assert w['role'] in ('fit','development')
        c=windows[w['role']]
        for depth,(token,reached) in enumerate(zip(w['labels'],w['native_supervision_reachable_mask'])):
            c['labels']+=1;c['supported_labels']+=token in supported;c['static_parent_reachable_labels']+=bool(reached);c['supported_and_reachable_labels']+=bool(reached and token in supported)
            c['depth%d_labels'%depth]+=1;c['depth%d_unsupported'%depth]+=token not in supported
    dev=P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json'
    groups=[]
    for item in read(dev)['rows']:
        r=item['reference'];assert pin(r['path'])==r;v=read(r['path']);tokens=v['tokens']
        groups.append(dict(id=item['id'],tokens=len(tokens),unsupported=sum(t not in supported for t in tokens),source=r))
    natural=P/'runs/w1a8-modern-protected-20261103-01/natural-replay-confirmation01/root-analysis.json';nr=read(natural);result=nr['result'];assert pin(result['path'])==result;records=read(result['path'])['records']
    natural_groups=[]
    for row in records:
        if row['arm']!='target_only' or row['repetition']!=0 or row.get('warmup') or row.get('kind')=='quality':continue
        # The full runner records quality separately; accept only measured performance rows.
        if 'full_http_wall_seconds' not in row:continue
        r=row['response'];assert pin(r['path'])==r;v=read(r['path']);tokens=v['tokens']
        natural_groups.append(dict(id=row['id'],tokens=len(tokens),unsupported=sum(t not in supported for t in tokens),source=r))
    assert len(groups)==12 and len(natural_groups)==24,(len(groups),len(natural_groups))
    summary={name:dict(prompts=len(rows),tokens=sum(r['tokens'] for r in rows),unsupported_tokens=sum(r['unsupported'] for r in rows)) for name,rows in [('consumed_dev12',groups),('consumed_natural24',natural_groups)]}
    for r in summary.values():r['token_membership_fraction']=1-r['unsupported_tokens']/r['tokens']
    report=dict(status='CURRENT_FIXED_HEAD_CONSUMED_LABEL_COVERAGE_AUDIT',model=ref,absolute_mapping_sha256=hashlib.sha256(np.asarray(mapping,dtype='<i8').tobytes()).hexdigest(),vocab_size=248320,head_classes=50000,window_manifest=pin(union),window_label_occurrences=dict(windows),summary=summary,groups=groups,natural_groups=natural_groups,scope='Current head only; no alternative vocabulary, architecture or precision tested. Window occurrences overlap and are not unique tokens. Membership does not predict acceptance or throughput. All evaluation responses were already consumed; this opens no fresh prompts.')
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
