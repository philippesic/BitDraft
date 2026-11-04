"""Audit retained native Q4 teacher rows against consumed actual FIT histories.

This checks eligibility and probability accounting, not teacher interchangeability,
KD training gradients, a new target precision, or a serving-speed improvement.
"""
from pathlib import Path
import collections
import hashlib
import json
import sys
import numpy as np
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'third_party/llama.cpp/gguf-py'))
from gguf import GGUFReader


def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve()
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha)


def main():
    base=P/'runs/w1a8-modern-generalization-capture-20261103-02';out=P/'experiments/design-readiness-20261104/teacher-coverage.json';assert not out.exists()
    up=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01/union01/manifest.json');u=read(up)
    mapping=next(t for t in GGUFReader(str(P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf')).tensors if t.name=='d2t').data.reshape(-1).astype(np.int64)
    support=set(int(i) for i in mapping)
    rows=[];totals=collections.Counter();pins=[];mass={1:[],2:[]};label_mismatches=[]
    for group in u['groups']:
        root=base/group['id']
        if group['role']!='fit' or not (root/'packet.json').exists():continue
        packet=read(root/'packet.json');meta=read(root/'capture/capture.json');admission=read(root/'admission.json')
        assert packet['target']['sha256']=='3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630'
        assert meta['settings']['consistent']==32 and meta['settings']['kv']=='F16'
        for ref in [packet['capture'],packet['arrays']['logits.f32']]:assert pin(ref['path'])==ref;pins.append(ref)
        logits=np.memmap(packet['arrays']['logits.f32']['path'],dtype='<f4',mode='r',shape=tuple(meta['logits_shape']))
        positions={p:i for i,p in enumerate(meta['logit_positions'])};eligible=set(admission['eligible_row_ids']);excluded=set(admission['excluded_row_ids'])
        assert not eligible&excluded
        group_counts=collections.Counter();used=set()
        for w in u['windows']:
            if w['group']!=group['id']:continue
            for depth,label in enumerate(w['labels']):
                pos=meta['prompt_tokens']+w['response_label_start']+depth-1
                if pos not in positions:continue
                index=positions[pos];prefix=[meta['input_tokens'][0]]+w['prefix_tokens']+[w['seed_token']]+w['labels'][:depth]
                assert prefix==meta['input_tokens'][:pos+1],(w['id'],depth)
                values=np.asarray(logits[index],dtype=np.float64);assert np.isfinite(values).all()
                argmax=int(values.argmax());is_match=argmax==label
                if not is_match:
                    assert index in excluded and index not in eligible
                    label_mismatches.append(dict(window=w['id'],depth=depth,capture_row=index,label=label,teacher_argmax=argmax,previously_excluded=True))
                if index in eligible:
                    assert is_match;group_counts['eligible_window_label_occurrences']+=1
                    group_counts['currently_supported_and_reachable_occurrences']+=bool(label in support and w['native_supervision_reachable_mask'][depth])
                    if index not in used:
                        for temperature in mass:
                            z=values/temperature;prob=np.exp(z-z.max());prob/=prob.sum();inside=float(prob[mapping].sum());assert 0<=inside<=1+1e-12;mass[temperature].append(inside)
                        used.add(index)
                else:group_counts['excluded_window_label_occurrences']+=1
        group_counts['unique_eligible_teacher_rows_joined']=len(used);totals.update(group_counts);rows.append(dict(group=group['id'],counts=dict(group_counts),admission=pin(root/'admission.json'),packet=pin(root/'packet.json')))
    assert len(rows)==24
    summary={str(t):dict(rows=len(v),mean_head_probability_mass=float(np.mean(v)),min_head_probability_mass=float(np.min(v)),max_head_probability_mass=float(np.max(v))) for t,v in mass.items()}
    result=dict(status='CONDITIONAL_NATIVE_Q4_TEACHER_SOURCE_JOIN_AND_PROBABILITY_ACCOUNTING_AUDIT',window_manifest=pin(up),groups=rows,totals=dict(totals),head_probability_mass_by_temperature=summary,known_excluded_label_mismatches=label_mismatches,source_pins=pins,scope='24 original FIT groups only; 24 added FIT groups have no corresponding retained dense capture in this source. No DEV/SEALED opened. Existing native-row exclusions enforced, not silently relabeled. Conditional captures have different execution histories from actual serving traces; feature drift is recorded separately. This is not admission of a KD training objective or evidence of a feature win.',remaining=['Declare conditional-head versus full-vocabulary objective; outside-head probability cannot be represented by the current head.','Authenticate any new serving-state probability teacher and quantify gradient impact before calling sources interchangeable.','Implement and qualify teacher integration, loss gradients/noop/resume/export and matched native serving if selected.'])
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'totals':dict(totals),'mass':summary,'known_excluded_occurrences':len(label_mismatches)}))
if __name__=='__main__':main()
