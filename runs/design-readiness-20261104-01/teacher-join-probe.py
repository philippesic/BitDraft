from pathlib import Path
import json,numpy as np
P=Path('/home/philip/binary-eagle-decoding');u=json.loads(Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01/union01/manifest.json').read_text());m=json.loads(Path(u['members'][0]['manifest']['path']).read_text());caps=P/'runs/w1a8-modern-generalization-capture-20261103-02';rows=[]
for g in u['groups']:
 if g['role']!='fit' or not (caps/g['id']/'capture/capture.json').exists():continue
 meta=json.loads((caps/g['id']/'capture/capture.json').read_text());known={pos:i for i,pos in enumerate(meta['logit_positions'])};response=json.loads(Path(g['response']['path']).read_text());mismatches=0;matches=0;checkedfeatures=0;maxrel=0.;teacher=np.memmap(caps/g['id']/'capture/logits.f32',dtype='<f4',mode='r',shape=tuple(meta['logits_shape']))
 for w in [w for w in u['windows'] if w['group']==g['id']]:
  for d,label in enumerate(w['labels']):
   pos=meta['prompt_tokens']+w['response_label_start']+d-1
   if pos not in known:continue
   actual=[meta['input_tokens'][0]]+w['prefix_tokens']+[w['seed_token']]+w['labels'][:d]
   assert actual==meta['input_tokens'][:pos+1],(w['id'],d,pos,len(actual))
   assert label==response['tokens'][w['response_label_start']+d]
   top=int(teacher[known[pos]].argmax());matches+=top==label;mismatches+=top!=label
  if checkedfeatures<2 and w['anchor'] in known:
   raw=m['raw_rows'][w['seed_raw_row']];spec=m['raw_files'][raw['file']];a=np.memmap(spec['path'],dtype='<f4',mode='r',shape=tuple(spec['shape']))[raw['row']];tp=raw['target_position'];b=np.concatenate([np.memmap(caps/g['id']/f'capture/layer-{k}.f32',dtype='<f4',mode='r',shape=tuple(meta['feature_shape']))[tp] for k in (2,16,29)]);maxrel=max(maxrel,float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)));checkedfeatures+=1
 rows.append(dict(group=g['id'],matching_teacher_labels=matches,mismatched_teacher_labels=mismatches,feature_checks=checkedfeatures,max_relative_feature_error=maxrel))
print(json.dumps(rows,indent=2));(P/'runs/design-readiness-20261104-01/teacher-join-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
