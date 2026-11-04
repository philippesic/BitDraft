"""Independent CPU recomputation of the tiny model screen and artifact integrity."""
from pathlib import Path
import sys,numpy as np
P=Path('/home/philip/binary-eagle-decoding');sys.path[:0]=[str(P/'scripts/contender_pipeline'),str(P/'third_party/llama.cpp/gguf-py')]
from runner import pin,read,atomic,require
from gguf import GGUFReader
R=P/'runs/binary-screen-20261104-01';inputs=read(R/'quality-inputs.json');require(len(inputs['selected'])==12 and len({r['group'] for r in inputs['selected']})==12,'group selection differs');labels=[]
for domain in ('code','reasoning','prose'):require(sum(r['domain']==domain for r in inputs['selected'])==4,'unbalanced TRAIN domains')
for item in inputs['selected']:
 require(pin(item['manifest']['path'])==item['manifest'],'manifest changed');m=read(item['manifest']['path']);w=next(w for w in m['windows'] if w['id']==item['window']);require(w['role']=='fit' and w['group_id']==item['group'],'not selected TRAIN group');labels+=w['labels']
reader=GGUFReader(str(P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf'));mapping=np.array(next(t for t in reader.tensors if t.name=='d2t').data,copy=True);reverse={int(t):i for i,t in enumerate(mapping)};classes=np.array([reverse[t] for t in labels]);labels=np.array(labels);sets={};max_ce_delta=0.;expected={'A8_control':0,'A4_all':9,'A1_all':9,'A4_body_A8_head':8,'A1_body_A8_head':8,'A4_mlp_only':3,'A4_all_A8_fc':8,'A4_body_A8_fc_head':7,'A1_mlp_only':3};first=None
for name in ['quality-screen','quality-followup']:
 v=read(R/name/'result.json');require(v['training_updates']==0 and v['A8_repeat_bit_exact'] and v['parameters_unchanged'],'not zero-update stable-state screen');base=np.load(v['variants']['A8_control']['logits']['path'])
 if first is None:first=base
 else:require(np.array_equal(first,base),'between-run A8 control differs')
 for variant,row in v['variants'].items():
  require(pin(row['logits']['path'])==row['logits'],'logits changed');x=np.load(row['logits']['path']);require(x.shape==(24,50000) and np.isfinite(x).all(),'invalid logits');pred=x.argmax(-1);matches=mapping[pred]==labels
  require(int(matches.sum())==row['gold_top1'] and int(matches[::2].sum())==row['gold_first_position'] and int(matches.reshape(-1,2).all(-1).sum())==row['gold_both_positions'],'gold counts differ');require(int((pred==base.argmax(-1)).sum())==row['agreement_with_A8'],'control agreement differs');require(sum(b!=8 for b in row['precision_by_projection'].values())==expected[variant],'wrong precision coverage')
  z=x.astype('f8');mx=z.max(-1);ce=float(np.mean(mx+np.log(np.exp(z-mx[:,None]).sum(-1))-z[np.arange(24),classes]));delta=abs(ce-row['supported_mean_ce']);require(delta<1e-5,'CE recomputation differs');max_ce_delta=max(delta,max_ce_delta)
 sets[name]=pin(R/name/'result.json')
for name in ['kernel-screen-v2','dflash-shape-screen']:
 v=read(R/name/'result.json');require(pin(v['library']['path'])==v['library'] and pin(v['source']['path'])==v['source'],'kernel source/library changed');require(len(v['checks'])==100 and len(v['records'])==96 and all(x['integer_and_scaled_exact'] for x in v['checks']),'incomplete kernel checks')
 for row in v['records']:
  for field in ['pack_dot_us','pack_us','dot_us']:require(len(row[field])==5 and np.isfinite(row[field]).all() and min(row[field])>0,'invalid timing')
  require(row['median_pack_dot_us']==float(np.median(row['pack_dot_us'])),'timing median differs')
 sets[name]=pin(R/name/'result.json')
atomic(R/'audit.json',dict(status='PRELIMINARY_SCREEN_RAW_AUDIT_PASS',source=pin(__file__),inputs=pin(R/'quality-inputs.json'),sets=sets,teacher_forced_positions=24,train_groups=12,variant_results=10,unique_policies=9,between_run_A8_exact=True,max_ce_delta=max_ce_delta,final_kernel_cases=200,scope='Recomputes logits-to-token/CE summaries and policy coverage; authenticates completed operator checks. No native model-serving or generalization certification.'))
print('PASS:10 result sets /9 policies;200 final kernel correctness cases; max CE delta',max_ce_delta)
