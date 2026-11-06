from pathlib import Path
import json,hashlib,copy,datetime
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';N=P/'runs/w1a8-bounded-duration-predictor-20261104-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve(strict=True)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def atomic(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
m=read(R/'capacity-replay-models-quiet01.json');template=next(a for a in m['arms'] if a['id']=='w1a8_control5760');m['arms']=[a for a in m['arms'] if a['id'] in ['target_only','dflash_Q4KM','w1a8_initial']];assert len(m['arms'])==3
sources={
 'old3840':C/'w1a8-expanded-ce-control3840-01/result.json',
 'expanded3840':C/'w1a8-expanded-ce-resident-new24-3840-01/result.json',
 'expanded5760':C/'w1a8-head-pair-20261103-01/control5760/result.json'}
# Derive quiet qualifications solely from completed verbose model/runtime cells.
verbose=read(R/'data-CE-models-verbose02.json')
for name,oldid in [('old3840','w1a8_oldCE3840'),('expanded3840','w1a8_newCE3840')]:
 a=copy.deepcopy(next(x for x in verbose['arms'] if x['id']==oldid));cell=R/'data-CE-http-qualification02'/('rep-00-'+oldid);r=read(cell/'release.json');assert r['owned_process_reaped'] and r['owned_process_group_absent'] and r['returncode']==0
 rows=[x for x in read(cell.parent/'result.json')['records'] if x['arm']==oldid];assert len(rows)==14 and all(x['token_text_stop_parity'] for x in rows)
 log=(cell/'server.log').read_text();assert 'complete nine-linear fixed row W1A8' in log and log.count('W1A8 packed ')==9
 runtime={k:v for k,v in a['runtime'].items() if k not in ['log_verbosity','loader_qualification']}
 q=dict(schema='w1a8_model_runtime_native_loader_qualification_v1',model=a['model'],runtime=runtime,native_outputs_audited=14,raw_log=pin(cell/'server.log'),release=pin(cell/'release.json'),command=pin(cell/'command.json'),maps=[pin(cell/'loaded-images-warm.json'),pin(cell/'loaded-images-final.json')],models=pin(cell.parent/'models.json'),responses=[x['response'] for x in rows]);qpath=N/(name+'-qualification.json');atomic(qpath,q)
 a['id']=name;a['runtime']['log_verbosity']=3;a['runtime']['loader_qualification']=pin(qpath);m['arms'].append(a)
a=copy.deepcopy(template);a['id']='expanded5760';m['arms'].append(a)
# New old5760 model will be bound only after exact intermediate reproduction gate.
for name,path in sources.items():
 result=read(path);assert next(x for x in m['arms'] if x['id']==name)['model']==result['model'];m['source_pins'].append(pin(path))
inputs=P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json'
freeze=dict(schema='bounded_duration_native_predictor_freeze_v1',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_utc='2026-11-04T01:47:00+00:00',gpu_stop_by_utc='2026-11-04T01:43:00+00:00',tasks=['Equal 5760 cumulative optimizer updates old-only CE versus expanded CE, NOT equal measured GPU seconds','Two-recipe retrospective native throughput rank at 3840 versus 5760 cumulative updates'],arms=[a['id'] for a in m['arms']]+['old5760'],new_endpoint_plan=pin(N/'plan.json'),existing_endpoints={k:pin(v) for k,v in sources.items()},panel=pin(inputs),panel_scope='All12 existing roster DEV prompts and2 FIT warmups; 80token max; DEV consumed, no fresh confirmation; not fresh24 rates.',runtime='Existing mode32/RTX5080/Qwen3.5 4B Q4_K_M, original W1A8 nine-linear fixed-row W1A8, depth2; native default caches/masks/verifier unchanged.',repetitions=5,order='forward/reverse arms on alternating repetitions',metric='total returned tokens / sum full HTTP wall seconds, excluding startup/warmups',contrasts=[['expanded5760','old5760'],['expanded3840','old3840'],['old5760','old3840'],['expanded5760','expanded3840']],predictor_decision='Early sign expanded3840/old3840 compared with later expanded5760/old5760. Call ranking unresolved if paired prompt/repetition bootstrap95 spans1 at either stage. Only two recipes, retrospective, one target/length/cohort; no validated general predictor claim.',interval='10000 paired whole-prompt and repetition block bootstrap, seed41001104',no_extra_training_or_fresh_inputs=True)
atomic(N/'frozen-study.json',freeze);atomic(N/'models-pre-endpoint.json',m);print(json.dumps(freeze,indent=2))
