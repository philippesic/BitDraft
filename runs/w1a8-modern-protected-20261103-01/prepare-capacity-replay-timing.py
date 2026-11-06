from pathlib import Path
import json,hashlib,copy
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
m=read(R/'added-replay-HTTP-models-verbose01.json');headmodels=read(R/'head-HTTP-models-verbose01.json');head=next(a for a in headmodels['arms'] if a['id']=='w1a8_head5760');m['arms'].insert(-2,copy.deepcopy(head));assert len(m['arms'])==8
proofdir=R/'capacity-replay-loader-qualifications01';proofdir.mkdir(exist_ok=False)
for arm in m['arms']:
 runtime=arm['runtime']
 if arm['id'] in ['w1a8_control5760','w1a8_head5760','w1a8_added_replay5760']:
  pair='w1a8-head-pair-20261103-01' if arm['id']=='w1a8_head5760' else 'w1a8-added-replay-pair-20261103-01';cell=C/pair/'http-qualification01'/('rep-00-'+arm['id']);verbose=read(cell.parent/'models.json');actual=next(a for a in verbose['arms'] if a['id']==arm['id']);assert actual==arm
  result=read(cell.parent/'result.json');responses=[x for x in result['records'] if x['arm']==arm['id']];assert len(responses)==14 and all(x['token_text_stop_parity'] for x in responses);log=(cell/'server.log').read_text();assert 'complete nine-linear fixed row W1A8' in log and log.count('W1A8 packed ')==9
  release=read(cell/'release.json');assert release['owned_process_reaped'] and release['owned_process_group_absent'] and release['returncode']==0
  normalized={k:v for k,v in runtime.items() if k not in ['log_verbosity','loader_qualification']};q=dict(schema='w1a8_model_runtime_native_loader_qualification_v1',model=arm['model'],runtime=normalized,native_outputs_audited=14,raw_log=pin(cell/'server.log'),release=pin(cell/'release.json'),command=pin(cell/'command.json'),maps=[pin(cell/'loaded-images-warm.json'),pin(cell/'loaded-images-final.json')],models=pin(cell.parent/'models.json'),responses=[x['response'] for x in responses]);file=proofdir/(arm['id']+'.json');file.write_text(json.dumps(q,indent=2)+'\n');runtime['loader_qualification']=pin(file)
 runtime['log_verbosity']=3
 for field in ['server','libraries']:
  for ref in runtime[field] if field=='libraries' else [runtime[field]]:assert pin(ref['path'])==ref
 if arm['model'] is not None:assert pin(arm['model']['path'])==arm['model']
 if arm['kind']=='w1a8':
  q=read(runtime['loader_qualification']['path']);assert q['model']['sha256']==arm['model']['sha256'] and q['runtime']=={k:v for k,v in runtime.items() if k not in ['log_verbosity','loader_qualification']}
m['source_pins'] += [pin(R/'head-matched5760-result.json'),pin(R/'head-finalization5760-proof.json'),pin(R/'head-http-root-audit.json'),pin(R/'added-replay-http-root-audit.json'),pin(__file__)];m['data_usage']=dict(scope='8arms12reusedDEVprompts5reps480measured+80FITwarmups; samebaselineprecision/runtime; nofreshclaim',quiet_window='No otherGPU, builds, tests, largehashing or compression duringmeasuredcells')
out=R/'capacity-replay-models-quiet01.json';out.write_text(json.dumps(m,indent=2)+'\n');print('Models ready',len(m['arms']),pin(out))
