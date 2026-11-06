from pathlib import Path
import json,hashlib,math,collections,os
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';D=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/repeated-serving01');run=P/'runs/local-w1a8-capacity-replay-requests-20261103-01'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(ref):
 assert pin(ref['path'])==ref
 return read(ref['path'])
def sig(v):return {k:v[k] for k in ['tokens','content','stop_type','stopping_word','truncated']}
state=read(run/'state.json');assert state['status']=='finished' and state['exit_code']==0
marker=read(R/'capacity-replay-quiet-window.json');marker['status']='TIMING_TERMINAL_ROOT_AUDIT_IN_PROGRESS';(R/'capacity-replay-quiet-window.json').write_text(json.dumps(marker,indent=2)+'\n')
v=read(D/'result.json');assert v['status']=='DEVELOPMENT_NATIVE_REQUESTS_PASS' and v['measured_requests']==480 and v['excluded_warmups']==80
models=read(D/'models.json');inputs=read(D/'inputs.json');arms={a['id']:a for a in models['arms']};expected={x['id']:x for x in inputs['rows']+inputs['warmups']};gold={k:sig(checked(x['reference'])) for k,x in expected.items()};original={k:checked(x['request']) for k,x in expected.items()};seen=set();counts=collections.Counter();measured=[];spec=0
for row in v['records']:
 key=(row['arm'],row['repetition'],row['id']);assert key not in seen;seen.add(key);request=checked(row['request']);response=checked(row['response']);assert request==original[row['id']] and request['cache_prompt'] is False and request['seed']==1234 and request['temperature']==0 and request['return_tokens'] is True
 assert sig(response)==gold[row['id']] and len(response['tokens'])==row['completion_tokens']==response['tokens_predicted'];assert response['timings']['cache_n']==0;assert math.isfinite(row['full_http_wall_seconds']) and row['full_http_wall_seconds']>0
 counts[row['arm']]+=1;spec+=int(arms[row['arm']]['kind']!='target')
 if not row['warmup']:measured.append(row)
assert len(seen)==560 and len(measured)==480 and spec==490 and all(n==70 for n in counts.values())
images={};groups=set();cells=list(D.glob('rep-*'));assert len(cells)==40
for cell in cells:
 rel=read(cell/'release.json');assert rel['owned_process_reaped'] and rel['owned_process_group_absent'] and rel['returncode']==0;groups.add(rel['identity']['pgid']);command=read(cell/'command.json');argv=command['argv'];env=command['environment'];records=[x for x in v['records'] if Path(x['response']['path']).parent==cell];assert len(records)==14;arm=arms[records[0]['arm']]
 assert argv[0]==arm['runtime']['server']['path'] and argv[argv.index('-m')+1]==models['target']['path']
 for option,value in [('-c','4096'),('-b','512'),('-ub','512'),('-t','4'),('-fa','on'),('--cache-type-k','f16'),('--cache-type-v','f16'),('-lv','3')]:assert argv[argv.index(option)+1]==value,(cell,option)
 assert env['GGML_CUDA_DECODE_CONSISTENT']=='32' and env['CUDA_VISIBLE_DEVICES']=='0'
 if arm['kind']!='target':assert argv[argv.index('-md')+1]==arm['model']['path']
 if arm['kind']=='w1a8':assert env['GGML_W1AX_ACT_BITS']=='8'
 else:assert 'GGML_W1AX_ACT_BITS' not in env
 for phase in ['warm','final']:
  for ref in read(cell/f'loaded-images-{phase}.json')['images']:
   assert ref['path'] not in images or images[ref['path']]==ref;images[ref['path']]=ref
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:assert int((p/'stat').read_text().rsplit(')',1)[1].split()[2]) not in groups
 except (OSError,ValueError,IndexError):pass
for ref in images.values():assert pin(ref['path'])==ref
for arm in arms.values():
 if arm['model']:assert pin(arm['model']['path'])==arm['model']
assert pin(models['target']['path'])==models['target'];freeze=read(D/'freeze.json')
for entry in freeze['source_snapshots']:
 for field in ['snapshot']:
  if field in entry:assert pin(entry[field]['path'])==entry[field]
prompts=sorted(x['id'] for x in inputs['rows']);tables={};summary=[]
for name in arms:
 matrix=np.empty((12,5));tokens=np.empty((12,5));rows=[x for x in measured if x['arm']==name]
 for x in rows:matrix[prompts.index(x['id']),x['repetition']]=x['full_http_wall_seconds'];tokens[prompts.index(x['id']),x['repetition']]=x['completion_tokens']
 assert len(rows)==60 and np.isfinite(matrix).all();tables[name]=(matrix,tokens);tps=float(tokens.sum()/matrix.sum());framework=next(x for x in v['arms'] if x['arm']==name);assert abs(tps-framework['tokens_per_second'])<1e-10
 summary.append(dict(arm=name,kind=arms[name]['kind'],tokens=int(tokens.sum()),wall_seconds=float(matrix.sum()),TPS=tps,per_repetition_TPS=(tokens.sum(0)/matrix.sum(0)).tolist(),draft_accepted=sum(x['draft_accepted'] or 0 for x in rows),draft_proposed=sum(x['draft_proposed'] or 0 for x in rows)))
rng=np.random.default_rng(37001103);pw=rng.multinomial(12,[1/12]*12,size=10000);rw=rng.multinomial(5,[.2]*5,size=10000);contrasts=[]
for a,b in [('w1a8_added_replay5760','w1a8_control5760'),('w1a8_head5760','w1a8_control5760'),('w1a8_control5760','w1a8_quarter'),('w1a8_quarter','w1a8_initial')]+[(name,'dflash_Q4KM') for name in arms if name!='dflash_Q4KM']:
 ta,na=tables[a];tb,nb=tables[b];assert np.array_equal(na,nb);samples=np.einsum('bi,ij,bj->b',pw,tb,rw)/np.einsum('bi,ij,bj->b',pw,ta,rw);ratio=float((na.sum()/ta.sum())/(nb.sum()/tb.sum()));ci=np.quantile(samples,[.025,.975]).tolist();contrasts.append(dict(candidate=a,control=b,pooled_ratio=ratio,paired_prompt_and_repetition_bootstrap95=ci,per_repetition_ratios=(tb.sum(0)/ta.sum(0)).tolist()))
report=dict(status='ROOT_FULL_8ARM_5REP_RAW_REQUEST_AUDIT_PASS',result=pin(D/'result.json'),models=pin(D/'models.json'),source=pin(__file__),requests=560,measured=480,warmups=80,speculative_equal_target=490,released_groups=len(groups),loaded_images_rehashed=len(images),image_pins=list(images.values()),summary=summary,contrasts=contrasts,bootstrap=dict(seed=37001103,samples=10000,unit='joint wholeDEVprompt and repetition blocks, paired acrossarms'),scope='12reusedDEVprompts/80tokenlimit, notuntouched/independentquality/predictor/backlog; originalDFlash235.9TPS notdenominator',runner_wall_seconds=v['wall_seconds'])
(R/'capacity-replay-repeated-root-analysis.json').write_text(json.dumps(report,indent=2)+'\n');marker['status']='COMPLETE_ROOT_AUDITED_RELEASED';marker['root_audit']=pin(R/'capacity-replay-repeated-root-analysis.json');(R/'capacity-replay-quiet-window.json').write_text(json.dumps(marker,indent=2)+'\n');print(json.dumps(dict(TPS={x['arm']:x['TPS'] for x in summary},primary_contrasts=contrasts[:4],groups=len(groups),images=len(images),wall_seconds=v['wall_seconds'])))
