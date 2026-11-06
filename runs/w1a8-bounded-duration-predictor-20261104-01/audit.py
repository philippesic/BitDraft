from pathlib import Path
import json,hashlib,math,collections,os,datetime
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');N=P/'runs/w1a8-bounded-duration-predictor-20261104-01';D=N/'repeated-serving01'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve(strict=True)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(ref):
 assert pin(ref['path'])==ref,ref['path']
 return read(ref['path'])
def sig(v):return {k:v[k] for k in ['tokens','content','stop_type','stopping_word','truncated']}
v=read(D/'result.json');assert v['status']=='DEVELOPMENT_NATIVE_REQUESTS_PASS' and v['measured_requests']==420 and v['excluded_warmups']==70
models=read(D/'models.json');inputs=read(D/'inputs.json');arms={a['id']:a for a in models['arms']};expected={x['id']:x for x in inputs['rows']+inputs['warmups']};gold={k:sig(checked(x['reference'])) for k,x in expected.items()};original={k:checked(x['request']) for k,x in expected.items()};seen=set();counts=collections.Counter();measured=[];spec=0
for row in v['records']:
 key=(row['arm'],row['repetition'],row['id']);assert key not in seen;seen.add(key);request=checked(row['request']);response=checked(row['response']);assert request==original[row['id']] and request['cache_prompt'] is False and request['seed']==1234 and request['temperature']==0 and request['return_tokens'] is True
 assert sig(response)==gold[row['id']] and len(response['tokens'])==row['completion_tokens']==response['tokens_predicted'];assert response['timings']['cache_n']==0;assert math.isfinite(row['full_http_wall_seconds']) and row['full_http_wall_seconds']>0
 counts[row['arm']]+=1;spec+=int(arms[row['arm']]['kind']!='target')
 if not row['warmup']:measured.append(row)
assert len(seen)==490 and len(measured)==420 and spec==420 and all(n==70 for n in counts.values())
images={};groups=set();cells=list(D.glob('rep-*'));assert len(cells)==35
for cell in cells:
 rel=read(cell/'release.json');assert rel['owned_process_reaped'] and rel['owned_process_group_absent'] and rel['returncode']==0;groups.add(rel['identity']['pgid']);command=read(cell/'command.json');argv=command['argv'];env=command['environment'];records=[x for x in v['records'] if Path(x['response']['path']).parent==cell];assert len(records)==14;arm=arms[records[0]['arm']]
 assert argv[0]==arm['runtime']['server']['path'] and argv[argv.index('-m')+1]==models['target']['path']
 for option,value in [('-c','4096'),('-b','512'),('-ub','512'),('-t','4'),('-fa','on'),('--cache-type-k','f16'),('--cache-type-v','f16'),('-lv','3')]:assert argv[argv.index(option)+1]==value,(cell,option)
 assert env['GGML_CUDA_DECODE_CONSISTENT']=='32' and env['CUDA_VISIBLE_DEVICES']=='0'
 if arm['kind']!='target':
  assert argv[argv.index('-md')+1]==arm['model']['path']
  assert argv[argv.index('--spec-draft-n-max')+1]=='2'
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
 if 'snapshot' in entry:assert pin(entry['snapshot']['path'])==entry['snapshot']
prompts=sorted(x['id'] for x in inputs['rows']);tables={};summary=[]
for name in arms:
 matrix=np.full((12,5),np.nan);tokens=np.full((12,5),np.nan);rows=[x for x in measured if x['arm']==name]
 for x in rows:matrix[prompts.index(x['id']),x['repetition']]=x['full_http_wall_seconds'];tokens[prompts.index(x['id']),x['repetition']]=x['completion_tokens']
 assert len(rows)==60 and np.isfinite(matrix).all();tables[name]=(matrix,tokens);tps=float(tokens.sum()/matrix.sum());framework=next(x for x in v['arms'] if x['arm']==name);assert abs(tps-framework['tokens_per_second'])<1e-10
 summary.append(dict(arm=name,kind=arms[name]['kind'],tokens=int(tokens.sum()),wall_seconds=float(matrix.sum()),TPS=tps,per_repetition_TPS=(tokens.sum(0)/matrix.sum(0)).tolist(),draft_accepted=sum(x['draft_accepted'] or 0 for x in rows),draft_proposed=sum(x['draft_proposed'] or 0 for x in rows)))
rng=np.random.default_rng(41001104);pw=rng.multinomial(12,[1/12]*12,size=10000);rw=rng.multinomial(5,[.2]*5,size=10000);contrasts=[]
for a,b in read(N/'frozen-study.json')['contrasts']+[[name,'w1a8_initial'] for name in ['old3840','expanded3840','old5760','expanded5760']]:
 ta,na=tables[a];tb,nb=tables[b];assert np.array_equal(na,nb);samples=np.einsum('bi,ij,bj->b',pw,tb,rw)/np.einsum('bi,ij,bj->b',pw,ta,rw);ratio=float((na.sum()/ta.sum())/(nb.sum()/tb.sum()));ci=np.quantile(samples,[.025,.975]).tolist();contrasts.append(dict(candidate=a,control=b,pooled_ratio=ratio,paired_prompt_and_repetition_bootstrap95=ci,per_repetition_ratios=(tb.sum(0)/ta.sum(0)).tolist(),ranking='candidate_faster' if ci[0]>1 else 'control_faster' if ci[1]<1 else 'unresolved'))
early=next(x for x in contrasts if x['candidate']=='expanded3840' and x['control']=='old3840');later=contrasts[0];predictor='unresolved' if 'unresolved' in [early['ranking'],later['ranking']] else 'rank_preserved' if early['ranking']==later['ranking'] else 'rank_reversed'
report=dict(status='RAW_7ARM_5REP_NATIVE_AUDIT_PASS',result=pin(D/'result.json'),models=pin(D/'models.json'),frozen_study=pin(N/'frozen-study.json'),source=pin(__file__),requests=490,measured=420,warmups=70,speculative_equal_target=420,released_groups=len(groups),loaded_images_rehashed=len(images),image_pins=list(images.values()),summary=summary,contrasts=contrasts,predictor_outcome=predictor,predictor_validated=False,bootstrap=dict(seed=41001104,samples=10000,unit='joint wholeDEVprompt and repetition blocks paired acrossarms'),scope='12reusedDEVprompts/80tokenlimit, two training recipes at3840/5760. Not untouched or broad predictor validation. Equal updates, not equal GPU seconds. Existingfresh24 transfer unchanged.',runner_wall_seconds=v['wall_seconds'],audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(N/'root-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(TPS={x['arm']:x['TPS'] for x in summary},contrasts=contrasts[:4],predictor=predictor,groups=len(groups),images=len(images),wall_seconds=v['wall_seconds'])))
