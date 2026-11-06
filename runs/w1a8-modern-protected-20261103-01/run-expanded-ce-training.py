from pathlib import Path
import json,subprocess,time,hashlib,signal,os
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');D=C/'w1a8-modern-fit-expansion-20261103-01';start=time.monotonic();stages=[]
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':s}
assert read(R/'fit-expansion-data-root-result.json')['status']=='ROOT_NEW24FIT_CAPTURE_AUDIT_AND_DATA_UNION_COMPLETE_NOT_TRAINED'
assert read(R/'expanded-data-parent-native01/result.json')['status']=='ALL407_NEW_NATIVE_PARENT_SUCCESSES_REPRODUCED'
script=W/'scripts/w1a8_generalization/modern_train_expanded_ce.py';assert pin(script)['sha256']=='76420c95fb0e5cd0550c90bd9b22319bb04dd4548cfd241c4b5d0adf74fd2315'
control=C/'w1a8-expanded-ce-control3840-01';treatment=C/'w1a8-expanded-ce-new24-3840-01';assert not control.exists() and not treatment.exists()
common=[str(P/'.venv/bin/python'),str(script),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--union-manifest',str(D/'union01/manifest.json'),'--reference',str(P/'runs/w1a8-modern-actual-training-20261103-01/reference02/reference.json'),'--warm-start',str(C/'w1a8-modern-protected-20261103-01/all-replay01/checkpoint-00001920.pt'),'--final-updates','3840','--extra-fit-readout','--allow-cuda']
plan={'status':'PAIRED_OLD_VS_NEW24_CE_TRAINING_START','source':pin(script),'data':pin(D/'union01/manifest.json'),'control':str(control),'treatment':str(treatment),'common':common,'objective':'CEfixed4/2windows, replayoriginal356FITfixed24groups/lambda10, 9scales/Adam1e-5seed37/horizon19200','horizon':19200,'comparison_checkpoint':3840,'quiet_timing':False};(R/'expanded-ce-training-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for expanded,out in [('false',control),('true',treatment)]:
 t=time.monotonic();code=subprocess.run(common+['--expanded-ce',expanded,'--out',str(out)],cwd=W).returncode;assert code==0,(expanded,code);result=read(out/'result.json');stage={'expanded_ce':expanded,'exit_code':code,'wall_seconds':time.monotonic()-t,'out':str(out),'stage_training':result['stage'],'ce_stage_exposure':result['ce_stage_exposure']};stages.append(stage)
 if expanded=='false':
  import torch
  import sys
  sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
  from modern_train_protected import equal
  base=C/'w1a8-modern-protected-20261103-01/nativefit3840-01';a=torch.load(base/'checkpoint-00003840.pt',map_location='cpu',weights_only=False);b=torch.load(out/'checkpoint-00003840.pt',map_location='cpu',weights_only=False)
  checks={key:equal(a[key],b[key]) for key in ['cursor','rng','cuda_rng','python_rng','exposure','scales','optimizer']}
  checks['nativeGGUF']=pin(base/'model.gguf')['sha256']==result['model']['sha256']
  for split in ['fit','development']:
   old=read(base/f'after-{split}.json')['rows'];new=read(out/f'after-{split}.json')['rows'];fields=['id','role','native_prefix','actor_prefix','proposals','labels','supported'];checks[split+'_decisions']=[{k:x[k] for k in fields} for x in old]==[{k:x[k] for k in fields} for x in new]
  proof={'status':'EXACT_DATA_DISABLED_CONTROL' if all(checks.values()) else 'CONTROL_MISMATCH_STOP','checks':checks,'all_exact':all(checks.values()),'new_checkpoint':result['checkpoint'],'old_checkpoint':pin(base/'checkpoint-00003840.pt')};(R/'expanded-ce-training-control-parity.json').write_text(json.dumps(proof,indent=2)+'\n');assert proof['all_exact'],checks;stage['all_control_exact']=True
 (R/'expanded-ce-training-progress.json').write_text(json.dumps({'stages':stages,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
(R/'expanded-ce-training-result.json').write_text(json.dumps({'status':'DATA_ONLY_PAIRED_TRAINING_COMPLETE_NOT_NATIVE_QUALIFIED','stages':stages,'wall_seconds':time.monotonic()-start,'scope':'Phase1dataeffect; no freshgeneralization/predictor/backloglabels'},indent=2)+'\n')
