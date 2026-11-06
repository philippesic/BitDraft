from pathlib import Path
import os,json,subprocess,signal,time,hashlib,sys
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';O=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-protected-20261103-01');C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');trainer=W/'scripts/w1a8_generalization/modern_train_selected_sign.py';start=time.monotonic();stopping=False;records=[]
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
plan=json.loads((R/'selected-training-plan.json').read_text());assert pin(trainer)==plan['trainer'];env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
for mode,name in [('false','w1a8-selected-sign-control3840-01'),('true','w1a8-selected-sign-learn3840-01')]:
 assert not stopping;out=C/name;assert not out.exists();t=time.monotonic();argv=[str(P/'.venv/bin/python'),str(trainer),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--reference',str(P/'runs/w1a8-modern-actual-training-20261103-01/reference02/reference.json'),'--warm-start',str(O/'all-replay01/checkpoint-00001920.pt'),'--out',str(out),'--learn-signs',mode,'--sign-lr','0.001','--scale-lr','0.00001','--scope','all','--retention','10','--steps','3840','--allow-cuda'];print(json.dumps({'stage':mode,'status':'starting','out':str(out)}),flush=True)
 with (R/(name+'.log')).open('x') as log:
  child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc',str(child.pid));write(R/(name+'-identity.json'),{'pid':child.pid,'pgid':os.getpgid(child.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()});code=child.wait()
 assert code==0,(mode,code);assert pin(trainer)==plan['trainer'];record={'learn_signs':mode=='true','exit_code':code,'wall_seconds':time.monotonic()-t,'out':str(out)}
 if mode=='false':
  import torch
  torch.set_num_threads(2);sys.path.insert(0,str(W/'scripts/w1a8_generalization'));from modern_train_selected_sign import checkpoint_equal
  actual=torch.load(out/'checkpoint-00003840.pt',map_location='cpu',weights_only=False);expected=torch.load(O/'nativefit3840-01/checkpoint-00003840.pt',map_location='cpu',weights_only=False)
  checks={k:checkpoint_equal(actual[k],expected[k]) for k in ['cursor','rng','cuda_rng','python_rng','exposure']};checks['nine_raw_offsets']=checkpoint_equal(actual['selected_state']['scale_offsets'],expected['scales']['scale_offsets']);checks['actor_ancestry']=actual['selected_state']['ancestry']==expected['scales']['ancestry']
  aopt=actual['optimizer'];eopt=expected['optimizer'];apaths=actual['contract']['optimized_scale_paths'];epaths=expected['contract']['optimized_scale_paths'];amap=dict(zip(apaths,aopt['param_groups'][1]['params']));emap=dict(zip(epaths,eopt['param_groups'][0]['params']));checks['scale_Adam_by_path']=set(amap)==set(emap) and all(checkpoint_equal(aopt['state'][amap[p]],eopt['state'][emap[p]]) for p in amap);checks['O_moments_absent']=aopt['param_groups'][0]['params'][0] not in aopt['state']
  latent=actual['selected_state']['latent_signs']['blk.0.attn_output'];initial_hash=actual['selected_state']['selection']['frozen_initial_sign_hashes']['midlayer.self_attn.o_proj'];checks['O_latents_original']=hashlib.sha256(latent.numpy().tobytes()).hexdigest()==initial_hash
  for split in ['fit','development']:
   ar=json.loads((out/('after-'+split+'.json')).read_text())['rows'];er=json.loads((O/'nativefit3840-01'/('after-'+split+'.json')).read_text())['rows'];fields=['id','role','native_prefix','actor_prefix','proposals','labels','supported','protected'];checks[split+'_readout_decisions']=len(ar)==len(er) and all({k:x[k] for k in fields}=={k:y[k] for k in fields} for x,y in zip(ar,er))
  checks['native_GGUF']=pin(out/'model.gguf')['sha256']==pin(O/'nativefit3840-01/model.gguf')['sha256'];write(R/'selected-training-control-parity.json',{'checks':checks,'all_exact':all(checks.values()),'control':pin(out/'checkpoint-00003840.pt'),'reference':pin(O/'nativefit3840-01/checkpoint-00003840.pt')});assert all(checks.values()),checks;record['control_all_exact']=True;del actual,expected,latent
 records.append(record);write(R/'selected-training-progress.json',{'stages':records,'wall_seconds':time.monotonic()-start});print(json.dumps(record),flush=True)
write(R/'selected-training-result.json',{'status':'MATCHED_SELECTED_SIGN_TRAINING_COMPLETE_NOT_NATIVE_QUALIFIED','stages':records,'wall_seconds':time.monotonic()-start,'old_fresh_inputs_used':False})
