from pathlib import Path
import json,hashlib,os,time,sys,uuid,shutil
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');root=C/'w1a8-head-pair-20261103-01';out=root/'head5760-resume01';started=time.monotonic();sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
import torch
import modern_train_selected_head as trainer
from modern_actual_union import UnionActualDataset
from modern_actor import build_modern_actor
from modern_selected_head import SelectedHead
from w1a1_eagle.native_step import ExternalNativeF32Embedding
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def write(p,v):
 b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();assert shutil.disk_usage(C).free>=2*1024**3+64*1024**2+len(b);tmp=p.with_suffix(p.suffix+'.finalize-tmp')
 with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
assert read(R/'head-resume5760-job-result.json')['exit_code']==1 and not (out/'result.json').exists();assert pin(trainer.__file__)['sha256']=='54ed1dd8bb927148abbdc1bdd23d08ebf2bb3b7b23aeb14bfb1aa3ccbc39b961'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
checkpoint=out/'checkpoint-00005760.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['cursor']==5760 and saved['head_stage']['stage_updates']==1920;assert saved['exposure']['ce_windows']==11520 and saved['exposure']['replay_windows']==5760;trainer.validate_head_optimizer(saved['optimizer'],5760,True)
U=C/'w1a8-modern-fit-expansion-20261103-01/union01/manifest.json';data=UnionActualDataset(U);calls=[0]
def lookup(token):
 calls[0]+=1
 return data.embedding_lookup(token,device='cuda')
embedding=ExternalNativeF32Embedding(lookup=lookup,vocab_size=248320,hidden_size=2560,target_identity='frozen native Qwen3.5-4B Q4_K_M actual parent',target_sha256=data.manifest['target']['sha256'],capture_sha256=pin(U)['sha256'])
actor=build_modern_actor(P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',P/'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json',embedding_lookup=embedding,device='cuda',scale_only=False)
parent=C/'w1a8-expanded-ce-resident-new24-3840-01/checkpoint-00003840.pt';initial=torch.load(parent,map_location='cpu',weights_only=False);trainer.load_parent_scales(actor,initial['scales']);selection=read(out/'selection.json');selected=SelectedHead(actor,selection,base_state_pin=pin(parent),learn_signs=True);selected.load_state(saved['selected_state']);selected.check(integrity=True)
model=out/'model.gguf';existing=pin(model);scratch=Path('/dev/shm')/('w1a8-head-final-export-'+uuid.uuid4().hex+'.gguf')
try:
 exported=selected.export(scratch);reexport=exported['output'];assert reexport['sha256']==existing['sha256'] and reexport['bytes']==existing['bytes']==40802176
finally:scratch.unlink(missing_ok=True)
assert calls[0]==0,'export must not perform input/forward evaluation';metrics=trainer.sign_metrics(selected);views={};by_id={w['id']:w for w in data.windows}
for name,count in [('after-fit.json',1414),('after-development.json',724),('after-extra-fit.json',1465)]:
 v=read(out/name);assert len(v['rows'])==count and len({x['id'] for x in v['rows']})==count
 for row in v['rows']:
  w=by_id[row['id']];assert row['labels']==w['labels'] and row['role']==w['role'] and row['native_prefix']==w['native_prefix']
 views[name]=dict(file=pin(out/name),windows=count,prefix=sum(x['actor_prefix'] for x in v['rows']),lost_original=sum(max(0,x['native_prefix']-x['actor_prefix']) for x in v['rows']))
proof=dict(status='FULL5760_CHECKPOINT_AND_EXISTING_GPU_EXPORT_EXACT_REEXPORT_VERIFIED',source=pin(__file__),trainer=pin(trainer.__file__),checkpoint=pin(checkpoint),existing_model=existing,reexport_in_RAM=reexport,temporary_removed=True,embedding_calls=0,forward_calls=0,training_updates_performed=0,full_saved_training_updates=5760,new_head_training_updates=1920,selected_head_metrics=metrics,readouts=views,wall_seconds=time.monotonic()-started,scope='finalization only after source trainer completed1920newupdates/readouts/GGUF but metadata floorfailed; actualnative/serving pending')
write(R/'head-finalization5760-proof.json',proof);export_report={**exported,'output':existing,'publication_recovery':pin(R/'head-finalization5760-proof.json'),'RAM_reexport':reexport};write(out/'export.json',export_report)
first=read(R/'head-matched5760-progress.json');last=read(R/'head-resume5760-job-result.json');head_wall=first['stages'][1]['wall_seconds']+last['wall_seconds'];result=dict(status='TRAINED_EXPORTED_NOT_NATIVE_QUALIFIED',publication_owner='root exact-state finalizer',updates=5760,trainer_seconds=saved['trainer_seconds'],wall_seconds=head_wall,wall_seconds_scope='sum of two observed training subprocesses, excludes recovery/finalization/preparation',checkpoint=pin(checkpoint),model=existing,selected_head_metrics=metrics,head_stage=saved['head_stage'],stage=saved['stage'],costs=saved['costs'],exposure=saved['exposure'],ce_stage_exposure=saved['ce_stage_exposure'],resident_runtime_at_checkpoint=saved['resident_runtime'],finalization=pin(R/'head-finalization5760-proof.json'),scope=proof['scope']);write(out/'result.json',result)
control=first['stages'][0];head=dict(arm='head',exit_code=0,out=str(out),model=existing,checkpoint=pin(checkpoint),source=pin(trainer.__file__),original_attempt=first['stages'][1],resume=last,finalization=pin(R/'head-finalization5760-proof.json'));final=dict(status='MATCHED1920_CAPACITY_TRAINING_COMPLETE_NOT_NATIVE_ADMITTED',stages=[control,head],original_pair_wrapper_wall_seconds=first['wall_seconds'],resume_wrapper_wall_seconds=last['wall_seconds'],scope='same3840parent/1920newupdates each; control unchanged; allpaidheadsteps preserved through5280resume; actualGGUF reexportverified from5760state, native/serving pending');write(R/'head-matched5760-result.json',final);print(json.dumps(dict(status=proof['status'],metrics=metrics,readouts=views,wall_seconds=proof['wall_seconds'])))
