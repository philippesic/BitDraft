"""Matched common-parent R1 recovery, dense DEV and authenticated rolling recovery state."""
import argparse,copy,gc,json,os,random,signal,time,resource
from pathlib import Path
import numpy as np
import torch
from .common import *
from .simulation import install as install_component
from scripts.dflash_baseline.model import Model,Capture
from scripts.dflash_baseline.prototype_model import install,optimizer
from scripts.dflash_baseline.prototype import digest
from scripts.dflash_baseline.pipeline import load_parameters,parameters,rng,update,restore_rng
from .tensor_checkpoint import load
from .checkpoint import save_state as save_bsz
from scripts.dflash_baseline.components import install_components
from scripts.dflash_baseline.export_precision import export
from .export import rewrite

def evaluation(m,captures):
 records=[]
 with torch.no_grad():
  for cap in captures:
   for a in [cap.anchors[i] for i in sorted(set((0,len(cap.anchors)//2,len(cap.anchors)-1)))]:
    logits=m(cap.features,cap.tokens,a)[1:];labels=torch.tensor([cap.labels[a+j-1] for j in range(1,16)],device='cuda')
    wanted=logits.gather(1,labels[:,None]);rank=(logits>wanted).sum(-1)+1
    competitors=logits.clone();competitors.scatter_(1,labels[:,None],-torch.inf)
    records.append(dict(capture=str(cap.path),anchor=a,positions=15,ce=float(torch.nn.functional.cross_entropy(logits,labels)),agree=int((logits.argmax(-1)==labels).sum()),rank_sum=int(rank.sum()),margin_sum=float((wanted[:,0]-competitors.max(-1).values).sum())))
 return dict(records=records,positions=sum(x['positions'] for x in records),ce=float(np.mean([x['ce'] for x in records])),agreement=sum(x['agree'] for x in records)/sum(x['positions'] for x in records),mean_rank=sum(x['rank_sum'] for x in records)/sum(x['positions'] for x in records),mean_margin=sum(x['margin_sum'] for x in records)/sum(x['positions'] for x in records))

def main():
 began=time.perf_counter()
 ap=argparse.ArgumentParser();ap.add_argument('spec',type=Path);ap.add_argument('--updates',type=int,default=4096);ap.add_argument('--resume',type=Path);ap.add_argument('--out-name');ap.add_argument('--pilot',action='store_true');ap.add_argument('--checkpoint-points',type=int,nargs='*',default=[1024]);a=ap.parse_args()
 spec=json.load(open(a.spec));out=R/'arms'/(a.out_name or spec['name']);out.mkdir(parents=True,exist_ok=False)
 save(out/'spec.json',spec);save(out/'ownership.json',identity())
 torch.set_num_threads(2);torch.manual_seed(8101);random.seed(8101);np.random.seed(8101);torch.backends.cuda.matmul.allow_tf32=False
 # Recover exact parent into RAM-backed scratch, avoiding a 5.1GB disk transient.
 import tempfile
 tempfile.tempdir='/dev/shm'
 parent=load(B/'checkpoint.pt.bsz')
 receipt=json.load(open(B/'checkpoint.pt.json'))
 if parent['config_hash']!=digest(parent['config']) or parent['progress']['updates']!=2048:raise ValueError('wrong full B0 recovery state')
 c=copy.deepcopy(parent['config']);c['parent']={k:pin(B/'checkpoint.pt.bsz')[k] for k in ('path','sha256')};c['budget_seconds']=86400
 manifest=json.load(open(c['data']['path']));trainrows=[r for r in manifest['records'] if r['split']=='TRAIN'];devrows=[r for r in manifest['records'] if r['split']=='consumed_DEV']
 # Every data artifact and source is authenticated before any exposure.
 for p in [c['source'],c['data'],c['evaluation']['target']]+[p for r in trainrows+devrows for p in [r['metadata'],*r['files']]]:
  if pin(p['path'])['sha256']!=p['sha256']:raise ValueError('data/source pin changed '+p['path'])
 m=Model(c['source']['path'],c['evaluation']['target']['path']);install(m,c,{})
 load_parameters(m,parent['parameters'])
 progress=dict(updates=0,cursor=parent['progress']['cursor'],order=parent['progress']['order'],update_seconds=0.,supervised_tokens=0)
 lineage=dict(parent=pin(B/'checkpoint.pt.bsz'),parent_native=pin(B/'model-final.json'),parent_progress={k:v for k,v in parent['progress'].items() if k!='order'},recipe='R1 existing signs/scale/midpoint/gain/learned clip/norm; frozen quantized vocabulary; Adam reset identically',spec=spec)
 del parent;gc.collect()
 install_components(m,c);install_component(m,spec)
 if spec.get('r2_vocab_scale'):
  from .head_qat import install as install_head,optimizer as head_optimizer
  lineage['head_qat']=install_head(m,spec);lineage['recipe']='R2 existing R1 groups plus one native half vocabulary block-scale/256weights; identical capacity/calibration policy in matched privateQ6 control'
 m.to('cuda');opt=head_optimizer(m) if spec.get('r2_vocab_scale') else optimizer(m)
 if a.resume:
  state=load(a.resume)
  if state['component_spec']!=spec or state['config_hash']!=digest(c):raise ValueError('resume spec/config mismatch')
  load_parameters(m,state['parameters']);opt.load_state_dict(state['optimizer']);progress=state['progress'];restore_rng(state['rng']);del state;gc.collect()
 train=[Capture(Path(r['metadata']['path']).parent) for r in trainrows];dev=[Capture(Path(r['metadata']['path']).parent) for r in devrows]
 windows=[(i,x) for i,cap in enumerate(train) for x in cap.anchors]
 if sorted(progress['order'])!=list(range(len(windows))):raise ValueError('parent TRAIN order changed')
 unique_labels={(str(cap.path),a+j-1) for cap in train for a in cap.anchors for j in range(1,16)}
 save(out/'resolved.json',dict(config=c,spec=spec,lineage=lineage,data_windows=len(windows),unique_label_positions=len(unique_labels),unique_sequences=len(train),coverage_note='Overlapping windows/repeated epochs are exposure, not fresh labels.',parameter_map={n:dict(shape=list(p.shape),numel=p.numel(),master_dtype=str(p.dtype),trainable=p.requires_grad) for n,p in m.named_parameters()},source_pins={str(p.relative_to(ROOT)):pin(p) for directory in (ROOT/'scripts/component_campaign',ROOT/'scripts/dflash_baseline') for p in directory.glob('*.py')}))
 stopped=False
 def stop(*_):
  nonlocal stopped;stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 readouts=[];setup_seconds=time.perf_counter()-began
 def point(label):
  torch.cuda.synchronize();start=time.perf_counter();metrics=evaluation(m,dev)
  dev_seconds=time.perf_counter()-start;export_start=time.perf_counter()
  from .deployment import values
  stored_exps,rounding_differences=values(m,spec)
  state=dict(version=2,config=c,config_hash=digest(c),parameters=parameters(m),progress=copy.deepcopy(progress),lineage=lineage,deployment_exps=stored_exps)
  base=out/f'base-{label}.gguf'
  export(None,base,state=state,checkpoint_pin=lineage['parent'])
  rewrite(base,out/f'model-{label}.gguf',spec,head_log_d=state['parameters'].get('head_log_d'),deployment_exps=stored_exps);base.unlink();base.with_suffix('.json').unlink()
  model_path=out/f'model-{label}.gguf';model_pin=pin(model_path)
  qat_diagnostics=None
  if spec.get('r2_vocab_scale'):
   from .r2_diagnostics import inspect
   qat_diagnostics=inspect(m,opt,spec,model_path)
  export_seconds=time.perf_counter()-export_start;archive_start=time.perf_counter()
  # One frozen native basis per vocabulary format; later points keep only lossless deltas.
  from .archive import archive
  family=spec.get('head') or spec.get('embedding') or 'shared'
  bases=R/'bases';bases.mkdir(exist_ok=True);basis=bases/(family+'.gguf')
  if not basis.exists():
   # Hard link prevents duplicate physical storage for the first basis point.
   basis.hardlink_to(model_path)
  elif not model_path.samefile(basis):archive(model_path,basis)
  record=dict(head_qat_diagnostics=qat_diagnostics,update=progress['updates'],progress={k:v for k,v in progress.items() if k!='order'},dev=metrics,seconds=time.perf_counter()-start,dev_seconds=dev_seconds,native_export_seconds=export_seconds,archive_seconds=time.perf_counter()-archive_start,peak_training_bytes=torch.cuda.max_memory_allocated(),model=model_pin,cpu_gpu_half_exp_rounding_differences=rounding_differences)
  readouts.append(record);save(out/'readouts.json',readouts);print('READOUT',spec['name'],record['update'],metrics['ce'],metrics['agreement'],flush=True)
 def recoverable():
  io_start=time.perf_counter()
  directory=Path('/mnt/c/binary-eagle-component-20261105-01');directory.mkdir(exist_ok=True)
  destination=directory/(spec['name']+f"-{progress['updates']}.pt")
  policy=json.load(open(R/'checkpoint-policy.json')) if (R/'checkpoint-policy.json').exists() else {}
  delta=policy.get('format')=='tensor-delta-v1'
  existing=destination.with_suffix('.pt.cpt' if delta else '.pt.bsz')
  if existing.exists() and (out/'recovery.json').exists():
   old=json.load(open(out/'recovery.json'))
   if old['archive']==str(existing) and pin(existing)['sha256']==old['archive_sha256']:return old
   raise ValueError('unexpected existing recovery state')
  from .deployment import values
  stored_exps,_=values(m,spec)
  state=dict(version=2,config=c,config_hash=digest(c),parameters=parameters(m),optimizer=opt.state_dict(),progress=copy.deepcopy(progress),rng=rng(),lineage=lineage,component_spec=spec,resume_capable=True,deployment_exps=stored_exps)
  free=__import__('shutil').disk_usage(directory).free
  required=int(policy['observed_bytes']*1.05)+(512<<20) if delta else 4_800_000_000
  if free<required:raise RuntimeError(f'insufficient rolling checkpoint space: need{required}bytes free; observed{free}')
  if delta:
   from .tensor_checkpoint import save_state
  else:save_state=save_bsz
  receipt=save_state(destination,state);save(out/'recovery.json',receipt)
  # Only NEW campaign states participate in this declared rolling policy.
  # Atomically publish the replacement before retiring its predecessor.
  rolling=directory/'rolling.json';previous=json.load(open(rolling)) if rolling.exists() else None
  save(rolling,dict(arm=spec['name'],update=progress['updates'],receipt=receipt))
  if previous:
   old=Path(previous['receipt']['archive'])
   if old!=Path(receipt['archive']):
    if old.parent!=directory or old.is_symlink() or pin(old)['sha256']!=previous['receipt']['archive_sha256']:raise ValueError('rolling predecessor changed')
    with (R/'rolling-retention.jsonl').open('a') as f:f.write(json.dumps(dict(retired=previous,replaced_by=receipt,policy='NEW campaign rolling state; immutable B0 and all native readouts preserved; final shortlist recovery states regenerated under same history and retained'))+'\n')
    old.unlink()
  with (out/'recovery-cost.jsonl').open('a') as f:f.write(json.dumps(dict(update=progress['updates'],seconds=time.perf_counter()-io_start,receipt=receipt))+'\n')
  return receipt
 start_update=progress['updates']
 if not a.resume:point(0)
 times=[]
 while progress['updates']<a.updates:
  try:pause()
  except RuntimeError:
   stopped=True;break
  if stopped:break
  i,anchor=windows[progress['order'][progress['cursor']%len(windows)]]
  torch.cuda.synchronize();start=time.perf_counter();loss=update(m,opt,lambda:train[i].loss(m,anchor)[0]);torch.cuda.synchronize();elapsed=time.perf_counter()-start
  progress['update_seconds']+=elapsed;progress['updates']+=1;progress['cursor']+=1;progress['supervised_tokens']+=15;times.append(elapsed)
  with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(update=progress['updates'],loss=loss,gradient_norm=m.last_gradient_norm,seconds=elapsed,sequence=str(train[i].path),anchor=anchor))+'\n')
  if progress['updates'] in (256,1024,a.updates):
   point(progress['updates'])
   if progress['updates']!=a.updates and progress['updates'] in a.checkpoint_points:recoverable()
  if progress['updates']%32==0:
   save(out/'live.json',dict(progress={k:v for k,v in progress.items() if k!='order'},last_loss=loss,seconds_per_update=float(np.median(times[-32:])),identity=identity()));print(spec['name'],progress['updates'],loss,flush=True)
 recovery=recoverable()
 save(out/'result.json',dict(status='INTERRUPTED' if stopped else 'COMPLETE',spec=spec,progress={k:v for k,v in progress.items() if k!='order'},updates_this_process=progress['updates']-start_update,unique_label_positions=len(unique_labels),unique_windows=len(windows),supervised_positions=progress['supervised_tokens'],median_update_seconds=float(np.median(times)) if times else None,peak_training_bytes=torch.cuda.max_memory_allocated(),peak_training_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,recovery=recovery,readouts=readouts,setup_seconds=setup_seconds,process_wall_seconds=time.perf_counter()-began,checkpoint_points=a.checkpoint_points,native_points=[0,256,1024,a.updates]))
 print('TRAINING_COMPLETE',spec['name'],progress['updates'],flush=True)

if __name__=='__main__':
 main()
 from .frozen_source import finalize
 finalize()
