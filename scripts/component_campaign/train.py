"""Matched common-parent R1 recovery, dense DEV and authenticated rolling recovery state."""
import argparse,copy,gc,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .simulation import install as install_component
from scripts.dflash_baseline.model import Model,Capture
from scripts.dflash_baseline.prototype_model import install,optimizer
from scripts.dflash_baseline.prototype import digest
from scripts.dflash_baseline.pipeline import load_parameters,parameters,rng,update,restore_rng
from scripts.dflash_baseline.checkpoint_io import load,save_streamed
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
 ap=argparse.ArgumentParser();ap.add_argument('spec',type=Path);ap.add_argument('--updates',type=int,default=4096);ap.add_argument('--resume',type=Path);ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
 spec=json.load(open(a.spec));out=R/'arms'/spec['name'];out.mkdir(parents=True,exist_ok=False)
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
 install_components(m,c);install_component(m,spec);m.to('cuda');opt=optimizer(m)
 if a.resume:
  state=load(a.resume)
  if state['component_spec']!=spec:raise ValueError('resume spec mismatch')
  load_parameters(m,state['parameters']);opt.load_state_dict(state['optimizer']);progress=state['progress'];restore_rng(state['rng']);del state;gc.collect()
 train=[Capture(Path(r['metadata']['path']).parent) for r in trainrows];dev=[Capture(Path(r['metadata']['path']).parent) for r in devrows]
 windows=[(i,x) for i,cap in enumerate(train) for x in cap.anchors]
 if sorted(progress['order'])!=list(range(len(windows))):raise ValueError('parent TRAIN order changed')
 unique_labels={(str(cap.path),a+j-1) for cap in train for a in cap.anchors for j in range(1,16)}
 save(out/'resolved.json',dict(config=c,spec=spec,lineage=lineage,data_windows=len(windows),unique_label_positions=len(unique_labels),unique_sequences=len(train),coverage_note='Overlapping windows/repeated epochs are exposure, not fresh labels.',source_pins={str(p.relative_to(ROOT)):pin(p) for directory in (ROOT/'scripts/component_campaign',ROOT/'scripts/dflash_baseline') for p in directory.glob('*.py')}))
 stopped=False
 def stop(*_):
  nonlocal stopped;stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 readouts=[]
 def point(label):
  torch.cuda.synchronize();start=time.perf_counter();metrics=evaluation(m,dev)
  state=dict(version=2,config=c,config_hash=digest(c),parameters=parameters(m),progress=copy.deepcopy(progress),lineage=lineage)
  base=out/f'base-{label}.gguf'
  export(None,base,state=state,checkpoint_pin=lineage['parent'])
  rewrite(base,out/f'model-{label}.gguf',spec);base.unlink();base.with_suffix('.json').unlink()
  record=dict(update=progress['updates'],progress={k:v for k,v in progress.items() if k!='order'},dev=metrics,seconds=time.perf_counter()-start,peak_training_bytes=torch.cuda.max_memory_allocated(),model=pin(out/f'model-{label}.gguf'))
  readouts.append(record);save(out/'readouts.json',readouts);print('READOUT',spec['name'],record['update'],metrics['ce'],metrics['agreement'],flush=True)
 def recoverable():
  directory=Path('/mnt/c/binary-eagle-component-20261105-01');directory.mkdir(exist_ok=True)
  destination=directory/(spec['name']+f"-{progress['updates']}.pt")
  state=dict(version=2,config=c,config_hash=digest(c),parameters=parameters(m),optimizer=opt.state_dict(),progress=copy.deepcopy(progress),rng=rng(),lineage=lineage,component_spec=spec,resume_capable=True)
  free=__import__('shutil').disk_usage(directory).free
  if free<5_500_000_000:raise RuntimeError(f'insufficient rolling checkpoint space: need5.5GB free; observed{free}')
  receipt=save_streamed(destination,state);save(out/'recovery.json',receipt)
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
  if progress['updates'] in (256,1024,a.updates):point(progress['updates'])
  if progress['updates']%32==0:
   save(out/'live.json',dict(progress={k:v for k,v in progress.items() if k!='order'},last_loss=loss,seconds_per_update=float(np.median(times[-32:])),identity=identity()));print(spec['name'],progress['updates'],loss,flush=True)
 recovery=recoverable()
 save(out/'result.json',dict(status='INTERRUPTED' if stopped else 'COMPLETE',spec=spec,progress={k:v for k,v in progress.items() if k!='order'},updates_this_process=progress['updates']-start_update,unique_label_positions=len(unique_labels),unique_windows=len(windows),supervised_positions=progress['supervised_tokens'],median_update_seconds=float(np.median(times)) if times else None,peak_training_bytes=torch.cuda.max_memory_allocated(),recovery=recovery,readouts=readouts))
 print('TRAINING_COMPLETE',spec['name'],progress['updates'],flush=True)

if __name__=='__main__':main()
