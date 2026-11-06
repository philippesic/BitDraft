"""Restore authentic parent and paid EAL state; no optimization or forward changes."""
from pathlib import Path
from dataclasses import replace,asdict
import sys,json,time,gc
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
sys.path[:0]=[str(ROOT/'scripts'),str(OLD/'source')]
from shortlist_pilots.objectives.runtime import configure_imports,load_full_model,pin,native_export
from shortlist_pilots.training_transfer.metrics import score

def main():
 configure_imports()
 import torch,numpy as np
 from actor_gate import block
 from train_nine_model_qat import tensor_batch
 from joint_tools import joint_parameters,binary_buffers
 from shortlist_pilots.binary.parent_loader import copy_named_state,equal,sign_hashes
 from w1a1_eagle.block_training import _block_rng_state,_restore_block_rng
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 out=RUN/'historical-current-actor';out.mkdir(exist_ok=False);setup=out/'setup';setup.mkdir();data=json.loads((RUN/'historical-matrix.json').read_text());spec=json.loads((OLD/'train-control/start.json').read_text())['spec'];parent=spec['parent'];eal=json.loads((OLD/'train-control/finish.json').read_text())
 model,opt,baseline,meta=load_full_model(setup,parent,source_checkpoint=spec['source_checkpoint']['path']);model.eval();original=model.config;qualifications={};started=time.monotonic()
 for arm in ('parent3504','EAL2160'):
  dest=out/arm;dest.mkdir()
  if arm=='EAL2160':
   assert pin(eal['checkpoint']['path'])['sha256']==eal['checkpoint']['sha256']
   saved=torch.load(eal['checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False)
   assert saved['schema']=='shortlist_objective_full48_v1' and saved['step']==5664 and saved['config']==asdict(original)
   copy_named_state(model,saved,joint_parameters,binary_buffers);assert sign_hashes(model,joint_parameters)==saved['final_sign_hashes']
   opt.load_state_dict(saved['optimizer']);assert equal(opt.state_dict(),saved['optimizer']) and all(int(v['step'])==5664 for v in opt.state.values())
   _restore_block_rng(saved['rng'],model.token_embd.device);assert equal(saved['rng'],_block_rng_state(model.token_embd.device));del saved;gc.collect()
   trial=dest/'zero-update.gguf';receipt=native_export(model,trial,parent=parent);assert receipt['output']['sha256']==eal['model']['sha256'];(dest/'zero-update-export.json').write_text(json.dumps(receipt,indent=2));trial.unlink()
  rng=_block_rng_state(model.token_embd.device);rows=[]
  with torch.no_grad():
   for w in data['windows']:
    batch=tensor_batch(block(w,pin(RUN/'historical-matrix.json')));arrays={};preds={}
    for mode,conditioning in [('teacher_forced','captured_prefix'),('greedy','native_greedy')]:
     model.config=replace(original,conditioning=conditioning);output=model(batch);raw=output.logits.detach().cpu().numpy().astype('<f4');path=dest/f'{w["case_index"]:03d}-{mode}.f32';raw.tofile(path);arrays[mode]=pin(path);preds[mode]=score(raw,w['labels']);predecessors=output.predecessor_ids.cpu().tolist()
     expected=[w['anchor_token'],*(w['labels'][:6] if mode=='teacher_forced' else raw.argmax(-1).tolist()[:6])];assert predecessors==expected
     del output,raw
    rows.append({'window_index':w['case_index'],'panel':w['panel'],'role':w['role'],'domain':w['domain'],'group_id':w['group_id'],'teacher_forced':preds['teacher_forced'],'greedy':preds['greedy'],'raw':arrays})
    (out/'progress.json').write_text(json.dumps({'arm':arm,'windows':len(rows),'wall_seconds':time.monotonic()-started}))
  model.config=original;_restore_block_rng(rng,model.token_embd.device);assert equal(rng,_block_rng_state(model.token_embd.device))
  result={'arm':arm,'rows':rows,'zero_update_export_byteexact':True,'optimizer_restored_no_steps':True,'rng_restored':True,'summary':{p:{role:{'windows':len(rr:=[r for r in rows if r['panel']==p and r['role']==role]),'prefix5':sum(r['greedy']['prefix'] for r in rr),'TF_CE':float(np.mean([v for r in rr for v in r['teacher_forced']['ce']])),'TF_top1':sum(sum(r['teacher_forced']['top1']) for r in rr)} for role in ('historical_seen','historical_not_seen')} for p in ('error_enriched','ordinary')}}
  (dest/'result.json').write_text(json.dumps(result,indent=2));qualifications[arm]=result['summary'];print(json.dumps({'arm':arm,'summary':result['summary']}),flush=True)
 (out/'result.json').write_text(json.dumps({'status':'CURRENT_ACTOR_COMPLETE','summary':qualifications,'wall_seconds':time.monotonic()-started,'TRAIN_seconds':0,'updates':0,'precision':'Original decoded F32 frozen-operand/native-A8 approximate actor, original codec/13 binary projections. Greedy mode changes only predecessor selection; teacher_forced is authentic training forward.'},indent=2))
if __name__=='__main__':main()
