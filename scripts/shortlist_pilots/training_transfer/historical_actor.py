"""One authentic historical actor process per checkpoint with unchanged resume guards."""
from pathlib import Path
from dataclasses import replace
import sys,json,time,gc,hashlib,argparse
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def pin(p):
 with Path(p).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h}
def main():
 arm=sys.argv[1];anchors=json.loads((RUN/'historical-anchors-qualified.json').read_text());q=anchors[arm];out=RUN/'historical-actor'/arm;out.mkdir(parents=True,exist_ok=False)
 try:
  sources=q['matching_source_roots'] or (anchors['16hour']['matching_source_roots'] if arm=='initializer' else [])
  if not sources:raise ValueError('No complete authentic runtime-source match; legacy guards remain unchanged')
  source=Path(sources[-1]);sys.path[:0]=[str(source/'src'),str(source/'scripts'),str(source/'third_party/llama.cpp/gguf-py'),str(ROOT/'third_party/llama.cpp/gguf-py'),str(ROOT/'scripts'),str(ROOT/'runs/overnight-quality-20261027/source')]
  import torch,numpy as np
  from w1a1_eagle.block_qat import BlockQATConfig,BlockDrafter
  from w1a1_eagle.block_training import load_block_gguf,block_optimizer,load_block_checkpoint,restore_original_scale_only_initializer,export_block_checkpoint,_block_rng_state,_restore_block_rng
  from train_nine_model_qat import calibration,tensor_batch,configure_cuda,validate_initializer_reference
  from export_block_binary import export_model
  from actor_gate import block
  from shortlist_pilots.training_transfer.metrics import score
  torch.set_num_threads(1);torch.set_num_interop_threads(1);hardware=configure_cuda(zero_updates=False);sp=json.loads((RUN/'historical-original-spec.json').read_text());cfg=BlockQATConfig(**q['contract']['config'])
  if arm=='initializer':cfg=replace(cfg,parameter_training_policy='scale_only')
  tensors=load_block_gguf(sp['model']['path'],sp['model']['sha256'],cfg);initializer=calibration(sp['initialization'],cfg);model=BlockDrafter(tensors,cfg,binary_initializer=initializer);validate_initializer_reference(getattr(model.fc,'initialization_report',{}),sp['initialization'],cfg.family);del tensors,initializer;gc.collect();model.to('cuda:0');optimizer=block_optimizer(model)
  if arm=='initializer':optimizer,transition=restore_original_scale_only_initializer(model,q['source'],q['checkpoint'],original_runtime_source=q['runtime']['source']);cursor=q['checkpoint']['cursor']
  else:cursor=load_block_checkpoint(model,optimizer,q['source'],q['checkpoint']);transition=None
  exported=export_block_checkpoint(model,q['source'],out/'zero-update');trial=out/'zero-update/model.gguf';audit=export_model(Path(sp['model']['path']),Path(exported['npz']),Path(exported['manifest']),trial);assert audit['output']['sha256']==q['model']['sha256'],'historical zero-update native export differs'
  (out/'qualification.json').write_text(json.dumps({'status':'AUTHENTIC_RESTORE_EXPORT_PASS','source':str(source),'checkpoint':q['checkpoint'],'model':q['model'],'zero_update_export':audit,'source_contract_unchanged':True,'initializer_transition':transition,'hardware':hardware,'TRAIN_seconds':0,'updates':0},indent=2));trial.unlink();Path(exported['npz']).unlink()
  model.eval();original=model.config;data=json.loads((RUN/'historical-matrix.json').read_text());rng=_block_rng_state(model.token_embd.device);rows=[];started=time.monotonic()
  with torch.no_grad():
   for w in data['windows']:
    batch=tensor_batch(block(w,pin(RUN/'historical-matrix.json')));arrays={};scores={}
    for mode,cond in [('teacher_forced','captured_prefix'),('greedy','native_greedy')]:
     model.config=replace(original,conditioning=cond);output=model(batch);raw=output.logits.detach().cpu().numpy().astype('<f4');path=out/f'{w["case_index"]:03d}-{mode}.f32';raw.tofile(path);arrays[mode]=pin(path);scores[mode]=score(raw,w['labels']);scores[mode+'_D7']=score(raw,w['labels'],depth=7)
     expected=[w['anchor_token'],*(w['labels'][:6] if mode=='teacher_forced' else raw.argmax(-1).tolist()[:6])];assert output.predecessor_ids.cpu().tolist()==expected;del output,raw
    rows.append({'window_index':w['case_index'],'panel':w['panel'],'domain':w['domain'],'group_id':w['group_id'],'seen':w['checkpoint_group_exposure'][arm],**scores,'raw':arrays});(out/'progress.json').write_text(json.dumps({'windows':len(rows),'wall_seconds':time.monotonic()-started}))
  model.config=original;_restore_block_rng(rng,model.token_embd.device);result={'status':'HISTORICAL_ACTOR_COMPLETE','arm':arm,'rows':rows,'wall_seconds':time.monotonic()-started,'TRAIN_seconds':0,'updates':0};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
 except Exception as e:
  (out/'failure.json').write_text(json.dumps({'status':'HISTORICAL_ACTOR_QUALIFICATION_FAILED','error':repr(e),'arm':arm,'no_tolerance_or_source_guard_relaxation':True},indent=2));raise
if __name__=='__main__':main()
