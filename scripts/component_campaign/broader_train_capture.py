"""Capture admitted TRAIN inputs with the pinned native DFlash taps and hard labels.

Only call in the owned GPU lane after a recorded recovery decision. Earlier
EAGLE captures contribute input ancestry only; none of their features are reused.
"""
import hashlib,json,os,re,shutil,subprocess
import numpy as np
from .common import *

def main():
 plan=R/'broader-train-plan/result.json';planned=json.load(open(plan));decision=R/'selection-evidence.json'
 if not decision.exists():raise ValueError('record the evidence-led rescue/data decision before capture')
 out=R/'broader-train-capture';out.mkdir(exist_ok=True)
 if (out/'manifest.json').exists():return
 expected=int(planned['planned_feature_bytes'])+int(planned['planned_one_cell_logit_transient_bytes'])+(1<<30)
 if shutil.disk_usage('/').free<expected:raise RuntimeError('Broader native TRAIN capture needs'+str(expected)+'WSL bytes free including1GiB headroom')
 template=json.load(open(P/'runs/dflash-modern-capture-20261104-04/data/fit-code-00/command.json'));records=[]
 if pin(TARGET)!=json.load(open(R/'freeze.json'))['target']:raise ValueError('frozen native target changed')
 for row in planned['records']:
  pause();cell=out/row['id'];cell.mkdir(exist_ok=True);capture=cell/'capture';proofpath=cell/'hard-label-proof.json'
  if pin(row['tokens']['path'])!=row['tokens']:raise ValueError('prepared native TRAIN tokens changed')
  tokens=[int(x) for x in Path(row['tokens']['path']).read_text().split()]
  if len(tokens)!=row['prompt_tokens']:raise ValueError('native input geometry changed')
  args=template['argv'].copy();i=args.index('--decode-tokens');del args[i:i+2]
  for key,value in [('--tokens',row['tokens']['path']),('--out',str(capture)),('--output-cap','129'),('--logit-positions',','.join(map(str,range(len(tokens)-1,len(tokens)+128)))),('--max-output-mib','256')]:args[args.index(key)+1]=value
  args+=['--greedy-steps','128'];env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','W1A8_','LLAMA_','CUDA_','LD_','OMP_'))};env.update(template['environment'])
  if not (capture/'capture.json').exists():
   save(cell/'command.json',dict(argv=args,environment=template['environment'],source=pin(args[0]),libraries={key:pin(args[args.index(key)+1]) for key in ('--cpu-library','--cuda-library')},template=pin(P/'runs/dflash-modern-capture-20261104-04/data/fit-code-00/command.json'),input=row,owner=identity()))
   with (cell/'capture.log').open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  meta=json.load(open(capture/'capture.json'))
  if meta['model']!=str(TARGET) or meta['input_tokens'][:len(tokens)]!=tokens or meta['taps']!=[2,6,10,14,18,22,26,30] or meta['feature_semantics']!='zero_based_pre_block_input_all_tokens' or meta['settings']['consistent']!=32 or meta['settings']['kv']!='F16':raise ValueError('native DFlash TRAIN capture contract differs')
  logits_path=capture/'logits.f32'
  if logits_path.exists():
   logits=np.memmap(logits_path,mode='r',dtype='<f4',shape=tuple(meta['logits_shape']))
   if not np.isfinite(logits).all() or logits.argmax(-1).tolist()!=meta['greedy_ids']:raise ValueError('native hard labels differ from full actual logits')
   proof=dict(status='ALL_NATIVE_FULL_VOCAB_ARGMAX_LABELS_VERIFIED',logits=pin(logits_path),metadata=pin(capture/'capture.json'),shape=meta['logits_shape'],greedy_ids=meta['greedy_ids'],logit_positions=meta['logit_positions'],row_sha256=[hashlib.sha256(x).hexdigest() for x in logits],native_inputs=row,retention='NEW disposable full-logit scratch removed after exact per-row argmax/hash verification. Native8-tap features, labels, inputs, producer/runtime pins and receipts retained; hard-label CE objective only. All PRIOR paid captures untouched.')
   del logits;save(proofpath,proof)
   if pin(logits_path)!=proof['logits']:raise ValueError('logit scratch changed before safe retirement')
   logits_path.unlink()
  proof=json.load(open(proofpath))
  if proof['metadata']!=pin(capture/'capture.json'):raise ValueError('label proof ancestry changed')
  # Stop supervision at the first actual end-of-generation token. Keep original
  # captured arrays/metadata intact; this is an explicit new training-view rule.
  eos_ids={int(x) for x in re.findall(r'EOG token\s+=\s+(\d+)',(cell/'capture.log').read_text())}
  if not eos_ids:raise ValueError('native EOG token list missing')
  end=next((i for i in range(len(tokens),len(meta['input_tokens'])) if meta['input_tokens'][i] in eos_ids),len(meta['input_tokens'])-1)
  records.append(dict(name='broader-'+row['id'],metadata=pin(capture/'capture.json'),split='TRAIN',files=[*[pin(capture/f'layer-{t}.f32') for t in meta['taps']],pin(proofpath)],TRAIN_group_id=row['group_id'],input_ancestry=row,training_last_input_position=end,native_eog_ids=sorted(eos_ids),stop_policy='Only supervision through first native EOG, including its prediction; no post-EOG labels; original captures retained.',hard_label_proof=pin(proofpath)))
  save(out/'progress.json',records);print('BROADER_NATIVE_TRAIN_CAPTURED',row['id'],end-len(tokens)+1,'eligible continuation tokens',flush=True)
 original=json.load(open(R/'freeze.json'))['B0_config']['data']
 actual_original=pin(original['path'])
 if any(actual_original[key]!=value for key,value in original.items()):raise ValueError('original training manifest changed')
 baseline=json.load(open(original['path']))
 save(out/'manifest.json',dict(status='ADMITTED_NATIVE_DFLASH_BROADER_TRAIN',records=baseline['records']+records,taps=[2,6,10,14,18,22,26,30],original_manifest=original,new_input_plan=pin(plan),selection_evidence=pin(decision),target=pin(TARGET),producer_template=pin(P/'runs/dflash-modern-capture-20261104-04/data/fit-code-00/command.json'),source=pin(__file__),policy='New matched-control recipe only; previous R1history/data unchanged. Native TRAIN bank inputs permanently excluded from heldout; earlier EAGLE feature/label files never reused.'))
 print('BROADER_NATIVE_TRAIN_MANIFEST_READY',flush=True)

if __name__=='__main__':main()
