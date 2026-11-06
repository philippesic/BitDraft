"""Bound native head arithmetic drift on actual captured hidden states, full vocab."""
import argparse,gc,json,os,subprocess,sys
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .archive import hydrate,release
from .measure import completed
from .quant import target_tensor
from scripts.dflash_baseline.model import GGUFReader,dequantize

def compile_fixture():
 source=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';runtime=R/'runtime04';cpp=ROOT/'scripts/component_campaign/replay.cpp';exe=R/'replay-binaries'/pin(cpp)['sha256'][:12];exe.parent.mkdir(exist_ok=True)
 command=['c++','-O2','-std=c++17','-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-I'+str(source/'vendor/nlohmann'),str(cpp),'-L'+str(runtime/'bin'),'-lllama','-lggml-base','-lggml','-ldl','-Wl,-rpath,'+str(runtime/'bin'),'-o',str(exe)]
 if not exe.exists():
  subprocess.run(command,check=True);save(exe.with_suffix('.build.json'),dict(command=command,cpp=pin(cpp),executable=pin(exe),compile_runtime=pin(runtime/'build.json')))
 return exe

def matrix(prefix,kind,meta):
 raw=np.fromfile(str(prefix)+'-'+kind+'.bin',dtype=np.uint8);ne=meta[kind+'_ne'];nb=meta[kind+'_nb']
 if meta[kind+'_type'].upper()!='F32' or ne[2:]!=[1,1]:raise ValueError('Unexpected native head operand layout')
 return np.ndarray((ne[1],ne[0]),dtype='<f4',buffer=raw,strides=(nb[1],nb[0])).copy()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--compile-only',action='store_true');ap.add_argument('--arms',nargs='+',default=['H0','H4','H3','HE4','H4R2','H6R2','HE4R2','HE6R2']);ap.add_argument('--tag',default='head-forward');a=ap.parse_args();compile_fixture()
 if a.compile_only:print('HEAD_CAPTURE_CPP_BUILD_PASS_NATIVE_PENDING',flush=True);return
 out=R/'qualification'/a.tag;out.mkdir(exist_ok=True)
 if (out/'verified.json').exists():return
 names=a.arms;registry=completed();models=dict(q4=str(Q4),**{n:registry[n]['model'] for n in names});save(out/'arms.json',models)
 if not (out/'replay/result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.replay',str(out/'arms.json'),'--runtime',str(R/'runtime04'),'--out',str(out/'replay'),'--reps','1','--audit','--head-capture'],cwd=ROOT,check=True)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;records=[];failures=[]
 for name,model_path in models.items():
  pause();private=name!='q4' and bool(registry[name]['spec'].get('head'));created=hydrate(model_path) if private else False
  try:
   tensor=next(t for t in GGUFReader(model_path).tensors if t.name=='output.weight') if private else target_tensor(TARGET)
   weight=torch.from_numpy(dequantize(tensor.data,tensor.tensor_type).reshape(248320,2560)).cuda()
   for domain in ('code','prose','reasoning'):
    prefix=out/'replay'/('00-'+name+'-'+domain);meta=json.load(open(str(prefix)+'-head.json'));hidden=matrix(prefix,'input',meta);native=matrix(prefix,'output',meta)
    if hidden.shape!=(2,2560) or native.shape!=(2,248320) or meta['weight_type'].upper()!=tensor.tensor_type.name:raise ValueError('Full required native head dimensions/format differ')
    capture=P/f'runs/dflash-modern-capture-20261104-04/data/development-{domain}-00/capture/capture.json';cap=json.load(open(capture));labels=dict(zip(cap['logit_positions'],cap['greedy_ids']));wanted=torch.tensor([labels[meta['anchor']+i] for i in range(2)],device='cuda')
    x=torch.tensor(hidden,device='cuda',requires_grad=True);predicted=F.linear(x,weight);loss=F.cross_entropy(predicted,wanted);loss.backward();sim=predicted.detach().cpu().numpy();error=np.abs(native-sim)
    if not np.isfinite(native).all() or not np.isfinite(sim).all() or not torch.isfinite(x.grad).all() or not torch.count_nonzero(x.grad):raise ValueError('Nonfinite head arithmetic/upstream gradient')
    rms=float(np.sqrt(np.mean(np.square(native-sim))));maximum=float(error.max());changes=[]
    for i in range(2):
     ni=int(native[i].argmax());si=int(sim[i].argmax());top=np.partition(native[i],-2)[-2:];margin=float(top.max()-top.min());label=int(wanted[i]);changes.append(dict(native_argmax=ni,simulation_argmax=si,agreement=ni==si,native_winner_margin=margin,max_absolute_error=float(error[i].max()),native_target_rank=int((native[i]>native[i,label]).sum()+1),simulation_target_rank=int((sim[i]>sim[i,label]).sum()+1)))
    native_loss=float(F.cross_entropy(torch.from_numpy(native),wanted.cpu()));row=dict(arm=name,domain=domain,anchor=meta['anchor'],format=tensor.tensor_type.name,vocabulary=248320,hidden_width=2560,required_columns=2,max_absolute_error=maximum,rms_error=rms,native_target_ce=native_loss,simulation_target_ce=float(loss),upstream_gradient_norm=float(x.grad.norm()),decisions=changes,native_input=pin(str(prefix)+'-input.bin'),native_output=pin(str(prefix)+'-output.bin'),capture_metadata=pin(capture))
    row['gate_pass']=rms<=.03 and maximum<=.25 and not any(not c['agreement'] and c['native_winner_margin']>.05 for c in changes)
    if not row['gate_pass']:failures.append(dict(arm=name,domain=domain,rms=rms,maximum=maximum,decisions=changes))
    records.append(row);del x,predicted,loss;gc.collect()
   del weight;gc.collect();torch.cuda.empty_cache()
  finally:
   if private:release(model_path,created)
  print('FULL_NATIVE_HEAD_FORWARD_AUDITED',name,flush=True)
 result=dict(status='FULL_NATIVE_HEAD_FORWARD_GRADIENT_PASS' if not failures else 'NUMERIC_GATE_REQUIRES_REVIEW',records=records,failures=failures,replay=pin(out/'replay/result.json'),source=pin(__file__),cpp=pin(ROOT/'scripts/component_campaign/replay.cpp'),gate=dict(full_vocab=248320,contexts=3,required_columns=2,rms_error_max=.03,max_absolute_error=.25,decision_change_requires_review_above_native_margin=.05),scope='Actual native captured histories and normalized head inputs; every vocabulary logit compared with exact dequantized packed weights used by QAT. Native activation quantization/reduction drift is bounded, not claimed bit-exact. Labels are immutable native target labels. Upstream gradient covers full vocabulary. Audit timing excluded; no new TRAIN or heldout exposure.')
 result['output_retention']='On a passing gate, only NEW full native head-output scratch is retired after pinning; actual input vectors, layout metadata, full-output hashes, discrepancy/rank/decision metrics, replay, native model/runtime/source pins remain. Models/checkpoints/captures are preserved. Re-running the diagnostic reconstructs the outputs.'
 save(out/'result.json',result)
 if failures:raise ValueError('Native head arithmetic requires bounded review; evidence preserved')
 save(out/'verified.json',result)
 # Full outputs are NEW numeric scratch; preserve hashes, input vectors and all
 # discrepancy/rank/decision metrics. No checkpoint/export/capture is removed.
 for row in records:
  p=Path(row['native_output']['path'])
  if pin(p)!=row['native_output']:raise ValueError('Numeric output scratch changed')
  p.unlink()
 print('HEAD_FORWARD_FULL_VOCAB_QUALIFIED',len(records),flush=True)

if __name__=='__main__':main()
