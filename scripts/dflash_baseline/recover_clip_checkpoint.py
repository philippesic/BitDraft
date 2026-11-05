"""Full trained streamed-state recovery, exported against a resident-model point."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from scripts.dflash_baseline.checkpoint_io import load
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.precision_benchmark import pin,save
from scripts.dflash_baseline.prototype import digest
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';U=R/'clip-control-followup'
def trainable_names(config,parameters):
 suffixes={'latent_sign':'sign','scale':'scale','midpoint':'midpoint','log_gain':'gain'}
 return {name for name in parameters if (config['quantizer']['mode']=='learned' if name.endswith('.log_clip') else config['features'][suffixes.get(name.rsplit('.',1)[-1],'norm')])}
def main():
 torch.set_num_threads(2);plan=json.load(open(U/'plan.json'));out=U/'training';path=out/'checkpoint.pt.zst';saved=load(path)
 foundation=json.load(open(R/'foundation/result.json'))['progress'];progress=saved['progress'];learned=json.load(open(R/'interactions/G4D8-learned-small/result.json'))['progress']
 if saved['config_hash']!=plan['resolved_hash'] or digest(saved['config'])!=plan['resolved_hash'] or progress['updates']!=1024 or progress['cursor']!=foundation['cursor']+1024 or progress['supervised_tokens']!=15360 or progress['order']!=foundation['order']:raise ValueError('fixed control ancestry/exposure mismatch')
 if any(progress[k]!=learned[k] for k in ('updates','cursor','order','supervised_tokens')):raise ValueError('fixed/learned history mismatch')
 if any(not torch.isfinite(x).all() for x in saved['parameters'].values()):raise ValueError('nonfinite recovered parameters')
 states=saved['optimizer']['state'];steps={int(x['step']) for x in states.values()}
 expected=trainable_names(saved['config'],saved['parameters'])
 if steps!={1024} or len(states)!=len(expected) or sum(len(x['params']) for x in saved['optimizer']['param_groups'])!=len(expected):raise ValueError('optimizer coverage/steps not recovered')
 if any(not torch.isfinite(t).all() for x in states.values() for t in x.values() if torch.is_tensor(t)):raise ValueError('nonfinite recovered Adam state')
 if len(saved['rng'])!=4 or saved['rng'][3] is None:raise ValueError('CUDA RNG state missing')
 result=export(path,out/'model-final.gguf',state=saved,checkpoint_pin=pin(path));resident=pin(out/'model-1024.gguf')
 if result['sha256']!=resident['sha256'] or result['bytes']!=resident['bytes']:raise ValueError('full streamed-state native export differs from resident-model export')
 save(U/'checkpoint-recovery.json',dict(status='FULL_STREAMED_CUDA_CHECKPOINT_ROUNDTRIP_PASS',checkpoint=pin(path),config_hash=saved['config_hash'],optimizer_states=len(states),trainable_parameters=len(expected),frozen_parameters=len(saved['parameters'])-len(expected),optimizer_step=1024,cuda_rng_states=len(saved['rng'][3]),matched_updates=progress['updates'],matched_cursor=progress['cursor'],supervised_positions=progress['supervised_tokens'],model=pin(out/'model-final.gguf'),resident_model=resident,source=pin(__file__),scope='Full trained F32 parameters/Adam/RNG/sampler restored from streamed Zstd; export bytes equal model-resident1024 point. No frozen-buffer duplication. Decode temporary resides in owned RAM filesystem.'))
 print('Full streamed CUDA optimizer/RNG/native-export recovery passed',flush=True)
if __name__=='__main__':main()
