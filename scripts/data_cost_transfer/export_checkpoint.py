"""Export a paid fixed-representation intermediate checkpoint without more training."""
from pathlib import Path
import argparse,importlib.util,sys
P=Path('/home/philip/binary-eagle-decoding')
sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import pin,read,atomic,require
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
require(not a.out.exists(),'export destination exists')
spec=importlib.util.spec_from_file_location('_fixed_export',P/'scripts/design_readiness/fixed_objective.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.torch.set_num_threads(2)
saved=m.torch.load(a.checkpoint,map_location='cpu',weights_only=False)
require(saved['schema']=='modern_fixed_data_transfer_checkpoint_v1' and saved['cursor'] in (6240,6720,7680),'bounded paid checkpoint required')
freeze=read(saved['contract']['freeze']['path']);require(pin(saved['contract']['freeze']['path'])==saved['contract']['freeze'],'training freeze changed')
for ref in freeze['sources']:require(pin(ref['path'])==ref,'training dependency changed')
def unused(*args,**kwargs):raise RuntimeError('export must not run model inference')
lookup=m.ExternalNativeF32Embedding(lookup=unused,vocab_size=248320,hidden_size=2560,target_identity='unchanged native target; export only',target_sha256='3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630',capture_sha256='0'*64)
actor=m.build_modern_actor(P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',P/'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json',embedding_lookup=lookup,device='cpu',scale_only=True)
require(actor.contract==saved['contract']['actor'],'actor representation differs')
m.load_scale_state(actor,saved['scales']);a.out.mkdir(parents=True)
export=m.export_scale_only_gguf(actor,a.out/'model.gguf')
atomic(a.out/'result.json',dict(status='TRAINED_EXPORTED_NOT_NATIVE_ADMITTED',checkpoint=pin(a.checkpoint),model=export['output'],contract=saved['contract'],cursor=saved['cursor'],local_exposure=saved['local_exposure'],local_trainer_seconds=saved['local_trainer_seconds'],source=pin(__file__),scope='Export existing checkpoint; zero additional updates'))
