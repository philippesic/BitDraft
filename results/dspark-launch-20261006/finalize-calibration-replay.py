import json,sys,shutil
from pathlib import Path
ROOT=Path('/home/philip/binary-eagle-decoding');OUT=ROOT/'runs/dspark-a8-full-pool-20261006-01';CHECKOUT=ROOT/'runs/checkouts/dspark-full-training-20261006-01'
sys.path.insert(0,str(CHECKOUT/'src'))
from w1a1_eagle.block_shard_lifecycle import FrozenShardPlan,ShardCache,pin,checked,tree_bytes,admit_replay,validate_replay_admission
from w1a1_eagle.nine_model_pipeline import LinuxResources,atomic_json
state=json.loads((CHECKOUT/'runs/dspark-a8-cold-replay-20261006-01/state.json').read_text());assert state['status']=='finished' and state['exit_code']==0
for pid in [81650,81655]:assert not (Path('/proc')/str(pid)).exists()
resources=LinuxResources('GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e');resources.require_released([],[])
plan=FrozenShardPlan.load(pin(OUT/'logical-plan-v2.json'));cache=ShardCache(plan,OUT)
publication=pin(ROOT/'results/dspark-launch-20261006/calibration-cold-replay-publication.json');pub=json.loads(checked(publication).read_text())
admission=admit_replay(cache,'chunk-0000',publication,pin(CHECKOUT/'scripts/replay_dspark_shard.py'),ROOT/'results/dspark-launch-20261006/replay-admission.json')
validate_replay_admission(plan,admission)
# Preserve original nonteacher/golden bytes from the actual verified archive.
# Existing recaptured files must already match originals; never replace them.
baseline=json.loads((ROOT/'results/dspark-launch-20261006/calibration-probe-baseline.json').read_text());arch=json.loads(checked(baseline['verified_archive']).read_text());archive=checked(arch['manifest']).parent;canonical=Path(baseline['directory'])
restored=[]
for old in archive.rglob('*'):
 if not old.is_file() or old.name in {'archive-manifest.json','recovery.json'}:continue
 rel=old.relative_to(archive);dest=canonical/rel
 if dest.exists():
  assert pin(dest)['sha256']==pin(old)['sha256'],str(dest)
 else:
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,dest);assert pin(dest)['sha256']==pin(old)['sha256'];restored.append(str(rel))
cache.state['resident']['chunk-0000']={'manifest':pub['manifest'],'admission':pub['admission'],'directory':pub['directory']};cache._save();cache.release('root-calibration-replay-probe',released_proof=True)
validate_replay_admission(plan,admission)
raw=tree_bytes(canonical);assert raw<=8*1024**3
receipt={'schema':'dspark_actual_cold_replay_release_v1','status':'PASS','replay_admission':admission,'actual_cold_publication':publication,'actual_original_tensor_identity_equal':True,'domains':['prose','code','reasoning'],'test_prompts':10,'original_nonteacher_files_restored':restored,'retained_bytes':raw,'owned_gpu_release':resources.snapshot(),'optimizer_updates':0,'train_cache_admission':False}
atomic_json(ROOT/'results/dspark-launch-20261006/cold-replay-finalized.json',receipt)
print(json.dumps(receipt),flush=True)
