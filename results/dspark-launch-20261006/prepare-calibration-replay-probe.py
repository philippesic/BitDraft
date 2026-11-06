"""Root calibration evidence preparation; never TRAIN-cache admission."""
import json,sys
from pathlib import Path
ROOT=Path('/home/philip/binary-eagle-decoding')
CHECKOUT=ROOT/'runs/checkouts/dspark-full-training-20261006-01'
sys.path.insert(0,str(CHECKOUT/'src'))
from w1a1_eagle.block_data import BlockDataset
from w1a1_eagle.block_shard_lifecycle import (FrozenShardPlan,ShardCache,pin,tree_bytes,
    rebind_physical_manifest,validate_capture_source,preserve_original_metadata,archive_shard)
from w1a1_eagle.nine_model_pipeline import atomic_json,LinuxResources
OUT=ROOT/'runs/dspark-a8-full-pool-20261006-01'
PL=pin(OUT/'logical-plan-v2.json');plan=FrozenShardPlan.load(PL)
original=OUT/'calibration-0000';report=json.loads((original/'capture-report.json').read_text())
assert report['status']=='PASS' and report['producer_closed'] is True
resources=LinuxResources('GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e');resources.require_released([],[])
reb=rebind_physical_manifest(plan,report['manifests']['dspark'],pin(original/'dspark/completed-admission.json'),original/'logical-binding-v2')
ds=BlockDataset(reb['data']['path'],expected_sha256=reb['data']['sha256'],admission_path=reb['admission']['path'],admission_sha256=reb['admission']['sha256'],allow_synthetic=False)
try:
 validate_capture_source(plan,'chunk-0000',ds)
 assert set(ds.chains)==set(plan.shards['chunk-0000'])
 assert {c['domain'] for c in ds.chains.values()}=={'code','prose','reasoning'}
 seal={cid:{'artifacts':{k:c[k]['sha256'] for k in ('tokens','features','logits','native_receipt')},'anchors':c['anchors'],'prompt_length':c['prompt_length'],'logits_indices':c.get('logits_indices')} for cid,c in ds.chains.items()}
finally:ds.close()
assert tree_bytes(original)<8*1024**3
# This cache-shaped evidence ledger is explicitly calibration-only. It does not
# call publish or claim the 24GiB TRAIN cache has admitted these 64GiB sources.
cache=ShardCache(plan,OUT)
assert not cache.state['resident'] and not cache.state['seals']
cache.state['resident']['chunk-0000']={'manifest':reb['data'],'admission':reb['admission'],'directory':str(original)}
cache.state['seals']['chunk-0000']={'identity':seal,'recovery':None}
cache._save()
metadata=preserve_original_metadata(cache,'chunk-0000',ROOT/'results/dspark-launch-20261006/calibration-original-records',max_bytes=2*1024**3)
archive=archive_shard(cache,'chunk-0000',ROOT/'runs/dspark-a8-calibration-replay-archive-20261006-01',max_bytes=8*1024**3,min_free_bytes=10*1024**3)
receipt={'schema':'dspark_calibration_probe_baseline_v1','status':'PASS_ACTUAL_BASELINE_AND_ARCHIVE_ONLY','train_cache_admission':False,'logical_plan':PL,'plan_sha256':plan.sha256,'shard':'chunk-0000','directory':str(original),'original_metadata':metadata,'original_identity':seal,'verified_archive':archive,'original_logical_manifest':reb['data'],'calibration_evidence_budget_bytes':64*1024**3,'source':plan.value['source']}
atomic_json(ROOT/'results/dspark-launch-20261006/calibration-probe-baseline.json',receipt)
request={'schema':'block_shard_restore_request_v1','plan':PL,'plan_sha256':plan.sha256,'shard':'chunk-0000','directory':str(original),'original_identity':seal,'original_metadata':metadata,'generation_variant':plan.value['generation_variant'],'capture_bytes_bound':8*1024**3,'replay_probe':True,'replay_admission':None}
atomic_json(ROOT/'results/dspark-launch-20261006/calibration-cold-replay-request.json',request)
# All baseline bytes were archived and reverified before canonical removal.
# The actual cold replay runs next; no replay PASS/eviction permission is inferred.
cache.acquire('root-calibration-replay-probe','restore')
cache.evict('chunk-0000')
print(json.dumps({'status':'ARCHIVED_COLD_TRIAL_READY','baseline':pin(ROOT/'results/dspark-launch-20261006/calibration-probe-baseline.json'),'request':pin(ROOT/'results/dspark-launch-20261006/calibration-cold-replay-request.json'),'archive':archive,'optimizer_updates':0}),flush=True)
