import json,sys,subprocess,time
from pathlib import Path
ROOT=Path('/home/philip/binary-eagle-decoding');OUT=ROOT/'runs/dspark-a8-full-pool-20261006-01';CHECKOUT=ROOT/'runs/checkouts/dspark-full-training-20261006-01'
sys.path.insert(0,str(CHECKOUT/'src'))
from w1a1_eagle.block_shard_lifecycle import FrozenShardPlan,pin,merge_preparation_manifests
from w1a1_eagle.nine_model_pipeline import atomic_json
final=json.loads((ROOT/'results/dspark-launch-20261006/cold-replay-finalized.json').read_text());assert final['status']=='PASS'
plan=FrozenShardPlan.load(pin(OUT/'logical-plan-v2.json'));pub=json.loads((ROOT/'results/dspark-launch-20261006/calibration-cold-replay-publication.json').read_text())
records=[{'manifest':pub['manifest'],'admission':pub['admission']}]
for n in range(1,14):
 d=OUT/f'calibration-{n:04d}';report=json.loads((d/'capture-report.json').read_text());assert report['status']=='PASS' and report['producer_closed'] is True;records.append({'manifest':report['manifests']['dspark'],'admission':pin(d/'dspark/completed-admission.json')})
d=OUT/'train-0000';report=json.loads((d/'capture-report.json').read_text());assert report['status']=='PASS';records.append({'manifest':report['manifests']['dspark'],'admission':pin(d/'dspark/completed-admission.json')})
atomic_json(OUT/'preparation-records.json',records)
merged=merge_preparation_manifests(plan,records,OUT/'preparation-merged');print('MERGE',json.dumps(merged),flush=True)
refs=ROOT/'data/block-production-assets-20261006-01/results/nine-model-qat-preparation/block-fusion-references-20261004/dspark'
weights=pin(refs/'fc-weight.npy');norm=pin(refs/'fc-norm.npy');meta=pin(refs/'reference.json');desc=json.loads((refs/'reference.json').read_text())
args=[sys.executable,str(CHECKOUT/'scripts/fit_block_fusion.py'),'--manifest',merged['data']['path'],'--manifest-sha256',merged['data']['sha256'],'--admission',merged['admission']['path'],'--admission-sha256',merged['admission']['sha256'],'--weights',weights['path'],'--weights-sha256',weights['sha256'],'--norm',norm['path'],'--norm-sha256',norm['sha256'],'--norm-metadata',meta['path'],'--norm-metadata-sha256',meta['sha256'],'--norm-epsilon',str(desc['epsilon']),'--activation-bits','8','--coordinate-flips','8','--rows-per-chain','32','--max-total-rows','3072','--max-array-bytes',str(3*1024**3),'--max-seconds','1800','--latent-initialization','preserve_reference_magnitudes','--output-dir',str(OUT/'fit-a8')]
atomic_json(ROOT/'results/dspark-launch-20261006/actual-fusion-fit-command.json',{'argv':args,'optimizer_updates':0,'calibration_fit_prompts':96,'calibration_validation_prompts':48,'scope':'CPU_FUSION_FIT_NOT_SM120_TRAINING'})
result=subprocess.run(args,check=False);assert result.returncode==0,result.returncode
print('ACTUAL_CPU_FIT_COMPLETE',flush=True)
