import json,sys
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';O=R/'runs/dspark-a8-full-pool-20261006-01';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.block_shard_lifecycle import pin
from w1a1_eagle.nine_model_pipeline import atomic_json
from prepare_nine_model_lane import build_lane
plan=json.loads((O/'timed-plan-source-v3/timed-evaluation-plan.json').read_text())
policy=D/'reviewed-research-policy-v1.json'
assert pin(policy)['sha256']=='127df3120ed439d15bdff04331fc5ac808535c4d7a13a6bf646fe231e08dd738'
plan['research_continuation_policy']=pin(policy)
for name in ['reviewed-research-policy-v1.json','research-operational-review-v1.json','research-historical-case-authority-v1.json','source-v3-export-equivalence.json']:
 plan['source']['artifact:research:'+name]=pin(D/name)
pt=O/'timed-evaluation-plan.reviewed-v3.json';assert not pt.exists();atomic_json(pt,plan)
v=json.loads((O/'production-inputs.canonical-alias-v3.json').read_text())
v['qa_ledger']=pin(D/'portable-qa-v3/qa-ledger.json')
v['inputs']['admission_plan']=pin(O/'admission-plan.canonical-v3.json')
v['timed_evaluation_plan']=pin(pt)
desc=O/'production-inputs.launch-v3.json';assert not desc.exists();atomic_json(desc,v)
print(json.dumps({'plan':pin(pt),'descriptor':pin(desc)}),flush=True)
print(json.dumps(build_lane(v,O/'lane-source-v3.json')),flush=True)
