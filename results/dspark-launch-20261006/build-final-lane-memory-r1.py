import json,sys
from pathlib import Path
from types import SimpleNamespace
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-memory-repair-968ab3d1-20261007';O=R/'runs/dspark-a8-full-pool-20261006-01';P=O/'packet-memory-r1';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.block_shard_lifecycle import pin
from w1a1_eagle.nine_model_pipeline import atomic_json
from prepare_block_lane_packet import bind
from prepare_nine_model_bundle import materialize_admission_plan
from prepare_nine_model_lane import build_lane
qa=D/'portable-qa-memory-repair1/qa-ledger.json';receipts=O/'packet-v4/native-goldens-receipts.jsonl'
print(json.dumps(bind(SimpleNamespace(packet=P,receipts=receipts,receipts_sha256=pin(receipts)['sha256'],qa_ledger=qa,qa_ledger_sha256=pin(qa)['sha256']))),flush=True)
v=json.loads((P/'production-inputs.json').read_text());old=json.loads((O/'production-inputs.canonical-alias-v3.json').read_text())
v['candidates']['dspark_a8']['initial_model']=old['candidates']['dspark_a8']['initial_model']
v['candidates']['dspark_a8']['initial_export_audit']=old['candidates']['dspark_a8']['initial_export_audit']
proof=D/'source-memory-r1-export-equivalence.json';assert json.loads(proof.read_text())['actual_same_model_sha256'] is True
v['preflight']['preparation_provenance']['memory_r1_export_equivalence']=pin(proof)
v['preflight']['preparation_provenance']['accounting_carry']=pin(D/'memory-r1-accounting-carry.json')
v['preflight']['preparation_provenance']['golden_reuse']=pin(D/'source-memory-r1-golden-reuse.json')
spec=json.loads(Path(v['candidates']['dspark_a8']['config']['path']).read_text())
for k,r in [('legacy_shard_capture_producer',spec['shard_lifecycle']['capture']['producer']),('legacy_shard_replay_producer',spec['shard_lifecycle']['restore']['replay_producer'])]:
 assert pin(Path(r['path']))==r
 v['preflight']['preparation_provenance'][k]=r
desc=O/'production-inputs.canonical-memory-r1.json';assert not desc.exists();atomic_json(desc,v)
admission_path=O/'admission-plan.canonical-memory-r1.json'
print(json.dumps(materialize_admission_plan(v,admission_path)),flush=True)
admission=json.loads(admission_path.read_text());plan=json.loads((O/'timed-evaluation-plan.reviewed-v3.json').read_text());selected=admission['candidates']['dspark_a8']
assert plan['training_source']==selected['source_bindings']
assert plan['initial']=={'model':selected['model'],'audit':selected['export']}
plan['config']=selected['config'];extra={k:r for k,r in plan['source'].items() if k.startswith('artifact:research:')}
plan['source']=dict(admission['source'],**extra)
plan['inference_runtime_provenance']['admission_plan']=pin(admission_path)
pt=O/'timed-evaluation-plan.reviewed-memory-r1.json';assert not pt.exists();atomic_json(pt,plan)
v['inputs']['admission_plan']=pin(admission_path);v['timed_evaluation_plan']=pin(pt)
desc=O/'production-inputs.launch-memory-r1.json';assert not desc.exists();atomic_json(desc,v)
print(json.dumps(build_lane(v,O/'lane-source-memory-r1.json')),flush=True)
