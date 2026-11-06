import json,sys,subprocess,tarfile
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';O=R/'runs/dspark-a8-full-pool-20261006-01';P=O/'packet-v4';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,atomic_json,process_identity
from w1a1_eagle.block_shard_lifecycle import pin
run=C/'runs/dspark-a8-v4-prelaunch-20261007-03';state=json.loads((run/'state.json').read_text());assert state['status']=='finished' and state['exit_code']==0
obs=LinuxResources('GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e');proof=obs.require_released([92407,735],[dict(pid=92402,start_ticks=26758996,boot_id='517c4a36-e475-4a5f-9fa6-65de57edc6fe'),dict(pid=92407,start_ticks=26759001,boot_id='517c4a36-e475-4a5f-9fa6-65de57edc6fe'),dict(pid=735,start_ticks=28178,boot_id='206feba6-9535-4483-892a-9807057589ee')]);snap=obs.snapshot();assert proof['other_context_pids']==[] and snap['dxg_holders']==[]
for p in [D/'source-zero-v4-operation.json',D/'source-v4-export-equivalence.json',D/'source-v4-goldens-operation.json']:
 assert json.loads(p.read_text())['status']=='PASS'
inventory=D/'source-cutover-v4-authenticated-inventory.json';assert not inventory.exists()
source={str(p.relative_to(C)):pin(p) for folder in [C/'scripts',C/'src/w1a1_eagle'] for p in sorted(folder.rglob('*.py'))}
assert len(source)==246
atomic_json(inventory,dict(schema='dspark_source_cutover_v4_authentic_inventory_v1',commit=subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip(),source=source,completed_job=pin(run/'state.json'),interrupted_original_job=pin(D/'v4-prelaunch-reboot-reconciliation.json'),release=proof,resources=snap))
files=[P/n for n in ['packet-inputs.json','initial-preparation-request.json','configs/dspark_a8.json','initial-prepare-receipt.json','initial-export-receipt.json','initial-export/export-audit.json','initial-export/binary.json','golden-source-joins.json','golden-replay-requests.jsonl','native-goldens-receipts.jsonl']]
files += [D/n for n in ['source-zero-v4-operation.json','source-v4-export-equivalence.json','source-v4-goldens-operation.json','source-cutover-v4-authenticated-inventory.json','replay-admission.json']]
files += [D/'v4-prelaunch-reboot-reconciliation.json',C/'runs/dspark-a8-v4-prelaunch-20261007-02/state.json']
files += [O/'runtime.inference-v2.json',O/'operator-inputs/native-protocol-packet/native-primary24.protocol.json',run/'state.json']
joins=json.loads((P/'golden-source-joins.json').read_text())
files += [Path(case['source_native_receipt']['path']) for case in joins['cases']]
out=D/'source-cutover-v4-recovery-evidence.tar';assert not out.exists()
with tarfile.open(out,'w') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(R)),recursive=False)
print(json.dumps({'archive':pin(out),'files':len(set(files)),'release':'PASS'}))
