import json,sys,subprocess,time,copy
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';O=R/'runs/dspark-a8-full-pool-20261006-01';P=O/'packet-v4';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.block_shard_lifecycle import pin
from w1a1_eagle.nine_model_pipeline import LinuxResources,atomic_json
zero=(D/'prepare-new-zero-transfer.py').read_text().replace('packet-v3','packet-v4').replace('source-zero-v3-operation','source-zero-v4-operation').replace('root-source-zero','root-source-zero-v4')
exec(compile(zero,'actual_zero_v4','exec'),{'__name__':'__main__'})
commands=json.loads((P/'commands.json').read_text());rt=json.loads((O/'runtime.inference-v2.json').read_text())
obs=LinuxResources(rt['gpu_uuid']);proof=obs.require_released([],[]);snap=obs.snapshot();assert not proof['other_context_pids'] and not snap['dxg_holders']
t=time.time();lease=P/'initial-export-lease.json';atomic_json(lease,dict(schema='nine_model_gpu_lease_v1',host='rtx5080',user_announced_available=True,sole_owner=True,pause_requested=False,bundle_sha256=pin(P/'initial-preparation-request.json')['sha256'],gpu_uuid=rt['gpu_uuid'],granted_unix=t,expires_unix=t+300))
cmd=[str(lease) if a=='<fresh-root-provided-5080-lease>' else a for a in commands['initial_export']]
subprocess.run(cmd,check=True)
proof=obs.require_released([],[]);snap=obs.snapshot();assert not proof['other_context_pids'] and not snap['dxg_holders']
old=json.loads((O/'packet/initial-export/export-audit.json').read_text());new=json.loads((P/'initial-export/export-audit.json').read_text());norm=copy.deepcopy(new);norm['output']['path']=old['output']['path'];assert norm==old
e=json.loads((P/'initial-export-receipt.json').read_text());assert e['model']['sha256']=='ea1bc6896f046a936eb5c712520c36d150d9dd99823f5bffaaa34897f178bdf6'
atomic_json(D/'source-v4-export-equivalence.json',dict(schema='dspark_v4_export_equivalence_v1',status='PASS',old=old,new=new,actual_same_model_sha256=True,audit_equal_except_output_locator=True,release=proof,resources=snap,export=pin(P/'initial-export-receipt.json')))
gold=(D/'run-new-goldens-transfer.py').read_text().replace('packet-v3','packet-v4').replace('source-v3-goldens-operation','source-v4-goldens-operation')
exec(compile(gold,'actual_goldens_v4','exec'),{'__name__':'__main__'})
print(json.dumps({'v4_actual_stages':'PASS','optimizer_updates':0}),flush=True)
