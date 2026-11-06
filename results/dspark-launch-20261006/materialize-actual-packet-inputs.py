import json,hashlib,sys,shutil
from pathlib import Path
ROOT=Path('/home/philip/binary-eagle-decoding');OUT=ROOT/'runs/dspark-a8-full-pool-20261006-01';CHECKOUT=ROOT/'runs/checkouts/dspark-full-training-20261006-01';T=OUT/'operator-inputs/controller-handoff'
sys.path.insert(0,str(CHECKOUT/'src'))
from w1a1_eagle.block_shard_lifecycle import pin,checked,write_logical_admission
from w1a1_eagle.nine_model_pipeline import atomic_json
fit=json.loads((OUT/'fit-a8/fit_report.json').read_text());assert fit['config']['activation_bits']==8 and len(fit['fit_rows'])==96 and len(fit['validation_rows'])==48
# Human authorization is already given; this receipt records it, not new approval.
authority={'schema':'dspark_human_authorization_record_v1','human_instruction':'Cancel the automated task that sent that and start training dspark a8, with an hourly health check/healing monitor.','owner_thread':'01a10c8a-22bb-7380-9a8c-d9802a51b679','approved_proposal':{'path':str(CHECKOUT/'experiments/nine-model-qat-overnight/dspark-a8-next-run.proposed.json'),'sha256':pin(CHECKOUT/'experiments/nine-model-qat-overnight/dspark-a8-next-run.proposed.json')['sha256']},'trainer_seconds':43200,'operational_calibration_bytes_cap':64*1024**3,'calibration_array_bytes_cap':3*1024**3,'hourly_monitor':'dspark-a8-hourly-health-and-healing'}
atomic_json(OUT/'authorization.json',authority);auth={'kind':'human_delegated_operational_settings','instruction':authority['human_instruction'],'record':pin(OUT/'authorization.json')}
for name in ['budget','coverage-policy','calibration-policy']:
 value=json.loads((T/(name+'.PENDING.json')).read_text());value['authorization']=auth;value.pop('status',None)
 if name=='budget':value['candidates']['dspark_a8']['wall_seconds']=604800
 atomic_json(OUT/(name+'.json'),value)
runtime=json.loads((T/'runtime.PENDING.json').read_text());runtime.pop('status',None);build=ROOT/'runs/build/native-indexed-ecff-sm120-20261005-v2/bin'
runtime['inputs']['backend_binary']=pin(build/'test-backend-ops');runtime['inputs']['block_native_binary']=pin(build/'test-block-binary');runtime['inputs']['binary']=pin(build/'llama-server')
for p in sorted(build.glob('*.so*')):
 if p.is_file():runtime['inputs']['runtime_library_'+p.name.replace('.','_').replace('-','_')]=pin(p)
runtime.update(gpu_uuid='GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e',device_name='NVIDIA GeForce RTX 5080',gpu_control_path=str(ROOT/'runs/qat-optimization-readiness/prepared-model-gates-packet-20261003-01/authorized-gpu-control.json'))
runtime['resource_policy'].update(host_return_tolerance_bytes=4*1024**3,gpu_return_tolerance_bytes=256*1024**2)
assert json.loads(Path(runtime['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is False
atomic_json(OUT/'runtime.packet.json',runtime)
data=json.loads((OUT/'rotating-data-v2-replay.json').read_text())
prep=pin(OUT/'preparation-merged-v2/manifest.json');prep_adm=pin(OUT/'preparation-merged-v2/completed-admission.json')
logical=write_logical_admission(data,prep,prep_adm,OUT/'logical-data-admission.json')
protocol=OUT/'operator-inputs/native-protocol-packet/native-primary24.protocol.json';assert pin(protocol)['sha256']=='e698c9550c112e5c63e2482e0a5f81a3b2f5dc0a88569e55727de4ad80f9daf2'
inputs=json.loads((T/'packet-inputs.PENDING.json').read_text());inputs.pop('status',None)
for name,path in [('runtime','runtime.packet.json'),('budget','budget.json'),('coverage_policy','coverage-policy.json'),('calibration_policy','calibration-policy.json')]:inputs[name]=pin(OUT/path)
inputs.update(data=data,data_admission=pin(OUT/'logical-data-admission.json'),preparation_data=prep,preparation_data_admission=prep_adm,evaluation_protocol=pin(protocol),initializer={'fit_report':pin(OUT/'fit-a8/fit_report.json'),'artifact':pin(OUT/'fit-a8/fusion_candidate.npz')})
inputs['shard_lifecycle']['plan']={k:data[k] for k in ('path','sha256')};inputs['shard_lifecycle']['replay_admission']=data['replay_admission']
for role in ['capture','restore']:
 producer=CHECKOUT/'scripts/capture_nine_model_train_data.py';inputs['shard_lifecycle'][role]['producer']=pin(producer);inputs['shard_lifecycle'][role]['argv']=[sys.executable,str(producer),'--request','{request}','--output','{output}']
inputs['shard_lifecycle']['restore']['replay_producer']=pin(CHECKOUT/'scripts/replay_dspark_shard.py')
atomic_json(OUT/'packet-inputs.json',inputs)
print(json.dumps({'status':'ACTUAL_INPUTS_MATERIALIZED_NOT_ACTOR_ADMITTED','inputs':pin(OUT/'packet-inputs.json'),'logical_data_admission':logical,'fit':pin(OUT/'fit-a8/fit_report.json'),'optimizer_updates':0}),flush=True)
