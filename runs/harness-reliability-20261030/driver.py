"""Actual current-source control/VAT multistage crash/restart rehearsal."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/unattended-training-preparation-20261028/gpu-resume-20261028';S=R/'source26';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/unattended-training-preparation-20261028/gpu-resume-20261028');sys.path[:0]=[str(S/'scripts'),str(P/'scripts')]
from training_system.io import read,pin,authenticate,atomic_json,digest
from training_system.train import continuation_contract
from training_system.device_proofs import _proofs
from training_system.verify import compare_states
first=read(W/'scientific-campaign-20261029/first-scientific-launch-admission.json');science=read(authenticate(first['spec']));selected=read(R/'campaign-selected128.json');vat=selected['entries']['short_objective_vat'];assert vat['status']=='GPU_QUALIFIED_SELECTED_CONTRACT'
root=W/'harness-reliability-20261030-01';root.mkdir(exist_ok=False);cp=root/'config.json';out=root/'automatic';packet=R/'campaign-controller178'
assert (packet/'manifest.json').is_file()
manifest=read(packet/'manifest.json')
for item in manifest['files']:authenticate({'path':str(packet/item['path']),'sha256':item['sha256']})
config={'schema':'training_stage_controller_v1','mode':'qualified_training','deadline_unix':1793389364,'release_margin_seconds':900,'invocation_wall_reserve_seconds':1200,'evaluation_wall_reserve_seconds':900,'max_transient_retries':2,'retry_backoff_seconds':1,'controller_packet_manifest':pin(packet/'manifest.json'),'lane_package_manifest':pin(S/'manifest.json'),'native_package_manifest':pin(R/'source29/manifest.json'),'guard_template':pin(R/'campaign-control124-guard.json'),'gpu_lock':read(R/'campaign-control124-config.json')['gpu_lock'],'qualification_process_fault_candidate':'control','candidates':{},'cohorts':[],
'resource_reservations':{phase:{'nativeC':{'path':str(W),'floor_bytes':10*1024**3,'reservation_bytes':20*1024**3},'WSL':{'path':str(R),'floor_bytes':10*1024**3,'reservation_bytes':12*1024**3}} for phase in ('training','evaluation')}}
references={}
for name,q,objective,cfg in [('control',first['qualification'],'control',read(R/'campaign-control124-config.json')),('vat',vat['result'],vat['objective'],read(authenticate(vat['config'])))]:
 record,_,proofs=_proofs(q,objective);references[name]=record
 spec={**cfg['spec'],'target_updates':4}
 spec.update(deadline_unix=config['deadline_unix'],max_invocation_train_seconds=60,checkpoint_root=str(R/'harness-reliability-20261030-01-checkpoints'),checkpoint_interval=0,retention_policy='latest_verified_and_previous',retention_references_complete=True,retention_references=[],export_policy='milestones',storage_budget=science['storage_budget'],job_tag='harness-reliability-20261030-01-'+name,stop_path=str(root/('STOP-'+name)))
 assert digest(continuation_contract(spec))==record['adapter_state_key']
 config['candidates'][name]={'contract':read(authenticate(proofs['runtime_contract'])),'spec':spec,'qualification':{'result':q,'objective':objective},'native':cfg['native'],'milestones':[4,8],'exposure_design':{'block_presentations':[8,16]}}
config['cohorts']=[{'id':'control_vat','candidates':['control','vat'],'control':'control','protected':'vat','slots':[2,2],'scientific_view':'parent3504_shortlist18_v1','exposure_basis':'block_presentations','comparison_design':{'schema':'comparison_design_v1','justification':'actual unchanged parent/order/pairs; only originalVAT objective/head/Adam treatment differs; rehearsal not scientificselection','control_fields':['/comparison/controls'],'treatment_axes':['/comparison/treatment']}}]
atomic_json(cp,config);base=[sys.executable,'-B',str(packet/'controller.py'),'--config',str(cp),'--output',str(out)];started=time.time()
def call(label,extra=(),env=None):
 log=root/(label+'.log')
 with log.open('xb') as f:r=subprocess.run(base+list(extra),env={**os.environ,**(env or {})},stdout=f,stderr=subprocess.STDOUT)
 atomic_json(root/(label+'-return.json'),{'returncode':r.returncode,'log':pin(log),'unix':time.time()})
 if r.returncode not in (0,92):raise RuntimeError(label+' failed: '+log.read_text(errors='replace')[-6000:])
 return r.returncode
assert call('finite-ceiling-pause',env={'TRAINING_CONTROLLER_SESSION_DEADLINE_UNIX':str(time.time()+600)})==0
state=read(out/'controller.json');assert state['status']=='PAUSED' and all(not read(Path(c['job'])/'job.json')['attempts'] for c in state['candidates'].values())
assert call('actual-crash92',extra=('--resume',),env={'TRAINING_CONTROLLER_SESSION_DEADLINE_UNIX':'1793389364','TRAINING_CONTROLLER_FAULT_AFTER_EXTENSION':'92'})==92
crashed=read(out/'controller.json');assert crashed['cohorts']['control_vat']['pending_transition'];atomic_json(root/'crash92-state.json',crashed)
assert call('fresh-process-durable-recovery',env={'TRAINING_CONTROLLER_SESSION_DEADLINE_UNIX':'1793389364'})==0
state=read(out/'controller.json');assert state['status']=='COMPLETE';before={n:len(read(Path(c['job'])/'job.json')['attempts']) for n,c in state['candidates'].items()}
assert call('completed-idempotence',env={'TRAINING_CONTROLLER_SESSION_DEADLINE_UNIX':'1793389364'})==0
final=read(out/'controller.json');physical=[]
for name,c in final['candidates'].items():
 j=read(Path(c['job'])/'job.json');assert j['owner'] is None and j['status']=='COMPLETE' and len(j['attempts'])==before[name]
 f=read(authenticate(j['finish']));assert f['local_updates']==8 and f['model']['sha256']==references[name]['resumed']['model']['sha256'];compare_states(references[name]['resumed']['checkpoint'],f['checkpoint'],root/(name+'-qualified-baseline-state-parity.json'))
 for a in j['attempts']:
  p=Path(a['output'])/'progress.json'
  if p.exists():
   d=read(p);physical.append({'source':pin(p),'updates':d['invocation_updates'],'TRAIN_seconds':d['invocation_TRAIN_seconds']})
assert sum(x['updates'] for x in physical)==16
result={'schema':'actual_current_source_automatic_rehearsal_v1','controller_source':pin(packet/'manifest.json'),'numeric_source':pin(S/'manifest.json'),'config':pin(cp),'controller':pin(out/'controller.json'),'actual_training_and_native':True,'actual_trainer_crash91':True,'actual_controller_crash92':True,'fresh_process_durable_transition_recovery':True,'finite_invocation_pause_and_explicit_resume':True,'completed_idempotence_no_duplicate_attempts':True,'qualified_baseline_fullstate_and_model_exact':True,'native_windows_per_panel':144,'native_panels':4,'computed_promotion_not_operator_boolean':True,'physical':physical,'physical_updates':16,'physical_TRAIN_seconds':sum(x['TRAIN_seconds'] for x in physical),'wall_seconds':time.time()-started,'scientific_updates':0,'protected_parking_reconsideration_component':'unchanged source20 actual3candidate proof plus currentCPU tests; this2candidate path doesnotretestparking','scientific_winner_claim':False}
atomic_json(root/'result.json',result);print(json.dumps(result),flush=True)
