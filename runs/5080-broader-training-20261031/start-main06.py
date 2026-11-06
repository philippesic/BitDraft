from pathlib import Path
import json,os,sys,time,subprocess,shutil,hashlib,shlex
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';S=R/'source02';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');OLD=P/'runs/unattended-training-preparation-20261028/gpu-resume-20261028';C=R/'controller01'
sys.path[:0]=[str(S/'scripts'),str(P/'scripts')];os.environ['PYTHONPATH']=str(S/'scripts')+':'+str(P/'scripts')
from training_system.io import read,pin,authenticate,atomic_json
from training_system.device_proofs import _proofs
for name in ['memory01']:
 while True:
  try:s=read(W/'supervisors'/name/'state.json')
  except (OSError,ValueError):time.sleep(3);continue
  if s['status']!='running':break
  time.sleep(3)
 assert s['exit_code']==0,(name,s)
assert read(R/'verification04-result.json')['correctness_pass'] is True
assert read(W/'parent-native01/result.json')['metrics']['development']['windows']==192
for name in ['controller.py','campaign_comparison.py']:
 assert pin(C/name)['sha256']==pin(OLD/'campaign-controller178'/name)['sha256']
base=read(W/'base-spec04.json');base['deadline_unix']=1793507400;q=pin(W/'qualification-complete05.json');qc=read(W/'qualification-config04.json');milestones=[4800,14400,32400];candidates={}
for objective in ['control','dpal']:
 record,hardware,proofs=_proofs(q,objective);contract=read(authenticate(proofs['runtime_contract']));spec={**base,'objective':objective,'target_updates':milestones[0],'max_invocation_train_seconds':840,'checkpoint_root':str(W/('training-checkpoints-'+objective)),'job_tag':'broader-'+objective,'retention_policy':'latest_verified_and_previous','retention_references_complete':True,'retention_references':[]}
 candidates[objective]={'spec':spec,'contract':contract,'qualification':{'result':q,'objective':objective},'native':qc['native'],'milestones':milestones,'exposure_design':{'block_presentations':[2*x for x in milestones],'input_rows':[14*x for x in milestones],'supervised_rows':[10*x for x in milestones],'unique_windows':960,'unique_groups':240,'complete_pair_pass_updates':480,'inherited_updates':3504,'whole_parent_unseen_proven':False,'measured_GPU_TRAIN_seconds':'measured separately; milestones are equal optimizer updates, not promised hours'}}
GiB=1024**3
cfg={'schema':'training_stage_controller_v1','mode':'qualified_training','controller_packet_manifest':pin(C/'manifest.json'),'lane_package_manifest':pin(S/'manifest.json'),'native_package_manifest':pin(S/'manifest.json'),'guard_template':pin(W/'guard04.json'),'deadline_unix':1793507400,'max_transient_retries':2,'retry_backoff_seconds':30,'gpu_lock':str(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'),'release_margin_seconds':300,'evaluation_wall_reserve_seconds':600,'invocation_wall_reserve_seconds':1200,'candidates':candidates,'cohorts':[{'id':'broader-serving-control-dpal','candidates':['control','dpal'],'control':'control','protected':'dpal','slots':[2,2,2],'exposure_basis':'block_presentations','comparison_design':{'schema':'comparison_design_v1','justification':'Same parent3504,960serving-fit windows/240groups,192development windows/48groups, frozen balanced evaluation panel, exact pair order, original precision/optimizer/schedule/guards. Original EAL versus D-PAL and its pinned teacher are the declared treatment. Inherited TRAIN ancestry; no unseen or throughput release. Equal optimizer/block exposure at each stage; three early/intermediate/long curves before deciding continuation or restart.','treatment_axes':['/comparison/objective','/comparison/teacher'],'control_fields':['/comparison/data','/comparison/initializer','/comparison/model_config','/comparison/precision','/comparison/schedule','/comparison/gradient','/comparison/calibration']}}], 'resource_reservations':{phase:{'WSL':{'path':str(R),'floor_bytes':2*GiB,'reservation_bytes':GiB//4},'nativeC':{'path':str(W),'floor_bytes':10*GiB,'reservation_bytes':(16 if phase=='training' else 6)*GiB}} for phase in ['training','evaluation']}}
atomic_json(W/'main-controller-config01.json',cfg)
# The controller validates this exact new source/qualification pairing before admission.
k=read(R/'interop-keeper02.json');assert Path(k['interop']).exists();os.environ['WSL_INTEROP']=k['interop'];sys.path.insert(0,str(C))
import importlib.util
ms=importlib.util.spec_from_file_location('owned_controller',C/'controller.py');mod=importlib.util.module_from_spec(ms);ms.loader.exec_module(mod);mod.validate(cfg)
print('MAIN_CONTROLLER_CONFIG_VALIDATED',flush=True)
subprocess.run([sys.executable,'-B',str(C/'controller.py'),'--config',str(W/'main-controller-config01.json'),'--output',str(W/'main-controller01')],check=True,env=os.environ.copy())
print('MAIN_CONTROLLER_FINITE_RETURN',flush=True)
