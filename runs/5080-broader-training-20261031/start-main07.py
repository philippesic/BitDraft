from pathlib import Path
import json,os,sys,time,subprocess,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';S=R/'source02';C=R/'controller02';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');old=W/'main-controller01';new=W/'main-controller02'
sys.path[:0]=[str(S/'scripts'),str(C)];os.environ['PYTHONPATH']=str(S/'scripts')+':'+str(P/'scripts');os.environ['WSL_INTEROP']=json.loads((R/'interop-keeper02.json').read_text())['interop']
from training_system.io import read,pin,authenticate,atomic_json
from training_system.lane import admit,alive,identity,locked
from training_system.train import continuation_contract
from training_system.io import digest
while read(W/'supervisors/main06/state.json')['status']=='running':
 assert time.time()<1793482800,'finite old-controller closure wait exceeded'
 time.sleep(3)
assert read(old/'controller.json')['status']=='PAUSED'
with locked(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'):
 assert subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()==''
for n in ['controller.py','campaign_comparison.py']:assert pin(C/n)['sha256']==pin(R/'controller01'/n)['sha256']
cfg=read(W/'main-controller-config01.json');cfg['controller_packet_manifest']=pin(C/'manifest.json');atomic_json(W/'main-controller-config02.json',cfg)
import importlib.util
m=importlib.util.spec_from_file_location('fixed_operational_controller',C/'controller.py');mod=importlib.util.module_from_spec(m);m.loader.exec_module(mod);mod.validate(cfg)
print('FIXED_OPERATIONAL_CONTROLLER_VALIDATED',flush=True)
new.mkdir(exist_ok=False);oldstate=read(old/'controller.json');snapshot=R/'main06-paused-snapshot.json';atomic_json(snapshot,oldstate)
state={'schema':'training_controller_state_v1','config':pin(W/'main-controller-config02.json'),'status':'PREPARED','owner':None,'candidates':{},'cohorts':{},'events':[]}
migration={'schema':'owned_operational_controller_repair_v1','reason':'Guard exists/stat race on DrvFS progress publication; exact qualified numerical package remains unchanged.','old_controller':pin(snapshot),'old_config':pin(W/'main-controller-config01.json'),'new_config':pin(W/'main-controller-config02.json'),'source02':pin(S/'manifest.json'),'jobs':{}}
for name,candidate in cfg['candidates'].items():
 prior=Path(oldstate['candidates'][name]['job']);job=read(prior/'job.json');assert not alive(job.get('owner'))
 if job['status']=='PAUSED' and job.get('checkpoint'):
  authenticate(job['checkpoint']);authenticate(job['finish']);f=read(job['finish']['path']);assert f['status']=='PAUSED'
  assert digest(continuation_contract(job['spec']))==digest(continuation_contract(candidate['spec']))
  jobroot=prior;record={'action':'continue_exact_paid_checkpoint','job':str(prior),'checkpoint':job['checkpoint'],'finish':job['finish'],'updates':f['local_updates']}
  (prior/'STOP').unlink(missing_ok=True)
 else:
  assert job['status']=='FAILED' and job.get('checkpoint') is None
  jobroot,_=admit(candidate['spec'],new/'jobs');record={'action':'fresh_original_parent_after_uncheckpointed_guard_failure','job':str(jobroot),'prior_failed_job':pin(prior/'job.json')}
 migration['jobs'][name]=record;state['candidates'][name]={'job':str(jobroot),'status':'ACTIVE','native':{},'transient_retries':0,'not_before_unix':0}
for cohort in cfg['cohorts']:state['cohorts'][cohort['id']]={'stage':0,'complete':False,'decisions':{}}
atomic_json(R/'main07-operational-repair.json',migration);state['events'].append({'action':'owned_operational_repair','receipt':pin(R/'main07-operational-repair.json'),'unix':time.time()});atomic_json(new/'controller.json',state)
subprocess.run([sys.executable,'-B',str(C/'controller.py'),'--config',str(W/'main-controller-config02.json'),'--output',str(new)],check=True,env=os.environ.copy())
print('FIXED_CONTROLLER_FINITE_RETURN',flush=True)
