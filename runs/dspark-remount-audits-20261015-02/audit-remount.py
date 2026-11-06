"""Run the existing original CPU auditor once for each remounted admission."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,stat,sys
sys.dont_write_bytecode=True
ROOT=Path('/home/philip/binary-eagle-decoding')
SOURCE=ROOT/'runs/dspark-direct-environment-20261015-source-10/source'
OUTPUT=ROOT/'runs/dspark-remount-audits-20261015-02'
BOOT='808f6149-6c87-47f4-890d-3a9aebe15dca'
sys.path.insert(0,str(SOURCE/'src'))
from w1a1_eagle.direct_training_runtime import metadata,checked,current_control,keeper_environment,write_new
assert os.getuid()==1000 and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
config=checked({'path':str(ROOT/'runs/dspark-direct-resume-20261015-01/config.json'),'sha256':'fcbbea86bc29ed2a88c76ed6d3913a91a4c1c7dee292d3f3d323a629f82e8f14'})
lane=checked(config['lane']);os.environ.update(keeper_environment(config['keeper']))
census_pin={'path':str(ROOT/'runs/dspark-direct-resume-20261015-01/remount-inspection.json'),'sha256':'7ea1fce6fe042f83c7d9404c47acb5ce04467afaf3dfa800765bba27ebb9802a'}
census=checked(census_pin)
assert census['boot_id']==BOOT and not census['nondevice_changes'] and not census['missing']
assert census['device_transitions']=={'(51, 61)':1088} and len(census['rows'])==16
native=Path(config['native_source_root']);auditor=native/'scripts/prepare_block_qat_data.py'
AUDITOR='a16a1d53d2b9f4098f8d3377e3f4829e17666bbf2eb1a33e831c34221cd93add'
assert hashlib.sha256(auditor.read_bytes()).hexdigest()==AUDITOR
assert hashlib.sha256((native/'src/w1a1_eagle/block_data.py').read_bytes()).hexdigest()=='d8531872a1641c7b35e822107a3186abf9e16542cabbe83e6c0dfc0bdfcd5544'
record=lane['source']['src/w1a1_eagle/nine_model_pipeline.py'];assert record['sha256']=='a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af'
assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
spec=importlib.util.spec_from_file_location('original_cpu_audit_supervision',record['path']);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
run=Path(config['run_dir']);budget_path=run/'training/budget-used.json';latest_path=run/'training/checkpoints/latest.json'
budget,budget_sha=metadata(budget_path);latest,latest_sha=metadata(latest_path)
assert budget['active_attempt'] is None and budget['training_seconds']==14400.819240198878 and latest['cursor']['step']==17135
def authorize():
 current_control(lane)
 assert not (run/'STOP').exists() and not (run/'training/STOP').exists()
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
 assert metadata(budget_path)[1]==budget_sha
assert OUTPUT.is_dir() and not (OUTPUT/'view.json').exists()
lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'
with lock.open('a') as stream:
 info=os.fstat(stream.fileno());assert stat.S_ISREG(info.st_mode) and info.st_uid==1000 and info.st_nlink==1 and not info.st_mode&0o022
 fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runner=api.SubprocessRunner(native,{**lane.get('environment',{}),'OMP_NUM_THREADS':'2'},authorization=authorize)
 entries=[]
 with api.stop_signals():
  for number,row in enumerate(census['rows']):
   authorize();assert hashlib.sha256(auditor.read_bytes()).hexdigest()==AUDITOR
   assert row['changed_fields']=={'device':68} and not row['missing']
   checked(row['manifest']);checked(row['admission'])
   directory=OUTPUT/'rows'/('%02d'%number);directory.mkdir(parents=True,exist_ok=False)
   admission=directory/'completed-admission.json';report=directory/'audit-report.json'
   argv=[config['python'],'-B',str(auditor),'--manifest',row['manifest']['path'],'--manifest-sha256',row['manifest']['sha256'],'--require-soft-teachers','--admission-output',str(admission),'--output',str(report)]
   print(json.dumps({'event':'original_cpu_audit_start','index':number,'origin':row['origin'],'payload_bytes':row['payload_bytes'],'argv':argv}),flush=True)
   runner.run(argv,directory=directory/'process',stop_path=run/'STOP',wall_seconds=18000)
   new_pin={'path':str(admission),'sha256':hashlib.sha256(admission.read_bytes()).hexdigest()}
   report_pin={'path':str(report),'sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
   fresh=checked(new_pin);measured=checked(report_pin)
   assert fresh['manifest_sha256']==row['manifest']['sha256'] and measured['completed_admission']==new_pin
   old=checked(row['admission']);assert len(fresh['artifacts'])==len(old['artifacts'])==68
   for a,b in zip(old['artifacts'],fresh['artifacts']):assert b==dict(a,device=61)
   changes=[{'path':item['path'],'old_device':item['old']['device'],'new_device':item['current']['device']} for item in row['differences']]
   entries.append({'manifest':row['manifest'],'old_admission':row['admission'],'new_admission':new_pin,'audit_report':report_pin,'device_changes':changes})
   api.atomic_json(OUTPUT/'progress.json',{'completed':number+1,'total':16,'last_origin':row['origin'],'entries':entries,'trainer_seconds_charged':0})
   print(json.dumps({'event':'original_cpu_audit_complete','index':number,'origin':row['origin'],'new_admission':new_pin,'report':report_pin}),flush=True)
  authorize();assert metadata(latest_path)[1]==latest_sha
  view={'schema':'original_completed_admission_current_boot_view_v1','reader_sha256':'d8531872a1641c7b35e822107a3186abf9e16542cabbe83e6c0dfc0bdfcd5544','boot_id':BOOT,'entries':entries,'input_census':census_pin,'original_full_auditor':{'path':str(auditor),'sha256':AUDITOR},'trainer_seconds_charged':0}
  write_new(OUTPUT/'view.json',view)
  print(json.dumps({'event':'all_original_cpu_audits_complete','view':{'path':str(OUTPUT/'view.json'),'sha256':hashlib.sha256((OUTPUT/'view.json').read_bytes()).hexdigest()},'count':16,'trainer_seconds_charged':0}),flush=True)
