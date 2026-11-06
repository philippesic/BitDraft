#!/usr/bin/env python3
"""Read-only fresh resource, process-birth and canonical-lock census."""
import fcntl, hashlib, importlib.util, json, shutil, time
from pathlib import Path

OUT=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-resume-20261010/dspark-evaluation-01')
PLAN=OUT/'plan.json'
PLAN_SHA='a6cbbd4e0614942e3f8c9eb3ab2a872a4efd966fb8082c749c5d9606af6a4568'
DESC=OUT/'inputs/dspark-evaluation-binding.inputs.resolved.json'
DEST=OUT/'fresh-census-03.json'
SOURCE=Path('/home/philip/binary-eagle-decoding/runs/5080-training-20261009/source-937af03/src/w1a1_eagle/nine_model_pipeline.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

assert not DEST.exists(), str(DEST)
assert sha(PLAN)==PLAN_SHA
plan=json.loads(PLAN.read_text())
assert plan['quality_policy']=='measurement_only' and plan['campaign_complete'] is False
spec=importlib.util.spec_from_file_location('dspark_fresh_census_pipeline',SOURCE)
pipeline=importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)
all_births=[]; groups=[]; process_files=json.loads(DESC.read_text())['process_files']
for loc in process_files:
    p=Path(loc['path']).resolve(strict=True)
    assert sha(p)==loc['sha256'], str(p)
    item=json.loads(p.read_text())
    if p.name=='process.json':
        birth=item['kernel_identity']; assert item['pid']==item['pgid']==birth['pid']
        groups.append(item['pgid']); births=[birth]
    else:
        assert p.name=='process-lineage.json'
        births=item['kernel_identities']
    for birth in births:
        assert set(birth)=={'pid','start_ticks','boot_id'}
        assert not pipeline.identity_active(birth), ('active original DSpark birth',birth)
        all_births.append(birth)
lane_state=json.loads(Path(plan['lane_state']['path']).read_text())
observer=pipeline.LinuxResources(plan['gpu_uuid'])
release=observer.require_released(plan['training_process_groups'],plan['training_process_identities'])
snapshot=observer.snapshot()
pipeline.resource_gate(snapshot,lane_state['resource_baseline'],plan['resource_policy'])
assert snapshot.get('dxg_holders')==[] and release.get('other_context_pids')==[]
pause_path=Path(plan['gpu_control_path']).resolve(strict=True)
pause_bytes=pause_path.read_bytes(); pause=json.loads(pause_bytes)
control=pause.get('rtx5080',{})
assert control.get('pause_requested') is False and control.get('stop_all_requested') in (None,False)
assert not (Path(plan['lane_state']['path']).parent/'STOP').exists()
locks=[]; lock_meta=[]
for p in (Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock',
          Path.home()/'.cache/binary-eagle-decoding/cuda-0.owner.lock'):
    before=p.read_bytes(); st=p.stat(); f=p.open('r+b')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks.append((p,f,before))
    lock_meta.append({'path':str(p),'sha256':hashlib.sha256(before).hexdigest(),'inode':st.st_ino,
                      'mode':st.st_mode&0o777,'size':st.st_size,'nonblocking_flock':'ACQUIRED'})
for p,f,before in reversed(locks):
    fcntl.flock(f,fcntl.LOCK_UN); f.close(); assert p.read_bytes()==before
lane=json.loads(Path(plan['frozen_lane']['path']).read_text())
config=json.loads(Path(lane['config']['path']).read_text())
checkpoint=Path(plan['endpoint']['checkpoint']['path']); base=Path(config['model']['path'])
floor=2*(checkpoint.stat().st_size+base.stat().st_size)
host_available=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024
output_free=shutil.disk_usage(OUT).free
assert host_available>=floor and output_free>=floor,(host_available,output_free,floor)
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
report={'schema':'dspark_fresh_readonly_census_v1','status':'PASS_FRESH_CENSUS','observed_unix':time.time(),
 'plan':{'path':str(PLAN),'sha256':sha(PLAN)},'descriptor':{'path':str(DESC),'sha256':sha(DESC)},
 'boot_id':boot,'gpu_uuid':plan['gpu_uuid'],'resources':snapshot,'release':release,
 'process_file_count':len(process_files),'original_process_group_count':len(groups),
 'original_birth_count':len(all_births),'unique_birth_count':len({(x['pid'],x['start_ticks'],x['boot_id']) for x in all_births}),
 'active_original_births':0,'pause_control':{'path':str(pause_path),'sha256':hashlib.sha256(pause_bytes).hexdigest(),
 'pause_requested':control['pause_requested'],'stop_all_requested':control.get('stop_all_requested')},
 'canonical_locks':lock_meta,'cpu_disk_floor':{'checkpoint_path':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
 'base_model_path':str(base),'base_model_bytes':base.stat().st_size,'required_bytes':floor,
 'host_available_bytes':host_available,'output_free_bytes':output_free},'gpu_queried':True,'training_changed':False}
DEST.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':report['status'],'receipt':{'path':str(DEST),'sha256':sha(DEST)},
 'boot_id':boot,'gpu_uuid':plan['gpu_uuid'],'process_files':len(process_files),'births':len(all_births),
 'groups':len(groups),'resources':snapshot,'release':release,'locks':lock_meta,'cpu_disk_floor':report['cpu_disk_floor']},sort_keys=True))
