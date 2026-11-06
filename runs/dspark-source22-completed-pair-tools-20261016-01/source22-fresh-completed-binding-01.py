"""One exact SOURCE20 completed-pair binding and guarded stop/check; no loop."""
from pathlib import Path
import argparse,hashlib,json,sys,time,datetime
p=argparse.ArgumentParser();p.add_argument('--number',required=True);p.add_argument('--arm',action='store_true');a=p.parse_args();assert a.number.isdecimal() and len(a.number)<=3
base=Path('/home/philip/binary-eagle-decoding');run=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-20261015-04/lane');op=run/'direct-runtime/direct-dspark-30e1728b885a4caa9202ad9b783b271d'
guard=base/'runs/dspark-source22-completed-pair-tools-20261016-01/stop_completed.py';raw=guard.read_bytes();assert hashlib.sha256(raw).hexdigest()=='8619e72b6d2afd1e81e38e9ea806f97e70db7a7805ba930e57ed8f0c95ed48ca';ns={'__name__':'source22_completed_guard'};exec(compile(raw,'frozen-completed-stop-8619e72b','exec'),ns)
sys.dont_write_bytecode=True;sys.path.insert(0,str(op/'source/src'));from w1a1_eagle import direct_training_runtime as rt
runtime,cfg=rt.authenticate_runtime(op/'runtime.json');assert rt.metadata(op/'runtime.json')[1]==ns['RUNTIME_SHA'];lane=rt.checked(cfg['lane']);rt.current_control(lane);rt.keeper_environment(cfg['keeper'])
assert {k:ns['identity'](v['pid']) for k,v in ns['EXPECTED'].items()}==ns['EXPECTED'];assert rt.metadata(op/'owned-identity.json')[0]==ns['EXPECTED']
def refuse(reason):print(json.dumps({'status':'REFUSED','reason':reason,'signal_sent':False,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}));raise SystemExit(0)
if Path('/proc/9292/task/9292/children').read_text().strip():refuse('live controller child')
if rt.metadata(run/'training/budget-used.json')[0]['active_attempt'] is not None:refuse('active TRAIN clock')
values={};records={};fingerprints={};auxiliary={}
def read(name,path):
 path=Path(path);before=ns['fingerprint'](path);v,h=rt.metadata(path);assert before==ns['fingerprint'](path),'binding metadata changed'
 values[name]=v;records[name]={'path':str(path),'sha256':h};fingerprints[str(path)]=before;return v
def aux(row):
 if row['path'] in auxiliary:assert auxiliary[row['path']]==row,'conflicting auxiliary pin'
 auxiliary[row['path']]=row
state=read('state',run/'state.json');latest=read('latest',run/'training/checkpoints/latest.json');active=read('active',op/'active-window.json')
if state['status']!='restoring_shards' or state.get('pending_shard_resume') is not True:refuse('not original shard boundary')
assert active['catalog']['sha256']=='e5673234f28ff2f4046692c097e7905e30ecf6a3408aed6e8359ffeb9fc133e8','different old catalog; do not use this run-specific operation'
catalog=read('catalog',active['catalog']['path']);assert records['catalog']==active['catalog']
oldroots={str(Path(row['path']).parent) for row in catalog['source_cache_states']};pending={root:reg for root,reg in state['prepared_pair_registrations'].items() if root not in oldroots};assert pending,'no pending pairs'
pending_states={};pair_values={}
for i,(root,reg) in enumerate(pending.items()):
 name='pending_'+str(i);v=read(name,Path(root)/'cache-state.json');pending_states[root]=records.pop(name);pair_values[root]=values.pop(name);aux(pending_states[root])
 if v['owner'] is not None or set(v['resident'])!=set(reg['requested_shards']):refuse('pending native pair active or partial')
 aux(reg['initial_state'])
spec=rt.checked(lane['config']);read('canonical',Path(spec['data']['cache_root'])/'cache-state.json')
if values['canonical']['owner'] is not None:refuse('canonical owner active')
for name,path in {'budget':run/'training/budget-used.json','status':run/'training/status.json','progress':run/'resource-handoff-progress.json','receipt':Path(state['train_receipt']['path'])}.items():read(name,path)
assert records['receipt']==state['train_receipt']
ret=next(row for row in reversed(state['prepared_window_retirements']) if row['original_catalog']==active['catalog'] and row['cursor']==latest['cursor']['data_cursor']);rd=Path(ret['source_state_archives'][0]['snapshot']['path']).parent
for name,file in [('retirement_receipt','receipt.json'),('retirement_context','context.json'),('archive_index','source-state-archives.json')]:read(name,rd/file)
for row in ret['source_state_archives']:aux(row['snapshot'])
for row in ret['owned_cache_states']:aux(row)
pending_controls={root:{} for root in pending};requests={}
for path in (run/'attempts'/op.name/'shard-phases').glob('*/request.json'):
 req,h=rt.metadata(path);key=(str(Path(req['directory']).parent),req['shard'])
 if key[0] in pending:
  assert key not in requests,'duplicate native request for pending pair'
  requests[key]=path
for root,reg in pending.items():
 cache=pair_values[root]
 for shard in reg['requested_shards']:
  aux(cache['seals'][shard]['original_metadata'])
  for role in ['manifest','admission']:aux(cache['resident'][shard][role])
  phase=requests[(root,shard)].parent;controls={}
  for role,rel in [('publication','publication.json'),('request','request.json'),('process','producer/process.json'),('exit','producer/process-exit.json')]:
   name='native_'+role;read(name,phase/rel);controls[role]=records.pop(name);values.pop(name);aux(controls[role])
  pending_controls[root][shard]=controls
assert len(auxiliary)<=256 and sum(ns['fingerprint'](row['path'])[6] for row in auxiliary.values())<=256*1024**2
assert all(ns['fingerprint'](path)==before for path,before in fingerprints.items()),'binding raced'
observed=time.time();binding={'schema':'source22_completed_pair_endpoint_binding_v1','observed_unix':observed,'expires_unix':observed+60,'minimum_release_unix':fingerprints[records['receipt']['path']][-2]/1e9,'scientific_config_sha256':lane['config']['sha256'],'gpu_uuid':lane['gpu_uuid'],'files':records,'metadata':list(auxiliary.values()),'pending_states':pending_states,'pending_controls':pending_controls}
path=base/('runs/dspark-source22-completed-pair-binding-20261016-'+a.number+'.json');rt.write_new(path,binding);digest=hashlib.sha256(path.read_bytes()).hexdigest();args=['--binding',str(path),'--binding-sha256',digest]
if a.arm:args+=['--output',str(base/('runs/dspark-source22-completed-pair-stop-20261016-'+a.number))]
else:args+=['--check']
print(json.dumps({'binding_path':str(path),'binding_sha256':digest,'requested_arm':a.arm,'auxiliary_files':len(auxiliary),'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True);ns['main'](args)
