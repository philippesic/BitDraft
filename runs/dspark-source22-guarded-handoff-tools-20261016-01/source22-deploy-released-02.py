from pathlib import Path
import sys,os,json,hashlib,datetime,fcntl,importlib.util,argparse
parser=argparse.ArgumentParser();parser.add_argument("--binding",type=Path,required=True);parser.add_argument("--binding-sha256",required=True);parser.add_argument("--stop-output",type=Path,required=True);args=parser.parse_args()
binding_raw=args.binding.read_bytes();assert hashlib.sha256(binding_raw).hexdigest()==args.binding_sha256;binding=json.loads(binding_raw);assert binding["schema"]=="source22_stable_endpoint_binding_v1"
assert args.stop_output.is_absolute() and args.stop_output.parent==Path("/home/philip/binary-eagle-decoding/runs") and args.stop_output.name.startswith("dspark-source22-fast-stable-stop-")
base=Path('/home/philip/binary-eagle-decoding');prepared=base/'runs/dspark-direct-environment-20261016-source-22';manifest_raw=(prepared/'manifest.json').read_bytes();assert hashlib.sha256(manifest_raw).hexdigest()=='88d493c0e9447174b7bd5168fc0b9baa51ff27e2e8d3a79053d99920c1a24e11';new=json.loads(manifest_raw)
old_raw=(base/'runs/dspark-direct-environment-20261016-source-20/manifest.json').read_bytes();assert hashlib.sha256(old_raw).hexdigest()=='104ee19138298370fedf2bfa972eb44146e042c95e858f77fe0ea3c2957249d6';old=json.loads(old_raw)
assert set(new['source_files'])==set(old['source_files'])
roles=[n for n in new['source_files'] if new['source_files'][n]!=old['source_files'][n]]
assert set(roles)=={'src/w1a1_eagle/direct_training_runtime.py','src/w1a1_eagle/sustained_block_handoff.py','src/w1a1_eagle/direct_environment_inputs.py','src/w1a1_eagle/direct_ensure_environment.py'}
for n,h in new['source_files'].items():assert hashlib.sha256((prepared/'source'/n).read_bytes()).hexdigest()==h
sys.path.insert(0,str(prepared/'source/src'));sys.dont_write_bytecode=True
from w1a1_eagle import direct_training_runtime as rt
cfg=rt.metadata(prepared/'source/configs/training.json')[0];assert 'continuation' not in cfg;lane=rt.checked(cfg['lane']);run=Path(cfg['run_dir']);op=run/'direct-runtime/direct-dspark-30e1728b885a4caa9202ad9b783b271d'
with (Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a') as owner:
 fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='808f6149-6c87-47f4-890d-3a9aebe15dca'
 assert all(not (Path('/proc')/str(pid)).exists() for pid in (9291,9292))
 assert not (run/'STOP').exists() and not (run/'training/STOP').exists(),'Preserve humanSTOP'
 rt.current_control(lane);os.environ.update(rt.keeper_environment(cfg['keeper']))
 runtime,rsha=rt.metadata(op/'runtime.json');assert rsha=='866c72fba3aa89e24f33332e20f7202e0be0001e8ef26472eb8fe89807d7fdfe'
 sup=rt.live_metadata(runtime['supervisor_state'])[0];assert sup['status'] in {'finished','interrupted','failed'} and sup['received_signal']['number']==15 and sup['stop_grace_seconds']==90
 intent=rt.metadata(args.stop_output/'intent.json')[0];assert intent['binding_sha256']==args.binding_sha256 and intent['owned']==rt.metadata(op/'owned-identity.json')[0]
 signal_receipt=rt.metadata(args.stop_output/'signal-issued.json')[0];assert signal_receipt['via']=='pidfd' and signal_receipt['pid']==9291 and signal_receipt['signal']=='SIGTERM'
 state,state_sha=rt.metadata(run/'state.json');ledger,budget_sha=rt.metadata(run/'training/budget-used.json');latest,latest_sha=rt.metadata(run/'training/checkpoints/latest.json')
 assert state['status']=='stopped' and state['failure']['type']=='InterruptedError' and not state.get('cleanup_failure')
 assert binding['files']['budget']=={'path':str(run/'training/budget-used.json'),'sha256':budget_sha} and ledger['active_attempt'] is None and ledger['max_seconds']==43200 and ledger['source_sha256']==state['bundle_sha256']
 assert binding['files']['latest']=={'path':str(run/'training/checkpoints/latest.json'),'sha256':latest_sha} and latest['committed'] and latest['cursor']['elapsed_seconds']==ledger['training_seconds']
 assert state['train_receipt']==binding['files']['receipt'];endpoint=rt.checked(binding['files']['receipt']);assert endpoint['status']=='PASS' and endpoint['completion_reason']=='shard_required' and endpoint['counters']==latest['cursor'] and endpoint['checkpoint']=={k:latest[k] for k in ('path','sha256')}
 assert binding['scientific_config_sha256']==lane['config']['sha256'] and binding['gpu_uuid']==lane['gpu_uuid'] and endpoint['config_sha256']==lane['config']['sha256'] and endpoint['bundle_sha256']==state['bundle_sha256']
 active=rt.checked(binding['files']['active']);assert active['catalog']==binding['files']['catalog'];catalog=rt.checked(binding['files']['catalog'])
 assert catalog['producer_source']=={'path':str(op/'source/src/w1a1_eagle/sustained_block_handoff.py'),'sha256':'1e62c9610f3c02a4a1fcba44696d50fc866a08ea9ade1c8ece202d0cbb7afefb'}
 source_roots=set()
 for row in catalog['source_cache_states']:
  value=rt.checked(row);assert value['owner'] is None and value['plan_sha256']==catalog['plan_sha256'];source_roots.add(str(Path(row['path']).parent))
 assert set(state.get('prepared_pair_registrations',{}))<=source_roots
 spec=rt.checked(lane['config']);assert rt.metadata(Path(spec['data']['cache_root'])/'cache-state.json')[0]['owner'] is None
 for n,h in old['source_files'].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,'rootSOURCE20differs:'+n
 check,check_sha=rt.metadata(prepared/'source-check.json');assert check_sha=='64eac15bf39fd509c8f91b331bcb6c98f478e81bf6705ccf863d420c706bfb2b' and check['exit_code']==0 and check['root_deployed'] is False
 original=lane['source']['src/w1a1_eagle/nine_model_pipeline.py'];assert original['sha256']=='a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af';assert hashlib.sha256(Path(original['path']).read_bytes()).hexdigest()==original['sha256']
 module_spec=importlib.util.spec_from_file_location('source22_original_release',original['path']);api=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(api)
 groups,identities=rt.recover_owned_process_inventory(run)
 for pid,birth in [(9291,2782865),(9292,2782878)]:
  ident={'boot_id':'808f6149-6c87-47f4-890d-3a9aebe15dca','pid':pid,'start_ticks':birth}
  if pid not in groups:groups.append(pid)
  if ident not in identities:identities.append(ident)
 resource=api.LinuxResources(lane['gpu_uuid']);proof=resource.require_released(groups,identities);snapshot=resource.snapshot();api.resource_gate(snapshot,state['resource_baseline'],lane['resource_policy']);assert proof['owned_process_groups_absent'] and proof['owned_cuda_pids_absent'] and proof['other_context_pids']==[] and snapshot['dxg_holders']==[]
 assert rt.metadata(run/'state.json')[1]==state_sha and rt.metadata(run/'training/budget-used.json')[1]==budget_sha and rt.metadata(run/'training/checkpoints/latest.json')[1]==latest_sha
 rt.current_control(lane)
 assert not (run/'STOP').exists() and not (run/'training/STOP').exists(),'Preserve humanSTOP before deployment'
 out=prepared/('root-deployment-'+args.stop_output.name.removeprefix('dspark-source22-fast-stable-stop-'));out.mkdir(mode=0o700,exist_ok=False)
 rt.write_new(out/'before.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_pipeline':original,'original_released':proof,'snapshot':snapshot,'baseline':state['resource_baseline'],'policy':lane['resource_policy'],'groups':groups,'identities':identities,'budget_sha256':budget_sha,'latest_sha256':latest_sha,'old_source_files':old['source_files'],'new_source_files':new['source_files'],'roles':roles})
 changes=[]
 for n in roles:
  target=base/n;before=target.read_bytes();backup=out/Path(n).name;backup.open('xb').write(before);raw=(prepared/'source'/n).read_bytes();temp=target.with_name('.'+target.name+'-source22.tmp');temp.open('xb').write(raw);temp.chmod(0o444);os.replace(temp,target);changes.append({'path':str(target),'backup':str(backup),'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(raw).hexdigest()})
 for n,h in new['source_files'].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h
 assert rt.metadata(run/'state.json')[1]==state_sha and rt.metadata(run/'training/budget-used.json')[1]==budget_sha and rt.metadata(run/'training/checkpoints/latest.json')[1]==latest_sha and rt.metadata(op/'runtime.json')[1]==rsha
 rt.write_new(out/'report.json',{'schema':'minimal_source22_root_deployment_fresh_binding_v2','endpoint_binding_sha256':args.binding_sha256,'stop_output':str(args.stop_output),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'changes':changes,'source19_runtime_not_adopted':True,'original_scientific_control_config_checkpoint_budget_and_data_unchanged':True,'old_frozenSOURCE20_unchanged':True,'new_training_or_lease':False,'original_release':proof,'resource_snapshot':snapshot,'private_source_check_sha256':check_sha})
print(json.dumps({'path':str(out/'report.json'),'sha256':hashlib.sha256((out/'report.json').read_bytes()).hexdigest(),'status':'PASS','changed':len(changes),'new_training':False}))
