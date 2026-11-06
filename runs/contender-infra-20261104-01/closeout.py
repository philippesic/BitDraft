import sys,json,math,os,shutil,subprocess,time
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01');sys.path.insert(0,str(R/'source-final'))
from runner import pin,read,atomic,group_live,identity_alive
from bundle import verify
# Independently recheck every qualification response, metrics and released native cell.
qualifications=[];all_groups=set();images={}
for name in ['optimized-native','native-retry','package-native']:
 root=R/name/'qualification';result=read(root/'result.json');inputs=read(root/'inputs.json');items={v['id']:v for v in inputs['rows']+inputs['warmups']};models=read(root/'models.json');arms={v['id']:v for v in models['arms']};seen=set()
 def checked(ref):assert pin(ref['path'])==ref;return read(ref['path'])
 def sig(v):return {k:v[k] for k in ['tokens','content','stop_type','stopping_word','truncated']}
 for row in result['records']:
  key=(row['arm'],row['id'],row['repetition']);assert key not in seen;seen.add(key)
  request=checked(row['request']);response=checked(row['response']);expected=checked(items[row['id']]['reference']);assert request==checked(items[row['id']]['request']);assert sig(response)==sig(expected)
  assert row['completion_tokens']==len(response['tokens'])==response['tokens_predicted'];assert response['timings']['cache_n']==0
  metrics=checked(row['native_metrics']);delta={k:metrics['after'].get(k,0)-metrics['before'].get(k,0) for k in set(metrics['after'])|set(metrics['before'])}
  assert all(n>=0 for n in delta.values()) and delta==row['native_metrics_delta']
  assert delta['spec_decode_num_draft_tokens_total']==row['draft_proposed'];assert delta['spec_decode_num_accepted_tokens_total']==row['draft_accepted'];assert delta['spec_decode_num_drafts_total']==row['verification_rounds']>0
  assert sum(n for k,n in delta.items() if 'accepted_tokens_per_pos_total' in k)==row['draft_accepted']
  assert math.isfinite(row['full_http_wall_seconds']) and row['full_http_wall_seconds']>0
 for cell in root.glob('rep-*'):
  release=read(cell/'release.json');assert release['owned_process_reaped'] and release['owned_process_group_absent'] and release['returncode']==0
  pgid=release['identity']['pgid'];assert not group_live(pgid);all_groups.add(pgid)
  arm=arms[cell.name.removeprefix('rep-00-')];argv=read(cell/'command.json')['argv'];assert argv[0]==arm['runtime']['server']['path'];assert argv[argv.index('-md')+1]==arm['model']['path'];assert argv[argv.index('-lv')+1]=='4'
  assert pin(arm['model']['path'])==arm['model']
  for suffix in ['warm','final']:
   for ref in read(cell/f'loaded-images-{suffix}.json')['images']:images[ref['path']]=ref
 qualifications.append(dict(panel=name,responses=len(seen),cells=len(list(root.glob('rep-*')))))
for ref in images.values():assert pin(ref['path'])==ref
for path in (R/'native-retry/serving').glob('rep-*/release.json'):
 row=read(path);assert row['owned_process_group_absent'] and row['owned_process_reaped'];assert not group_live(row['identity']['pgid']);all_groups.add(row['identity']['pgid'])
# Verify all supervised and adapter identities are terminal.
states=['contender-infra-stage0-20261104-01','contender-infra-stages12-20261104-01','contender-infra-optimized-20261104-01','contender-infra-package-20261104-01']
for name in states:
 state=read(P/'runs'/name/'state.json');assert state['status']=='finished';assert state['exit_code']==(1 if 'stages12' in name else 0)
for directory in ['execution','optimized-execution','package-execution']:
 for row in read(R/directory/'state.json')['jobs'].values():
  assert row.get('group_released') and not identity_alive(row['identity'])
  assert all(not identity_alive(ref) for ref in row.get('descendants',[]))
assert not subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
verify(R/'bundle-final')
# Preserve all paths/bytes while deduplicating this task's identical immutable exports.
before={str(p):shutil.disk_usage(p).free for p in [P,Path('/mnt/c')]};dedup=[]
for paths in [[C/name/'model.gguf' for name in ['continuous','resumed','optimized']],[R/name/'draft.gguf' for name in ['bundle','bundle-verified','bundle-final']]]:
 source=paths[0];source_pin=pin(source)
 for target in paths[1:]:
  target_pin=pin(target);assert target_pin['sha256']==source_pin['sha256'] and target_pin['bytes']==source_pin['bytes'];assert source.stat().st_dev==target.stat().st_dev
  if source.stat().st_ino==target.stat().st_ino:continue
  temporary=target.with_suffix('.dedup-link');assert not temporary.exists();os.link(source,temporary);os.replace(temporary,target);assert pin(target)==target_pin
  dedup.append(dict(path=str(target),bytes=target_pin['bytes'],source=str(source)))
after={str(p):shutil.disk_usage(p).free for p in [P,Path('/mnt/c')]}
# Recheck final packaging after dedup.
verify(R/'bundle-final')
last=read(R/'package-execution/state.json');assert last['status']=='COMPLETE'
train={name:read(C/name/'result.json') for name in ['continuous','resumed','optimized']}
updates=sum(len((C/name/'updates.jsonl').read_text().splitlines()) for name in ['continuous','interrupted','resumed','optimized'])
update_wall=sum(json.loads(line)['update_seconds'] for name in ['continuous','interrupted','resumed','optimized'] for line in (C/name/'updates.jsonl').read_text().splitlines())
audit=read(R/'native-retry/audit.json');raw=[]
for root in [R,C]:
 files=[p for p in root.rglob('*') if p.is_file()];inodes={}
 for p in files:
  st=p.stat();inodes[(st.st_dev,st.st_ino)]=st.st_size
 raw.append(dict(root=str(root),files=len(files),logical_bytes=sum(p.stat().st_size for p in files),unique_inode_payload_bytes=sum(inodes.values())))
report=dict(status='CONTENDER_WORKFLOW_QUALIFIED_CURRENT_REFERENCE',cpu_tests=dict(new=35,related=73,reference=11,total=119),training=dict(physical_updates=updates,distinct_trajectory_updates_within_rehearsal=48,new_research_treatments=0,update_wall_seconds=update_wall,exact_resume=pin(R/'exact.json'),exact_workflow=pin(R/'optimized-exact.json'),exports=pin(R/'export-and-contract.json'),peak_cuda_allocated_bytes=max(v['peak_cuda_allocated_bytes'] for v in train.values())),native=dict(measured_requests=audit['measured_requests'],all_responses=audit['all_responses']+sum(v['responses'] for v in qualifications),qualifications=qualifications,released_groups=len(all_groups),repeatability=pin(R/'repeatability.json')),cost=dict(cumulative_supervised_command_wall_seconds=last['charged_seconds'],includes='setup,training,export,native panels,supervised checks and failed first native adapter admission; prior_state carries earlier costs',not_additive='Update wall is contained in supervised wall; overlapping CPU tests/probes outside supervisor are separate.'),data=pin(R/'data-probe-v2.json'),stream=pin(R/'shared-manifest-stream.json'),setup_cost=pin(R/'audit-cost.json'),baseline_assets=pin(R/'baseline-assets.json'),bundle=pin(R/'bundle-final/result.json'),cleanup=dict(supervisors=states,no_compute_processes=True,inventory=raw,dedup=dedup,free_before=before,free_after=after),limitations=['GPU qualification uses existing4B Q4_K_M row-W1A8 fixed-sign reference and frozen greedy runtime; unknown architectures/exporters/kernels require adapter qualification.','All current data windows tested; larger selected corpus/source diversity and chosen long-run schedule remain design decisions.','Final-confirmation protections tested with synthetic fixtures; dedicated untouched inputs were not opened.','Other three target baseline assets reauthenticated, not new GPU precision comparisons.','No universal early-training predictor or permanent backlog pruning established.'])
atomic(R/'completion-evidence.json',report);print(json.dumps({k:report[k] for k in ['status','cpu_tests','training','native','cost']},indent=2))
