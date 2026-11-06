"""Fixed three-arm clean comparison using unchanged native consumers/protocol."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stop',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.source/'scripts'),str(a.source/'src')]
from benchmark_native_eagle import execute_request,request_body,wait_ready,available_port
from benchmark_dspark_screen import stop_owned_server
from evaluate_nine_model_native import native_command
from evaluate_paused_eagle_checkpoint import owner_lock,wall_cap
from w1a1_eagle.nine_model_pipeline import native_environment,process_identity,stop_signals,LinuxResources,load_opaque_prompts

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(ref):
 path=Path(ref['path']);assert sha(path)==ref['sha256'];return path
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def authorized():
 if a.stop.exists():raise InterruptedError('optimization STOP')
report=json.loads(a.report.read_text());protocol=json.loads(check(report['protocol']).read_text())
plan_path=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-operation-20261015-04/bindings/timed-evaluation-plan.with-policy.json');assert sha(plan_path)=='b8034cf88b810aca5f2be7ec601ae9a74726ad1d11c90590d52177c4378bd927';plan=json.loads(plan_path.read_text())
bundle={'inputs':{'binary':plan['runtime']['binary'],'target':plan['target']}}
prompts=load_opaque_prompts(check(protocol['primary_prompts']));assert len(prompts)==24 and protocol['repetitions']==5
models={'original':report['trained_export']['model'],'dp4a':report['trained_export']['model'],'eagle_q4':plan['control']['model']}
for arm,model in models.items():
 cmd=native_command(bundle,protocol,'eagle_q4' if arm=='eagle_q4' else 'dspark_a8',model,protocol['port']);assert all(Path(x).is_file() for x in [cmd[0],model['path'],plan['target']['path']])
assert a.output.is_dir();records=json.loads((a.output/'progress.json').read_text());assert len(records)==48
assert {(r['repetition'],r['arm']) for r in records}=={(0,'original'),(0,'dp4a')}
for arm in ['original','dp4a']:
 stage=a.output/'rep-00'/arm
 assert (stage/'released.json').is_file() and (stage/'cuda-library-maps.json').is_file()
 for i,prompt in enumerate(prompts):
  row=next(r for r in records if r['arm']==arm and r['prompt_id']==prompt['id']);assert json.loads((stage/f'prompt-{i:04d}'/'measurement.json').read_text())==row['result']
for prompt in prompts:
 pair=[next(r for r in records if r['arm']==arm and r['prompt_id']==prompt['id']) for arm in ['original','dp4a']]
 assert pair[0]['result']['generated_token_ids']==pair[1]['result']['generated_token_ids'] and pair[0]['result']['speculative']==pair[1]['result']['speculative']
snapshot=a.output/'initial48-progress.json';assert not snapshot.exists();snapshot.write_bytes((a.output/'progress.json').read_bytes())
parent=a.output.parent;prior_state=parent/'supervision/dp4a-clean-01/state.json';prior=json.loads(prior_state.read_text());assert prior['status']=='finished' and prior['exit_code']==1
metadata=[prior_state,parent/'supervision/dp4a-clean-01/stdout.log',parent/'clean_compare_dp4a_v2.py']+[a.output/'rep-00'/arm/name for arm in ['original','dp4a'] for name in ['command.json','process.json','released.json','cuda-library-maps.json']]
write(a.output/'recovery-admission.json',dict(original_progress_sha256=sha(snapshot),retained_requests=48,matched_original_candidate_outputs_counts=True,original_metadata=[dict(path=str(p),sha256=sha(p)) for p in metadata],reason='clean verbosity3 suppresses INFO marker; actual mapped-library and diagnostic marker already validated',runtime_or_protocol_changes=False))
owner=process_identity(os.getpid());write(a.output/'recovery-owner.json',owner);observer=LinuxResources(plan['gpu_uuid'])
with owner_lock(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock',owner),stop_signals(),wall_cap(1200):
 authorized();release=observer.require_released([],[]);res=observer.snapshot();assert not release['other_context_pids'] and not res['dxg_holders'];write(a.output/'recovery-resources.json',dict(release=release,resources=res))
 for rep in range(5):
  arms=['original','dp4a','eagle_q4'];arms=arms[rep%3:]+arms[:rep%3]
  if rep%2:arms=list(reversed(arms))
  for arm in arms:
   if rep==0 and arm in ['original','dp4a']:continue
   authorized();stage=a.output/f'rep-{rep:02d}'/arm;stage.mkdir(parents=True)
   family='eagle' if arm=='eagle_q4' else 'dspark';env=native_environment(family,None if family=='eagle' else 8,plan['environment']);env['CUDA_VISIBLE_DEVICES']=plan['gpu_uuid']
   for name in ['GGML_W1AX_DP4A','GGML_W1AX_WARP_REDUCE','GGML_W1AX_ASSERT_INT_DOT','GGML_CUDA_DISABLE_GRAPHS','DSPARK_SYNC_COMPONENT_TIMINGS','W1AX_ROUND_TRACE_JSONL','GGML_W1AX_ADMISSION_TRACE']:env.pop(name,None)
   env['LD_LIBRARY_PATH']='/home/philip/binary-eagle-decoding/runs/build/native-seven-5eb221-sm120-20261006/bin'
   if family=='dspark':env['DSPARK_REQUIRE_AUTHOR_LAYOUT']='1'
   if arm=='dp4a':env['GGML_W1AX_DP4A']='1';env['LD_LIBRARY_PATH']='/home/philip/binary-eagle-decoding/runs/dspark-draft-optimization-20261018/build-dp4a-01/bin'
   port=protocol['port'];assert available_port('127.0.0.1',port)
   command=native_command(bundle,protocol,'eagle_q4' if family=='eagle' else 'dspark_a8',models[arm],port)+['--log-verbosity',str(protocol['clean_log_verbosity'])]
   if arm=='dp4a':command[0]='/home/philip/binary-eagle-decoding/runs/dspark-draft-optimization-20261018/build-dp4a-01/bin/llama-server'
   write(stage/'command.json',dict(argv=command,environment={k:v for k,v in env.items() if k.startswith(('GGML_','DSPARK_','W1AX_','CUDA_','OMP_','LD_LIBRARY_PATH'))}))
   proc=None
   with (stage/'server.log').open('wb') as log:
    try:
     prior=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM,signal.SIGHUP})
     try:
      proc=subprocess.Popen(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,prior));identity=process_identity(proc.pid);write(stage/'process.json',identity)
     finally:signal.pthread_sigmask(signal.SIG_SETMASK,prior)
     wait_ready(proc,f'http://127.0.0.1:{port}',protocol['startup_wall_seconds'])
     maps=[line for line in Path(f'/proc/{proc.pid}/maps').read_text().splitlines() if 'libggml-cuda.so' in line]
     expected_library=str((Path(env['LD_LIBRARY_PATH'])/'libggml-cuda.so.0.25.1').resolve());assert maps and all(expected_library in line for line in maps);write(stage/'cuda-library-maps.json',dict(expected=expected_library,observed=maps))
     for warmup,i,prompt in [(True,i,prompts[i]) for i in range(protocol['warmups_per_cell'])]+[(False,i,p) for i,p in enumerate(prompts)]:
      authorized();out=execute_request(f'http://127.0.0.1:{port}',request_body({'evaluation':{'max_output_tokens':protocol['max_output_tokens'],'temperature':0.,'seed':protocol['seed'],'enable_thinking':False}},prompt),180,stage/(('warmup-' if warmup else 'prompt-')+f'{i:04d}'))
      if not warmup:
       assert out['generated_token_ids'];records.append(dict(arm=arm,repetition=rep,prompt_id=prompt['id'],result=out));write(a.output/'progress.json',records)
    finally:
     if proc is not None:stop_owned_server(proc)
   release=observer.require_released([proc.pid],[identity]);res=observer.snapshot();assert not release['other_context_pids'] and not res['dxg_holders'];write(stage/'released.json',dict(release=release,resources=res))
   # Clean verbosity suppresses INFO markers; actual mapped CUDA library was checked above.
 for rep in range(5):
  for prompt in prompts:
   pair=[next(r for r in records if r['arm']==arm and r['repetition']==rep and r['prompt_id']==prompt['id']) for arm in ['original','dp4a']]
   assert pair[0]['result']['generated_token_ids']==pair[1]['result']['generated_token_ids']
   assert pair[0]['result']['speculative']==pair[1]['result']['speculative']
 summary={}
 for arm in models:
  rr=[r['result'] for r in records if r['arm']==arm];tokens=sum(x['completion_tokens'] for x in rr);seconds=sum(x['request_wall_s'] for x in rr)
  summary[arm]=dict(requests=len(rr),tokens=tokens,seconds=seconds,tokens_per_s=tokens/seconds,speculative={k:sum(x['speculative'][k] for x in rr) for k in ['accepted','proposed','rounds']})
 write(a.output/'summary.json',dict(status='COMPLETE',exact_dspark_output_and_counts=True,quality='original strict FAILED/research-only preserved; no new DSpark differences',cells=summary))
 print(json.dumps(summary),flush=True)
