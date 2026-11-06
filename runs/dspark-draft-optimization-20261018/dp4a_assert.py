"""Bounded original-binary synchronization diagnostic; reuses admitted consumers."""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--report',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--stop',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.source/'scripts'),str(a.source/'src')]
from benchmark_native_eagle import execute_request,request_body,wait_ready,available_port
from benchmark_dspark_screen import rows,round_summary,stop_owned_server
from evaluate_nine_model_native import native_command
from evaluate_paused_eagle_checkpoint import owner_lock,wall_cap
from w1a1_eagle.nine_model_pipeline import native_environment,process_identity,stop_signals,LinuxResources,load_opaque_prompts

def digest(path):
 return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def checked(ref):
 path=Path(ref['path']);assert digest(path)==ref['sha256'],str(path);return path
def write(path,value):
 path.write_text(json.dumps(value,indent=2)+'\n')
def authorized():
 if a.stop.exists():raise InterruptedError('optimization STOP exists')
report=json.loads(a.report.read_text());protocol=json.loads(checked(report['protocol']).read_text())
lane=json.loads(checked(report['frozen_lane']).read_text())
plan_path=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-operation-20261015-04/bindings/timed-evaluation-plan.with-policy.json')
assert digest(plan_path)=='b8034cf88b810aca5f2be7ec601ae9a74726ad1d11c90590d52177c4378bd927'
plan=json.loads(plan_path.read_text())
bundle={'inputs':{'binary':plan['runtime']['binary'],'target':plan['target']},'gpu_uuid':plan['gpu_uuid'],'environment':plan['environment']}
assert native_command(bundle,protocol,'dspark_a8',report['trained_export']['model'],protocol['port'])[0]==plan['runtime']['binary']['path']
prompts=load_opaque_prompts(checked(protocol['primary_prompts']))
assert len(prompts)==24 and protocol['draft_lengths']['dspark']==7
indices=[0]
protocol['max_output_tokens']=8;protocol['warmups_per_cell']=0
a.output.mkdir(parents=True,exist_ok=False)
owner=process_identity(os.getpid());observer=LinuxResources(bundle['gpu_uuid'])
write(a.output/'owner.json',owner)
records=[];all_trajectories={}
with owner_lock(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock',owner),stop_signals(),wall_cap(600):
 authorized()
 release=observer.require_released([],[]);resources=observer.snapshot()
 assert not release['other_context_pids'] and not resources['dxg_holders']
 write(a.output/'initial-resources.json',dict(release=release,resources=resources))
 for sync in [False]:
  authorized();stage=a.output/('sync' if sync else 'original');stage.mkdir()
  trace=stage/'rounds.jsonl';env=native_environment('dspark',8,bundle.get('environment'))
  env.update(CUDA_VISIBLE_DEVICES=bundle['gpu_uuid'],DSPARK_REQUIRE_AUTHOR_LAYOUT='1',W1AX_ROUND_TRACE_JSONL=str(trace),GGML_W1AX_ADMISSION_TRACE='1')
  env.pop('DSPARK_SYNC_COMPONENT_TIMINGS',None)
  env.update(GGML_W1AX_DP4A='1',GGML_W1AX_ASSERT_INT_DOT='1',GGML_CUDA_DISABLE_GRAPHS='1',LD_LIBRARY_PATH='/home/philip/binary-eagle-decoding/runs/dspark-draft-optimization-20261018/build-dp4a-01/bin')
  port=protocol['port'];assert available_port('127.0.0.1',port)
  command=native_command(bundle,protocol,'dspark_a8',report['trained_export']['model'],port)+['--log-verbosity',str(protocol['diagnostic_log_verbosity'])]
  command[0]='/home/philip/binary-eagle-decoding/runs/dspark-draft-optimization-20261018/build-dp4a-01/bin/llama-server'
  write(stage/'command.json',dict(argv=command,environment={k:v for k,v in env.items() if k.startswith(('GGML_', 'DSPARK_', 'W1AX_', 'CUDA_', 'OMP_', 'LD_LIBRARY_PATH'))}))
  proc=None
  with (stage/'server.log').open('wb') as log:
   try:
    authorized()
    prior=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM,signal.SIGHUP})
    try:
     proc=subprocess.Popen(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,prior))
     identity=process_identity(proc.pid);write(stage/'process.json',identity)
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,prior)
    wait_ready(proc,f'http://127.0.0.1:{port}',protocol['startup_wall_seconds'])
    cases=[(True,i,i) for i in range(protocol['warmups_per_cell'])]+[(False,i,i) for i in indices]
    for warmup,i,idx in cases:
     authorized();before=len(rows(trace));prompt=prompts[idx]
     result=execute_request(f'http://127.0.0.1:{port}',request_body({'evaluation':{'max_output_tokens':protocol['max_output_tokens'],'temperature':0.,'seed':protocol['seed'],'enable_thinking':False}},prompt),180,stage/(('warmup-' if warmup else 'prompt-')+str(i)))
     if warmup:continue
     rr=rows(trace)[before:];summary=round_summary(rr,7)
     trajectory=[{k:r.get(k) for k in ['status','replay','proposed_token_ids','emitted_token_ids','verified_token_ids','n_accepted','n_proposed','n_emitted','n_accepted_usable_prefix','round','round_index','seed_token_id','seed']} for r in rr]
     all_trajectories[(sync,idx)]=trajectory
     records.append(dict(sync=sync,prompt_index=idx,prompt_id=prompt['id'],result=result,round_summary=summary))
     write(a.output/'progress.json',records)
   finally:
    if proc is not None:stop_owned_server(proc)
  release=observer.require_released([proc.pid],[identity]);resources=observer.snapshot()
  write(stage/'released.json',dict(release=release,resources=resources))
  assert not release['other_context_pids'] and not resources['dxg_holders']
 assert 'CUDA W1A8 opt-in DP4A exact integer dot dispatch' in (a.output/'original/server.log').read_text()
 write(a.output/'complete.json',dict(status='PASS',records=len(records),scope='Exact scalar integer-dot assertions on authentic native request; graphs disabled; not timing'))
