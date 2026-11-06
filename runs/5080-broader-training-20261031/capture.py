"""Finite TRAIN-only serving capture through unchanged deployment and verifier."""
from pathlib import Path
import json,sys,os,subprocess,time,signal,hashlib,urllib.request
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');A=ROOT/'runs/w1a8-algorithms-20261020'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2)
def main():
 import fcntl
 lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 base=json.loads((A/'openitems-markov-benchmark06-root/bindings-development.json').read_text())
 sys.path[:0]=[str(Path(base['source_root'])/'scripts'),str(Path(base['source_root'])/'src'),str(ROOT/'scripts')]
 from shortlist_pilots.evaluate import environment
 from benchmark_native_eagle import wait_ready,available_port
 from benchmark_dspark_screen import stop_owned_server
 from evaluate_nine_model_native import native_command
 out=RUN/'capture05';out.mkdir(exist_ok=False);split=json.loads((RUN/'split.json').read_text());ready=json.loads((RUN/'native02/ready.json').read_text());base=json.loads((A/'openitems-markov-benchmark06-root/bindings-development.json').read_text());protocol=json.loads(Path(base['protocol']['path']).read_text());protocol['draft_lengths']['dspark']=5;model=base['models']['parent3504'];plan=json.loads(Path(base['plan']['path']).read_text());port=18327
 assert available_port('127.0.0.1',port)
 config={'binary':ready['server'],'gpu_uuid':plan['gpu_uuid'],'environment':{'OMP_NUM_THREADS':'2','NVIDIA_TF32_OVERRIDE':'0'}}
 env=environment(config,model);env['LD_LIBRARY_PATH']=str(RUN/'native02/bin')+':'+str(ROOT/'runs/shortlist-throughput-20261027/native/build/bin')
 env.update(DSPARK_QUALITY_CAPTURE='1',EAGLE_CAPTURE_PREFIX=str(out/'capture'),EAGLE_CAPTURE_TARGET_FEATURES='1',EAGLE_CAPTURE_TARGET_FEATURES_LIMIT='300000',EAGLE_RECORD_ROUNDS_JSONL=str(out/'rounds.jsonl'))
 cmd=native_command({'inputs':{'binary':ready['server'],'target':plan['target']}},protocol,'dspark_a8',model,port)+['--no-context-shift','--ctx-checkpoints','0','--log-verbosity','3']
 assert cmd[cmd.index('--spec-draft-n-max')+1]=='5'
 write(out/'command.json',{'argv':cmd,'environment':env,'split':pin(RUN/'split.json'),'base_protocol':base['protocol'],'target':plan['target'],'model':model})
 proc=None;started=time.monotonic();records=[]
 try:
  with (out/'server.log').open('wb') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(out/'process.json',{'pid':proc.pid,'unix':time.time()});wait_ready(proc,f'http://127.0.0.1:{port}',180)
   (out/'loaded-library-maps.txt').write_text(Path(f'/proc/{proc.pid}/maps').read_text())
   for case in split['cases']:
    assert time.monotonic()-started<3600 and not (RUN/'STOP').exists()
    assert __import__('shutil').disk_usage(RUN).free>12*1024**3
    before_rounds=len((out/'rounds.jsonl').read_text().splitlines())
    body={'prompt':case['prompt_tokens'],'n_predict':64,'temperature':0.,'seed':42,'stream':False,'cache_prompt':False,'return_tokens':True,'verbose':True}
    request=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(request,timeout=120) as response:result=json.load(response)
    write(out/f'request-{case["index"]:02d}.json',{'case':case,'body':body,'response':result,'diagnostic_wall_seconds':time.monotonic()-start})
    rr=[json.loads(l) for l in (out/'rounds.jsonl').read_text().splitlines()[before_rounds:]]
    assert rr and len({r['task_id'] for r in rr})==1 and all(0<len(r['draft_token_ids'])<=5 for r in rr), 'actual corrected-D5 contract failed'
    records.append({'case_index':case['index'],'task_id':rr[0]['task_id'],'rounds':len(rr),'response':pin(out/f'request-{case["index"]:02d}.json')})
    (out/'progress.json').write_text(json.dumps({'completed_requests':len(records),'groups':len(records),'wall_seconds':time.monotonic()-started}))
 finally:
  if proc is not None:stop_owned_server(proc)
  write(out/'release.json',{'reaped':proc is not None and proc.poll() is not None,'returncode':proc.returncode if proc else None})
 write(out/'complete.json',{'status':'SERVING_CAPTURE_COMPLETE','records':records,'wall_seconds':time.monotonic()-started,'diagnostic_only':True,'source_split':'TRAIN','request_panels_used':False,'files':{p.name:pin(p) for p in out.glob('capture.*')}})
 print(json.dumps({'status':'SERVING_CAPTURE_COMPLETE','groups':len(records),'wall_seconds':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
