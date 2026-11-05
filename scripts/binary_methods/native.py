"""Bounded same-weight mechanism screen; forced A1 is not A1-trained quality."""
from pathlib import Path
import argparse,json,os,signal,socket,subprocess,sys,time,hashlib
P=Path('/home/philip/binary-eagle-decoding');S=P/'runs/dspark-q4km-campaign-20261018/astra/operation/source'
sys.path[:0]=[str(S/'scripts'),str(S/'src'),str(P/'scripts/w1a8_generalization')]
from benchmark_native_eagle import execute_request,request_body,wait_ready
from benchmark_dspark_screen import stop_owned_server
from evaluate_nine_model_native import native_command
from modern_collect import pin,process_identity
R=P/'runs/binary-methods-20261104-01'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trace',action='store_true');ap.add_argument('--build',default='native');ap.add_argument('--arms',default='A8_tensor,A8_dp4a,A8_planes7,A8_lut4,A1_xor,A1_bmma,A1_tensor');ap.add_argument('--reps',type=int,default=2);ap.add_argument('--out',required=True);a=ap.parse_args();out=R/a.out;out.mkdir();b=read(P/'runs/dspark-w1a8-dflash-20261104-01/config.json')
 assert not read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080']['pause_requested']
 for ref in [b['binary'],b['target'],*b['runtime_libraries'],b['models']['dspark_w1a8_parent3504'],*b['panel'].values()]:assert pin(ref['path'])['sha256']==ref['sha256'],ref['path']
 protocol=read(b['panel']['protocol']['path']);protocol['draft_lengths'].update(dspark=5,dflash=5)
 prompts=[json.loads(s) for s in Path(b['panel']['prompts']['path']).read_text().splitlines() if s.strip()];selected=[next(p for p in prompts if p['domain']==d) for d in ['code','prose','reasoning']]
 if a.trace:selected=selected[:1]
 gold={(r['repetition'],r['prompt_id']):r['generated_token_ids'] for r in read(b['panel']['target_reference']['path'])['records']}
 model=b['models']['dspark_w1a8_parent3504'];binary=R/a.build/'bin/llama-server';bdict={'inputs':{'binary':{'path':str(binary)},'target':b['target']}}
 write(out/'freeze.json',dict(config=pin(P/'runs/dspark-w1a8-dflash-20261104-01/config.json'),model=model,source=pin(__file__),build=pin(R/a.build/'build.json'),arms=a.arms,reps=a.reps,selected=[p['id'] for p in selected],trace=a.trace,warning='A1 runtime precision override on retained A8 weights; no training; historical Qwen3 F16 target, not modern primary baseline'))
 active=None;records=[]
 def stop(*args):raise InterruptedError('supervisor stop')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for rep in range(a.reps):
  arms=a.arms.split(',');arms=arms if rep%2==0 else arms[::-1]
  for arm in arms:
   cell=out/f'{rep}-{arm}';cell.mkdir()
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   cmd=native_command(bdict,protocol,'dspark_a8',model,port)
   env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','W1AX_','W1A8_','DSPARK_','DFLASH_','EAGLE_','LLAMA_','CUDA_','LD_','OMP_'))}
   env.update({k:v for k,v in os.environ.items() if k=='LD_PRELOAD' or k.startswith('CUDA_INJECTION')});env.update(b['environment'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(binary.parent),DSPARK_TARGET_WIDTH_MASK='3',DSPARK_REQUIRE_AUTHOR_LAYOUT='1',DSPARK_MARKOV_ROWS='full',GGML_W1AX_DP4A='1',GGML_W1A8_TOKEN7='1',GGML_W1A8_ALGORITHM={'tensor':'imma_mix','dp4a':'dp4a','planes7':'planes7','lut4':'lut4','sharedlut':'sharedlut'}.get(arm.split('_')[1],'imma_mix'))
   if arm.startswith('A1'):env.update(GGML_SCREEN_A1='1',GGML_SCREEN_A1_METHOD=arm.split('_')[1])
   if a.trace:env['W1AX_ROUND_TRACE_JSONL']=str(cell/'rounds.jsonl')
   write(cell/'command.json',dict(argv=cmd,environment={k:v for k,v in env.items() if k.startswith(('DSPARK_','W1AX_','GGML_','CUDA_','LD_'))}))
   try:
    with (cell/'server.log').open('w') as log:
     active=subprocess.Popen(cmd,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(cell/'identity.json',process_identity(active));wait_ready(active,f'http://127.0.0.1:{port}',120)
     maps=Path(f'/proc/{active.pid}/maps').read_text();(cell/'maps.txt').write_text(maps);assert str(R/a.build/'bin/libggml-cuda.so.0.25.1') in maps
     for j,prompt in enumerate(selected[:1]+selected):
      warmup=j==0;dest=cell/f'request-{j}';begin=time.time_ns();res=execute_request(f'http://127.0.0.1:{port}',request_body({'evaluation':{'max_output_tokens':128,'temperature':0.,'seed':42,'enable_thinking':False}},prompt),120,dest);end=time.time_ns()
      assert res['generated_token_ids']==gold[(rep,prompt['id'])],('target parity',arm,prompt['id'])
      row=dict(arm=arm,repetition=rep,prompt_id=prompt['id'],domain=prompt['domain'],warmup=warmup,result=res,bounds_unix_ns=[begin,end]);records.append(row)
      with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   finally:
    if active is not None:
     if active.poll() is None:
      active.send_signal(signal.SIGINT)
      try:active.wait(timeout=20)
      except subprocess.TimeoutExpired:stop_owned_server(active)
     write(cell/'release.json',dict(reaped=active.poll() is not None,returncode=active.returncode));active=None
   print(json.dumps(dict(arm=arm,rep=rep,status='target_parity_pass')),flush=True)
 # Arithmetic mechanisms must preserve draft acceptance within each precision.
 reference={}
 for row in records:
  key=(row['arm'].split('_')[0],row['repetition'],row['prompt_id'],row['warmup']);s=row['result']['speculative']
  if key not in reference:reference[key]=s
  assert s==reference[key],('draft counter mismatch',key,row['arm'],s,reference[key])
 write(out/'result.json',dict(status='TARGET_PARITY_AND_WITHIN_PRECISION_COUNTERS_PASS',requests=len(records)))
if __name__=='__main__':main()
