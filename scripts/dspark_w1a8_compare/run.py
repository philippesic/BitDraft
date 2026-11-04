"""Bounded original-Qwen3 DSpark W1A8 versus released DFlash Q4 comparison."""
from pathlib import Path
import argparse,copy,json,os,signal,socket,subprocess,sys,time
P=Path('/home/philip/binary-eagle-decoding')
S=P/'runs/dspark-q4km-campaign-20261018/astra/operation/source'
sys.path[:0]=[str(S/'scripts'),str(S/'src'),str(P/'scripts/w1a8_generalization')]
from benchmark_native_eagle import execute_request,request_body,wait_ready
from benchmark_dspark_screen import stop_owned_server
from evaluate_nine_model_native import native_command
from modern_collect import pin,process_identity

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def check(ref):
 actual=pin(ref['path']);assert actual['sha256']==ref['sha256'],ref['path'];return Path(ref['path'])
def sig(r):return {k:r[k] for k in ['generated_token_ids','completion_sha256','finish_reason','completion_tokens']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();b=read(a.config);a.out.mkdir(exist_ok=False)
 assert not read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080']['pause_requested']
 for ref in [b['binary'],b['build'],b['target'],*b['runtime_libraries'],*b['source_files'],*b['models'].values(),*b['panel'].values()]:check(ref)
 assert b['target']['sha256']=='05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6'
 assert b['models']['dspark_w1a8_parent3504']['sha256']=='79f3f73f99aa06ce022bdcdd63c1653545d0304195ba4358040d668950133bd8'
 protocol=read(b['panel']['protocol']['path']);protocol['draft_lengths'].update(dspark=5,dflash=5)
 prompts=[json.loads(s) for s in Path(b['panel']['prompts']['path']).read_text().splitlines() if s.strip()]
 selected=[row for domain in ['code','prose','reasoning'] for row in [p for p in prompts if p['domain']==domain][:2]];assert len(selected)==6
 gold={(r['repetition'],r['prompt_id']):r['generated_token_ids'] for r in read(b['panel']['target_reference']['path'])['records']}
 write(a.out/'freeze.json',dict(config=pin(a.config),source=pin(__file__),selected_ids=[p['id'] for p in selected],repetitions=5,draft_limit=5,scope='original Qwen3 F16 target, retained parent3504; six reused DEV prompts, no training, no EAGLE or Qwen3.5'))
 active=None;records=[];reference={};deadline=time.monotonic()+1200
 def stop(s,f):raise InterruptedError('supervisor stop')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for rep in range(5):
  arms=list(b['models']);arms=arms if rep%2==0 else arms[::-1]
  for arm in arms:
   for traced in ([False,True] if rep%2==0 else [True,False]):
    assert time.monotonic()<deadline
    mode='trace' if traced else 'plain';cell=a.out/f'rep-{rep:02d}-{arm}-{mode}';cell.mkdir()
    model=b['models'][arm]
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    kind='target_only' if arm=='target_only' else ('dflash_q4' if arm=='dflash_q4km' else 'dspark_a8')
    cmd=native_command({'inputs':{'binary':b['binary'],'target':b['target']}},protocol,kind,model,port)
    if arm=='dflash_q4km':cmd[cmd.index('--spec-type')+1]='draft-dflash'
    cmd+=['--log-verbosity','3']
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','W1AX_','W1A8_','DSPARK_','DFLASH_','EAGLE_','LLAMA_','CUDA_','LD_','OMP_'))}
    env.update(b['environment'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(Path(b['binary']['path']).parent),DSPARK_TARGET_WIDTH_MASK='3',CUDA_CACHE_PATH=str(P/'runs/dspark-w1a8-dflash-20261104-01/cuda-cache'))
    if arm.startswith('dspark'):env.update(DSPARK_REQUIRE_AUTHOR_LAYOUT='1',DSPARK_MARKOV_ROWS='full')
    if model.get('custom_ffn'):env.update(GGML_W1AX_DP4A='1',GGML_W1A8_ALGORITHM='imma_mix',GGML_W1A8_TOKEN7='1')
    trace=cell/'rounds.jsonl'
    if traced:env['W1AX_ROUND_TRACE_JSONL']=str(trace.resolve())
    write(cell/'command.json',dict(argv=cmd,environment={k:v for k,v in env.items() if k.startswith(('DSPARK_','W1AX_','GGML_','CUDA_','OMP_','LD_','NVIDIA_'))}))
    try:
     with (cell/'server.log').open('w') as log:
      active=subprocess.Popen(cmd,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(cell/'identity.json',process_identity(active));wait_ready(active,f'http://127.0.0.1:{port}',120)
      maps=Path(f'/proc/{active.pid}/maps').read_text();(cell/'maps.txt').write_text(maps)
      for ref in b['runtime_libraries']:
       if Path(ref['path']).name.startswith(('libllama','libggml','libmtmd')):assert str(Path(ref['path']).resolve()) in maps,ref['path']
      offset=0
      for j,prompt in enumerate(selected[:2]+selected):
       warmup=j<2;dest=cell/f'request-{j:02d}';begin=time.monotonic_ns()
       result=execute_request(f'http://127.0.0.1:{port}',request_body({'evaluation':{'max_output_tokens':128,'temperature':0.,'seed':42,'enable_thinking':False}},prompt),120,dest)
       end=time.monotonic_ns();assert result['generated_token_ids']==gold[(rep,prompt['id'])],('target parity',arm,prompt['id'])
       if prompt['id'] not in reference:assert arm=='target_only';reference[prompt['id']]=sig(result)
       assert sig(result)==reference[prompt['id']],('text/stop parity',arm,prompt['id'])
       row=dict(arm=arm,repetition=rep,mode=mode,prompt_id=prompt['id'],domain=prompt['domain'],warmup=warmup,result=result,response=pin(dest/'response.json'),bounds_ns=[begin,end])
       if traced and arm!='target_only':
        allrows=[json.loads(s) for s in trace.read_text().splitlines()];rs=allrows[offset:];offset=len(allrows)
        emitted=[t for r in rs if r['status']!='checkpoint_replay' for t in r['emitted_token_ids']]
        assert emitted in (result['generated_token_ids'],result['generated_token_ids'][1:]),'trace output mismatch'
        counted=[r for r in rs if r['status']=='complete' and not r['replay']]
        counts=dict(rounds=len(counted),accepted=sum(r['n_accepted'] for r in counted),proposed=sum(r['n_proposed'] for r in counted));assert counts==result['speculative'],('counter mismatch',counts,result['speculative'])
        row['cost_us']={k:sum(r[k] for r in rs) for k in ['draft_us','process_us','target_decode_sync_us','check_us','kv_repair_us','accept_hook_us','round_us','process_feature_copy_us','process_encoder_us','process_draft_decode_us']}
        row['trace_rows']=len(rs)
       records.append(row);write(dest/'receipt.json',row)
       with (a.out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    finally:
     if active is not None:stop_owned_server(active);write(cell/'release.json',dict(reaped=active.poll() is not None,returncode=active.returncode));active=None
    print(json.dumps(dict(stage='cell_done',rep=rep,arm=arm,mode=mode)),flush=True)
 lookup={(r['arm'],r['repetition'],r['prompt_id'],r['warmup']):r for r in records if r['mode']=='plain'}
 for row in records:
  if row['mode']=='trace':
   other=lookup.pop((row['arm'],row['repetition'],row['prompt_id'],row['warmup']));assert sig(row['result'])==sig(other['result']) and row['result']['speculative']==other['result']['speculative'],'trace/plain mismatch'
 assert not lookup
 write(a.out/'result.json',dict(status='MATCHED_ORIGINAL_DSPARK_DFLASH_PASS',responses=len(records),all_outputs_and_trace_counters_exact=True))
if __name__=='__main__':main()
