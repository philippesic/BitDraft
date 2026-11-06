"""Frozen no-target-override F16/EAGLE qualification; never trains."""
from pathlib import Path
import datetime, fcntl, hashlib, importlib.util, json, os, signal, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
frozen=json.loads((P/'confirmation-freeze.json').read_text())
for ref in frozen['sources']:
    assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
inputs=json.loads((P/'confirmation-inputs/receipt.json').read_text())
assert inputs['freeze_sha256']==hashlib.sha256((P/'confirmation-freeze.json').read_bytes()).hexdigest()
for name,sha in inputs['files'].items():assert hashlib.sha256((P/'confirmation-inputs'/name).read_bytes()).hexdigest()==sha
spec=importlib.util.spec_from_file_location('request_evaluator',ROOT/'runs/5080-broader-training-20261031/request-evaluator01.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
c=e.load_config(ROOT/'runs/upstream-consistency-20261101/historical-component-config.json')
from benchmark_native_eagle import available_port,execute_request,request_body,wait_ready
from evaluate_nine_model_native import native_command
sys.path.insert(0,str(ROOT/'runs/upstream-consistency-20261101'));import quality_judge
out=P/'confirmation-01';out.mkdir(exist_ok=False)
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
port=18943;server=None;started=time.perf_counter();rows=[];quality=[];startup=warmup=0.;completed=False;failure=None
def check():
    state=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']
    assert not state.get('pause_requested') and not state.get('stop_all_requested') and not (P/'STOP').exists()
def interrupted(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
check();assert available_port('127.0.0.1',port)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
protocol=json.loads(Path(c['panels']['development']['protocol']['path']).read_text())
protocol['draft_lengths']['eagle']=5
prompts=json.loads((P/'confirmation-inputs/requests.json').read_text())
cases=json.loads((P/'confirmation-inputs/quality-cases.json').read_text())
assert len(prompts)==len(cases)==24
draft=frozen['draft']
with Path(draft['path']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==draft['sha256']
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','W1AX_','DFLASH_','EAGLE_')) and k not in ('LD_LIBRARY_PATH','LD_PRELOAD','NVIDIA_TF32_OVERRIDE')}
env.update(LD_LIBRARY_PATH=str(Path(c['binary']['path']).parent),CUDA_VISIBLE_DEVICES=c['gpu_uuid'],OMP_NUM_THREADS='2',DSPARK_TARGET_WIDTH_MASK='0')
signature=lambda r:tuple(r[k] if k!='generated_token_ids' else tuple(r[k]) for k in ('generated_token_ids','completion_sha256','finish_reason'))
try:
    for phase in ('quality','performance'):
        for rep in range(1 if phase=='quality' else 5):
            for arm in (('target_only','eagle') if rep%2==0 else ('eagle','target_only')):
                check();cell=out/f'{phase}-rep-{rep}-{arm}';cell.mkdir()
                cmd=native_command({'inputs':{'binary':c['binary'],'target':c['target']}},protocol,'target_only' if arm=='target_only' else 'eagle_q4_0',draft,port)+['--threads','2','--flash-attn','on']
                (cell/'command.json').write_text(json.dumps({'argv':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('LD_','CUDA_','OMP_','DSPARK_','GGML_'))}},indent=2)+'\n')
                with (cell/'server.log').open('w') as log:
                    try:
                        begin=time.perf_counter();server=subprocess.Popen(cmd,cwd=c['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT)
                        wait_ready(server,f'http://127.0.0.1:{port}',180);startup+=time.perf_counter()-begin
                        maps=Path(f'/proc/{server.pid}/maps').read_text();(cell/'loaded-library-maps.txt').write_text(maps)
                        for ref in c['runtime_libraries']:
                            if Path(ref['path']).name.startswith(('libllama','libggml','libmtmd')):assert str(Path(ref['path']).resolve()) in maps
                        items=cases if phase=='quality' else prompts
                        items=items if rep%2==0 else list(reversed(items))
                        request_config={'evaluation':{'max_output_tokens':384 if phase=='quality' else 128,'temperature':0.,'seed':42,'enable_thinking':False}}
                        if phase=='performance':
                            for i,item in enumerate(prompts[:2]):
                                text=item['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
                                result=execute_request(f'http://127.0.0.1:{port}',request_body(request_config,{'messages':[{'role':'user','content':text}]}),180,cell/f'warmup-{i}')
                                warmup+=result['request_wall_s']
                        for i,item in enumerate(items):
                            check();text=item['prompt'] if phase=='quality' else item['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
                            folder=cell/f'prompt-{i:02d}';result=execute_request(f'http://127.0.0.1:{port}',request_body(request_config,{'messages':[{'role':'user','content':text}]}),180,folder)
                            assert result['completion_tokens']==len(result['generated_token_ids'])>0
                            counters=result['speculative'];assert all(type(v)is int and v>=0 for v in counters.values())
                            assert counters['accepted']<=counters['proposed']
                            response=json.loads((folder/'response.json').read_text())
                            row={'arm':arm,'rep':rep,'id':item['id'],'result':result}
                            if phase=='quality':
                                answer=response['choices'][0]['message']['content'];row['judgement']=quality_judge.judge(item,answer);quality.append(row)
                            else:rows.append(row)
                            with (out/(phase+'.jsonl')).open('a') as s:s.write(json.dumps(row)+'\n')
                        print(json.dumps({'phase':phase,'rep':rep,'arm':arm,'performance_requests':len(rows),'quality_requests':len(quality)}),flush=True)
                    finally:
                        if server is not None:
                            if server.poll() is None:server.terminate()
                            try:server.wait(timeout=5)
                            except subprocess.TimeoutExpired:server.kill();server.wait()
                            server=None
    by={(r['arm'],r['rep'],r['id']):r['result'] for r in rows}
    mismatches=[{'arm':r['arm'],'rep':r['rep'],'id':r['id']} for r in rows if signature(r['result'])!=signature(by['target_only',r['rep'],r['id']]) or signature(r['result'])!=signature(by[r['arm'],0,r['id']])]
    qb={(r['arm'],r['id']):r for r in quality}
    qdiff=[item['id'] for item in cases if signature(qb['eagle',item['id']]['result'])!=signature(qb['target_only',item['id']]['result']) or qb['eagle',item['id']]['judgement']!=qb['target_only',item['id']]['judgement']]
    cells={}
    for arm in ('target_only','eagle'):
        subset=[r['result'] for r in rows if r['arm']==arm];tokens=sum(r['completion_tokens'] for r in subset);wall=sum(r['request_wall_s'] for r in subset)
        cells[arm]={'requests':len(subset),'tokens':tokens,'full_request_seconds':wall,'tokens_per_second':tokens/wall,
            'by_repetition':[sum(r['result']['completion_tokens'] for r in rows if r['arm']==arm and r['rep']==rep)/sum(r['result']['request_wall_s'] for r in rows if r['arm']==arm and r['rep']==rep) for rep in range(5)],
            'independent_quality_passed':sum(r['judgement']['passed'] for r in quality if r['arm']==arm),'quality_cases':24}
    completed=True
    (out/'summary.json').write_text(json.dumps({'completed':True,'performance_requests':len(rows),'quality_requests':len(quality),'performance_mismatches':mismatches,'quality_mismatches':qdiff,'cells':cells,
        'scope':'Qwen3-4B F16 target, Q4_0 EAGLE D5, private existing drafter integration; target consistency overrides OFF; fixed settings only; not upstream-stock ancestry proof or universal invariance',
        'costs':{'training_seconds':0,'performance_request_seconds':sum(r['result']['request_wall_s'] for r in rows),'quality_request_seconds':sum(r['result']['request_wall_s'] for r in quality),'excluded_startup_seconds':startup,'excluded_performance_warmup_seconds':warmup,'total_wall_seconds':time.perf_counter()-started}},indent=2)+'\n')
except BaseException as exc:
    failure=repr(exc);raise
finally:
    (out/'receipt.json').write_text(json.dumps({'completed':completed,'failure':failure,'performance_requests':len(rows),'quality_requests':len(quality),'training_seconds':0,'total_wall_seconds':time.perf_counter()-started},indent=2)+'\n')
