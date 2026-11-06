from pathlib import Path
import datetime, fcntl, hashlib, importlib.util, json, os, signal, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('old_request_evaluator', ROOT/'runs/5080-broader-training-20261031/request-evaluator01.py')
evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
config = evaluator.load_config(ROOT/'runs/upstream-consistency-20261101/historical-component-config.json')
from benchmark_native_eagle import available_port, execute_request, request_body, wait_ready
from benchmark_dspark_screen import stop_owned_server
from evaluate_nine_model_native import native_command
from w1a1_eagle.nine_model_pipeline import load_opaque_prompts

OUT = P/'eagle-attempt-01'; OUT.mkdir(exist_ok=False)
PORT = 18942
rows = []; server = None; started = time.perf_counter(); completed = False
control = Path.home()/'.config/binary-eagle-decoding/gpu-control.json'
def check():
    state=json.loads(control.read_text())['rtx5080']
    assert not state.get('pause_requested') and not state.get('stop_all_requested')
    assert not (P/'STOP').exists()
def interrupted(sig, frame): raise KeyboardInterrupt(f'signal {sig}')
signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); check()
assert available_port('127.0.0.1',PORT)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
panel=config['panels']['development']
protocol=json.loads(Path(panel['protocol']['path']).read_text())
prompts=load_opaque_prompts(Path(panel['prompts']['path']))
selected=[x for domain in ('code','prose','reasoning') for x in [v for v in prompts if v['domain']==domain][:4]]
assert len(selected)==12 and len({x['id'] for x in selected})==12
protocol['draft_lengths']['eagle']=5
draft_path=ROOT/'models/gguf/Qwen3-4B-eagle3-q4_0.gguf'
draft={'path':str(draft_path.resolve()),'sha256':hashlib.sha256(draft_path.read_bytes()).hexdigest(),'custom_ffn':False}
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','W1AX_','DFLASH_','EAGLE_')) and k not in ('LD_LIBRARY_PATH','LD_PRELOAD','NVIDIA_TF32_OVERRIDE')}
env.update(LD_LIBRARY_PATH=str(Path(config['binary']['path']).parent),CUDA_VISIBLE_DEVICES=config['gpu_uuid'],OMP_NUM_THREADS='2',DSPARK_TARGET_WIDTH_MASK='0')
metadata={'scope':'Exploratory patch-free-target candidate gate, not final correctness/quality/performance success. Existing fork and drafter integration remain; target width policies OFF.',
    'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target':config['target'],'draft':draft,
    'binary':config['binary'],'runtime_libraries':config['runtime_libraries'],'protocol':protocol,
    'source_root':config['source_root'],'selected_ids':[x['id'] for x in selected],
    'environment':{k:v for k,v in env.items() if k.startswith(('LD_','CUDA_','OMP_','DSPARK_','GGML_'))},
    'new_training_updates':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'plan.json').write_text(json.dumps(metadata,indent=2)+'\n')
startup=warmups=0.0
try:
    for rep in range(2):
        for arm in (('target_only','eagle') if rep==0 else ('eagle','target_only')):
            check(); cell=OUT/f'rep-{rep}-{arm}';cell.mkdir()
            cmd=native_command({'inputs':{'binary':config['binary'],'target':config['target']}},protocol,'target_only' if arm=='target_only' else 'eagle_q4_0',draft,PORT)
            cmd+=['--threads','2','--flash-attn','on']
            (cell/'command.json').write_text(json.dumps({'argv':cmd,'environment':metadata['environment']},indent=2)+'\n')
            with (cell/'server.log').open('w') as log:
                try:
                    begin=time.perf_counter();server=subprocess.Popen(cmd,cwd=config['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT)
                    wait_ready(server,f'http://127.0.0.1:{PORT}',180);startup+=time.perf_counter()-begin
                    maps=Path(f'/proc/{server.pid}/maps').read_text();(cell/'loaded-library-maps.txt').write_text(maps)
                    for ref in config['runtime_libraries']:
                        if Path(ref['path']).name.startswith(('libllama','libggml','libmtmd')):assert str(Path(ref['path']).resolve()) in maps
                    request_config={'evaluation':{'max_output_tokens':128,'temperature':0.,'seed':42,'enable_thinking':False}}
                    for i,prompt in enumerate(selected[:2]):
                        check();result=execute_request(f'http://127.0.0.1:{PORT}',request_body(request_config,prompt),180,cell/f'warmup-{i}');warmups+=result['request_wall_s']
                    for i,prompt in enumerate(selected if rep==0 else reversed(selected)):
                        check();result=execute_request(f'http://127.0.0.1:{PORT}',request_body(request_config,prompt),180,cell/f'prompt-{i:02d}')
                        assert result['completion_tokens']==len(result['generated_token_ids'])>0
                        row={'arm':arm,'rep':rep,'prompt_id':prompt['id'],'result':result};rows.append(row)
                        with (OUT/'requests.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
                    print(json.dumps({'rep':rep,'arm':arm,'requests_complete':len(rows)}),flush=True)
                finally:
                    if server is not None:
                        if server.poll() is None: server.terminate()
                        try:server.wait(timeout=5)
                        except subprocess.TimeoutExpired:server.kill();server.wait()
                        server=None
    by={(r['arm'],r['rep'],r['prompt_id']):r['result'] for r in rows}
    comparisons=[]
    for rep in range(2):
        for prompt in selected:
            a=by['target_only',rep,prompt['id']]['generated_token_ids'];b=by['eagle',rep,prompt['id']]['generated_token_ids']
            first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            comparisons.append({'rep':rep,'prompt_id':prompt['id'],'tokens_equal':a==b,'first_difference':None if a==b else first,'output_lengths':[len(a),len(b)]})
    repeats={arm:all(by[arm,0,x['id']]['generated_token_ids']==by[arm,1,x['id']]['generated_token_ids'] for x in selected) for arm in ('target_only','eagle')}
    completed=True
    (OUT/'summary.json').write_text(json.dumps({'completed':True,'scope':metadata['scope'],'requests':len(rows),'comparisons':comparisons,'same_arm_repeat_identity':repeats,
        'all_tokens_equal':all(x['tokens_equal'] for x in comparisons),'training_seconds':0,
        'costs':{'measured_request_seconds':sum(x['result']['request_wall_s'] for x in rows),'excluded_startup_seconds':startup,'excluded_warmup_seconds':warmups,'total_wall_seconds':time.perf_counter()-started}},indent=2)+'\n')
finally:
    (OUT/'receipt.json').write_text(json.dumps({'completed':completed,'requests':len(rows),'training_seconds':0,'total_wall_seconds':time.perf_counter()-started},indent=2)+'\n')
