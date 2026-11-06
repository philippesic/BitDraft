from pathlib import Path
import fcntl, hashlib, json, os, signal, subprocess, time

ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def check():
    state=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']
    assert not state.get('pause_requested') and not state.get('stop_all_requested') and not (P/'STOP').exists()
def interrupted(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
check();assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
f=json.loads((P/'final-freeze.json').read_text());out=P/'final-01';out.mkdir(exist_ok=False)
started=time.perf_counter();records=[];completed=False;failure=None
try:
    for ref in f['model_files']:
        with Path(ref['path']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==ref['sha256']
    preflight=time.perf_counter()-started
    for rep in range(5):
        cells=[('invariant','target_only'),('invariant','eagle3'),('stock','target_only'),('stock','eagle3')]
        for mode,arm in cells if rep%2==0 else reversed(cells):
            check();name=f'{rep}-{mode}-{arm}';rpc=Path('/tmp')/f'be-vf-{rep}-{mode}-{arm}-20261101';rpc.mkdir(exist_ok=False)
            env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','W1AX_','DFLASH_','EAGLE_','VLLM_')) and k not in ('LD_LIBRARY_PATH','LD_PRELOAD','NVIDIA_TF32_OVERRIDE')}
            env.update(VLLM_BATCH_INVARIANT='1' if mode=='invariant' else '0',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_CACHE_ROOT=str(P/'vllm-cache'),VLLM_RPC_BASE_PATH=str(rpc))
            command=[str(P/'venv/bin/python'),'-B',str(P/'final_cell.py'),mode,arm,str(rep),str(out/name)]
            (out/(name+'-command.json')).write_text(json.dumps({'command':command,'environment':{k:v for k,v in env.items() if k.startswith(('VLLM_','CUDA_','OMP_','HF_','TRANSFORMERS_'))}},indent=2)+'\n')
            begin=time.perf_counter()
            with (out/(name+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            records.append({'mode':mode,'arm':arm,'rep':rep,'exit_code':result.returncode,'process_wall_seconds':time.perf_counter()-begin})
            (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
            if result.returncode:raise RuntimeError(f'{name} failed; preserve evidence, no engine patches')
    completed=True
except BaseException as exc:failure=repr(exc);raise
finally:
    (out/'receipt.json').write_text(json.dumps({'completed':completed,'failure':failure,'records':records,'training_seconds':0,'total_wall_seconds':time.perf_counter()-started,'scope':'ordinary upstream-engine foundation qualification only; no Binary EAGLE port or baseline migration'},indent=2)+'\n')
    subprocess.run(['tmux','-L','binary-eagle-runtime','wait-for','-S','stock-engine-confirmation-20261101-complete'],check=False)
