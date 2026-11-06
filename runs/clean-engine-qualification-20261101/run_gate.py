from pathlib import Path
import datetime, fcntl, hashlib, json, os, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']
assert not state.get('pause_requested') and not state.get('stop_all_requested')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','W1AX_','DFLASH_','EAGLE_')) and k not in ('LD_LIBRARY_PATH','LD_PRELOAD','NVIDIA_TF32_OVERRIDE')}
env.update(VLLM_BATCH_INVARIANT='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_CACHE_ROOT=str(P/'vllm-cache'),VLLM_RPC_BASE_PATH=str(Path('/tmp')/('be-vllm-'+sys.argv[1]+'-20261101')))
assert len(sys.argv)==2 and sys.argv[1].startswith('gate-') and '/' not in sys.argv[1]
out=P/sys.argv[1];out.mkdir(exist_ok=False);Path(env['VLLM_RPC_BASE_PATH']).mkdir(exist_ok=False);records=[];start=time.perf_counter();complete=False
try:
    for arm in ('target_only','eagle3'):
        command=[str(P/'venv/bin/python'),'-B',str(P/'cell.py'),arm,str(out/arm)]
        (out/(arm+'-command.json')).write_text(json.dumps({'command':command,'cell_source_sha256':hashlib.sha256((P/'cell.py').read_bytes()).hexdigest(),'environment':{k:v for k,v in env.items() if k.startswith(('VLLM_','CUDA_','OMP_','HF_','TRANSFORMERS_'))}},indent=2)+'\n')
        begin=time.perf_counter()
        with (out/(arm+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append({'arm':arm,'exit_code':result.returncode,'process_wall_seconds':time.perf_counter()-begin})
        if result.returncode:raise RuntimeError(f'{arm} exited {result.returncode}; no source patches allowed')
    a=[json.loads(s) for s in (out/'target_only/requests.jsonl').read_text().splitlines()]
    b=[json.loads(s) for s in (out/'eagle3/requests.jsonl').read_text().splitlines()]
    assert len(a)==len(b)==24 and [r['id'] for r in a]==[r['id'] for r in b]
    differences=[{'id':x['id'],'tokens_equal':x['token_ids']==y['token_ids'],'text_equal':x['text']==y['text'],'stop_equal':(x['finish_reason'],x['stop_reason'])==(y['finish_reason'],y['stop_reason'])} for x,y in zip(a,b)]
    (out/'comparisons.json').write_text(json.dumps(differences,indent=2)+'\n');complete=True
finally:
    (out/'receipt.json').write_text(json.dumps({'completed':complete,'records':records,'training_seconds':0,'total_wall_seconds':time.perf_counter()-start,'scope':'ordinary upstream engine foundation gate only; no custom binary port or migration'},indent=2)+'\n')
