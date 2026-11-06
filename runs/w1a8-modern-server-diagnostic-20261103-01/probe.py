"""Timeboxed native-server label diagnostic, existing pilot inputs only."""
from pathlib import Path
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/w1a8-modern-server-diagnostic-20261103-01'
def read(p): return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve()
    with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':sha}
def write(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
b = read(P/'runs/w1a8-modern-capture-build-20261103-08/build.json')
baseline = read(P/'experiments/w1a8-standing-20261102/baseline.json')
manifest = read(P/'runs/qwen35-correctness-matrix-20261101-01/pilot-freeze.json')
command_ref = P/'runs/qwen35-correctness-matrix-20261101-01/pilot-01/4B-Q4_K_M/rep-0-target_only-none-d0/command.json'
command = read(command_ref)['command']
for ref in [manifest['server'],baseline['rows'][0]['target_pin'],baseline['cuda']]+b['runtime_libraries']:
    assert pin(ref['path'])['sha256']==ref['sha256']
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
command[command.index('--port')+1]=str(port)
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE']}
env={**clean,**b['environment'],'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'}
prompts={x['id']:x for x in read(P/'runs/qwen35-correctness-matrix-20261101-01/pilot-inputs/requests.json')}
refs={}
for line in (P/'runs/qwen35-correctness-matrix-20261101-01/pilot-01/requests.jsonl').open():
    x=json.loads(line)
    if x['target_id']=='4B-Q4_K_M' and x['family']=='target_only' and x['phase']=='performance':refs[x['prompt_id']]=x['result']['tokens']
write(R/'freeze.json',{'source':pin(__file__),'server':manifest['server'],'target':baseline['rows'][0]['target_pin'],'cuda':baseline['cuda'],'command_ancestry':pin(command_ref),'command':command,'environment':b['environment'],'scope':'Existing development only; fresh and replayed server history, no timing or new quality claim'})
process=None; records=[]; started=time.monotonic()
try:
    with (R/'server.log').open('x') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        stat=Path('/proc')/str(process.pid)
        write(R/'server-identity.json',{'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid':process.pid,'pgid':os.getpgid(process.pid),'start_ticks':int((stat/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(stat/'cmdline').read_bytes().replace(b'\0',b' ').decode()})
        for _ in range(120):
            assert process.poll() is None, 'server exited during startup'
            try:
                with urllib.request.urlopen('http://127.0.0.1:%d/health'%port,timeout=1) as response:
                    if response.status==200:break
            except Exception:time.sleep(.25)
        else:raise RuntimeError('server startup timed out')
        mapped=(stat/'maps').read_text();(R/'loaded-library-maps.txt').write_text(mapped)
        cuda=sorted({x.split()[-1] for x in mapped.splitlines() if 'libggml-cuda.so' in x});assert cuda==[baseline['cuda']['path']]
        # Preserve exact API body fields. Test cold generation, repeat, then the
        # original two warmups and request order up to the long prompt.
        sequence=[('cold2',2,16),('repeat2',2,16),('warmup0',0,128),('warmup1',1,128)]+[('replay%d'%i,i,128) for i in range(5)]
        for name,pid,limit in sequence:
            control=read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'];assert not control.get('pause_requested') and not (R/'STOP').exists()
            payload={'prompt':prompts[pid]['prompt'],'n_predict':limit,'temperature':0,'seed':1234,'cache_prompt':False,'return_tokens':True,'stream':False}
            write(R/(name+'-request.json'),payload)
            request=urllib.request.Request('http://127.0.0.1:%d/completion'%port,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
            write(R/(name+'-response.json'),result)
            record={'name':name,'prompt_id':pid,'tokens':result['tokens'],'first9_matches_saved_server':result['tokens'][:9]==refs[pid][:9],'token_count':len(result['tokens'])}
            records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='tokens'}),flush=True)
finally:
    if process is not None:
        if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        write(R/'release.json',{'owned_server_reaped':process.poll() is not None,'returncode':process.returncode})
write(R/'result.json',{'status':'COMPLETE','records':records,'wall_seconds':time.monotonic()-started,'training_updates':0})
