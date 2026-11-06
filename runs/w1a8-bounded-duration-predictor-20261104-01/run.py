from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,sys
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');N=P/'runs/w1a8-bounded-duration-predictor-20261104-01'
plan=json.loads((N/'plan.json').read_text());started=time.monotonic()
def pin(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h)
for key in ['source','parent','old3840_reference','expanded5760_result']:
    assert pin(plan[key]['path'])==plan[key]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
child=subprocess.Popen(plan['argv'],cwd=W,env=env)
q=Path('/proc')/str(child.pid)
identity=dict(pid=child.pid,pgid=os.getpgid(child.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),root_owner=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(N/'trainer-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
code=child.wait();(N/'training-process.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-started,output=plan['output']),indent=2)+'\n');assert code==0,code
import torch
torch.set_num_threads(1);sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
from modern_train_protected import equal
old=torch.load(plan['old3840_reference']['path'],map_location='cpu',weights_only=False)
path=Path(plan['output'])/'checkpoint-00003840.pt';new=torch.load(path,map_location='cpu',weights_only=False)
fields=['cursor','scales','optimizer','rng','cuda_rng','python_rng','exposure','ce_stage_exposure']
checks={key:equal(old[key],new[key]) for key in fields}
proof=dict(status='OLD_RESIDENT3840_REPRODUCTION_EXACT' if all(checks.values()) else 'OLD_RESIDENT3840_REPRODUCTION_MISMATCH',all_exact=all(checks.values()),checks=checks,reference=plan['old3840_reference'],checkpoint=pin(path),scope='Original uncached old3840 vs resident old2400→5760 intermediate3840. Training/model states exact; source-specific checkpoint contracts/schemas intentionally differ.')
(N/'old3840-parity.json').write_text(json.dumps(proof,indent=2)+'\n');assert proof['all_exact'],checks
result=json.loads((Path(plan['output'])/'result.json').read_text());assert result['updates']==5760 and result['expanded_ce'] is False
assert pin(result['model']['path'])==result['model'] and pin(result['checkpoint']['path'])==result['checkpoint']
final=torch.load(result['checkpoint']['path'],map_location='cpu',weights_only=False);assert final['cursor']==5760 and final['contract']['expanded_ce'] is False
assert len(final['optimizer']['state'])==9
for state in final['optimizer']['state'].values():assert float(state['step'])==5760
record=dict(status='OLD_ONLY5760_TRAINED_OLD3840_REPRODUCTION_PASS_NOT_NATIVE_ADMITTED',plan=pin(N/'plan.json'),parity=pin(N/'old3840-parity.json'),training_result=pin(Path(plan['output'])/'result.json'),model=result['model'],checkpoint=result['checkpoint'],wall_seconds=time.monotonic()-started,scope=plan['isolation'],evaluation_scope=plan['evaluation_scope'])
(N/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
