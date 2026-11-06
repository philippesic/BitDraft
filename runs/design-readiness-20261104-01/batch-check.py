from pathlib import Path
import subprocess,json,os,time,sys
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';C=Path(json.loads((R/'freeze.json').read_text())['output']);assert json.loads((R/'followup-result.json').read_text())['status']=='BOUNDED_GRADIENT_OBSERVER_AND_SYNTHETIC_IO_COMPLETE'
argv=[str(P/'.venv/bin/python'),str(P/'scripts/design_readiness/batch_current.py'),'--freeze',str(R/'freeze.json'),'--run-output',str(R/'batch-check01'),'--out-root',str(C/'batch-check01'),'--recipes','control','control','--final-updates','5808','--allow-cuda'];env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','CUBLAS_WORKSPACE_CONFIG':':4096:8'};child=subprocess.Popen(argv,cwd=P,env=env);q=Path('/proc')/str(child.pid);(R/'batch-check-identity.json').write_text(json.dumps(dict(pid=child.pid,pgid=os.getpgid(child.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),argv=argv),indent=2)+'\n');code=child.wait();assert code==0,code
import torch
sys.path.insert(0,str(P/'scripts/design_readiness'));from checkpoint_compare import compare
from audit_panel import pin
reference=torch.load(C/'noop-new02/checkpoint-00005808.pt',map_location='cpu',weights_only=False);refmodel=json.loads((C/'noop-new02/result.json').read_text())['model'];results=json.loads((R/'batch-check01/result.json').read_text());checks=[]
for arm in results['arms']:
 state=torch.load(arm['checkpoint']['path'],map_location='cpu',weights_only=False);fields=compare(reference,state,['contract','cursor','scales','optimizer','rng','cuda_rng','python_rng','local_exposure']);assert all(fields.values()),fields
 actual=pin(arm['model']['path']);assert actual['sha256']==refmodel['sha256'] and actual['bytes']==refmodel['bytes'];checks.append(dict(index=arm['index'],byte_exact_fields=fields,native_export_identical=True))
report=dict(status='SHARED_PROCESS_PARENT_RESET_STATE_AND_EXPORT_BYTE_EXACT',checks=checks,arms=results['arms'],scope='Two identical48-update control arms reset to paid parent independently, compared to previously separate-process control. Wall-time observation is workflow preparation cost, not model throughput or a general architecture benchmark.')
(R/'batch-check-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
