"""Run only after phase2-selection02.json freezes fresh and previous native choices."""
from pathlib import Path
from types import SimpleNamespace
import json,os,sys,subprocess,importlib.util,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');S=R/'source03';E=R/'request-evaluator01.py'
sys.path[:0]=[str(S/'scripts'),str(P/'scripts')]
from training_system.io import read,pin,authenticate,atomic_json
k=read(R/'interop-keeper02.json');assert Path(k['interop']).exists();os.environ['WSL_INTEROP']=k['interop']
assert read(W/'supervisors/fresh-qualify01/state.json')['status']=='finished'
assert read(W/'supervisors/fresh-training01/state.json')['status']=='finished'
assert read(W/'supervisors/fresh-training01/state.json')['exit_code']==0
os.environ['CUDA_CACHE_PATH']=str(W/'fresh-qualification01/cuda')
selection=read(R/'phase2-selection02.json');assert selection['selection_basis']=='frozen native development; no request-based checkpoint choice';assert selection['candidates']
for name,entry in selection['candidates'].items():
 authenticate(entry['model']);native=read(authenticate(entry['native_result']));assert native['config']['model']==entry['model'] and native['config']['manifest']==selection['manifest']
modspec=importlib.util.spec_from_file_location('owned_request_evaluator',E);module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
config=W/'request-final-config02.json'
module.prepare(SimpleNamespace(panel=None,candidate=[name+'='+entry['model']['path'] for name,entry in selection['candidates'].items()],arm_specs=None,build=P/'runs/shortlist-throughput-20261027/native/ready.json',mode='claim',prompt_limit=24,wall_seconds=1800,balanced_screen=False,output=config))
c=read(config);c['panels']={k:v for k,v in c['panels'].items() if k=='development'};assert len(c['panels'])==1;c['environment']['NVIDIA_TF32_OVERRIDE']='0';c['models']['target_only']={**c['models']['q4_k_m'],'target_only':True};c['port']=18297;assert len(c['models'])==6;atomic_json(config,c)
module.run(SimpleNamespace(config=config,output=W/'request-final02',stop=W/'STOP',preflight=True,diagnostic=False,allow_native=False))
g=read(W/'guard04.json');g['env']={**g['env'],'WSL_INTEROP':k['interop'],'CUDA_CACHE_PATH':str(W/'fresh-qualification01/cuda')};g.update(output=str(W/'request-final-guard02'),max_train_seconds=0,wall_seconds=2400,deadline_unix=1793510100,startup_seconds=300,stall_seconds=300,progress_paths=[str(W/'request-final02/requests.jsonl')]);g['inputs']=g['inputs']+[pin(R/'controller02/guard.py')]+[pin(config),pin(R/'phase2-selection02.json'),pin(E),pin(__file__),c['build'],c['binary'],c['target'],*c['runtime_libraries'],*c['source_files'],*c['models'].values()];gp=W/'request-final-guard-config02.json';atomic_json(gp,g)
command=[sys.executable,'-B',str(R/'controller02/guard.py'),'--config',str(gp),'--',sys.executable,'-B',str(E),'run','--config',str(config),'--output',str(W/'request-final02'),'--stop',str(W/'STOP'),'--allow-native']
print('PHASE2_REQUEST_CONFIG_VALIDATED; native-selected models, matchedQ4/parent/target-only,120requests each',flush=True);subprocess.run(command,check=True)
d=read(W/'request-final02/summary.json');assert d['diagnostic'] is False;cells=d['panels']['development']['cells'];assert all(x['requests']==120 for x in cells.values());assert all(x['tokens']>0 for x in cells.values());print(json.dumps({'status':'PHASE2_MATCHED_REQUESTS_COMPLETE','cells':{k:{n:v[n] for n in ['requests','tokens','full_request_seconds','tokens_per_s','accepted','proposed','rounds']} for k,v in cells.items()}}),flush=True)
