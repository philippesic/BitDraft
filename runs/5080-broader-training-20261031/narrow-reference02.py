"""Run after current training quiesces; original historical jobs remain parked."""
from pathlib import Path
import sys,os,subprocess
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031'
W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');S=R/'source02'
sys.path.insert(0,str(S/'scripts'))
from training_system.io import read,pin,authenticate,atomic_json
state=read(W/'supervisors/main07/state.json');assert state['status']=='finished' and state['exit_code']==0
reference=read(R/'historical-narrow32400-reference.json');authenticate(reference['model'])
assert reference['local_updates']==32400 and reference['finish_status']=='EXPOSURE_COMPLETE'
k=read(R/'interop-keeper02.json');assert Path(k['interop']).exists();os.environ['WSL_INTEROP']=k['interop']
qc=read(W/'qualification-config04.json');base=read(W/'base-spec04.json')
cfg={**qc['native'],'model':reference['model'],'manifest':base['manifest']}
cp=W/'narrow32400-native-config01.json';atomic_json(cp,cfg)
g=read(W/'guard04.json');g.update(output=str(W/'narrow32400-native-guard01'),max_train_seconds=0,wall_seconds=1200,deadline_unix=1793510100,progress_paths=[str(W/'narrow32400-native01/native-0000/rows.jsonl')])
g['inputs']=g['inputs']+[pin(R/'controller02/guard.py')]+[pin(cp),cfg['model'],cfg['helper'],pin(R/'historical-narrow32400-reference.json'),pin(R/'diversity-comparison-design01.json')]
gp=W/'narrow32400-native-guard-config01.json';atomic_json(gp,g)
subprocess.run([sys.executable,'-B',str(R/'controller02/guard.py'),'--config',str(gp),'--',sys.executable,'-B',str(S/'scripts/training_system/native.py'),'--config',str(cp),'--output',str(W/'narrow32400-native01'),'--batch-cases','144'],check=True)
result=read(W/'narrow32400-native01/result.json');assert result['metrics']['development']['windows']==192
print('HISTORICAL_NARROW32400_FIXED_PANEL_COMPLETE',flush=True)
