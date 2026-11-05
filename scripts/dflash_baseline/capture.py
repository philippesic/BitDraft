"""Recapture existing TRAIN/consumed DEV sequences with all DFlash taps."""
from pathlib import Path
import json,os,subprocess,hashlib
P=Path('/home/philip/binary-eagle-decoding');OLD=P/'runs/w1a8-modern-generalization-capture-20261103-02';R=P/'runs/dflash-modern-capture-20261104-04/data'
R.mkdir(exist_ok=False)
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def pin(p):
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=d,bytes=p.stat().st_size)
records=[]
for name in [f'fit-{domain}-{i:02d}' for domain in ['code','prose','reasoning'] for i in range(8)]+[f'development-{domain}-00' for domain in ['code','prose','reasoning']]:
 assert not json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']
 old=OLD/name; original=json.load(open(old/'capture-command.json'));meta=json.load(open(old/'capture/capture.json'))
 cmd=original['argv'].copy();cell=R/name;cell.mkdir()
 cmd[cmd.index('--out')+1]=str(cell/'capture')
 cmd[cmd.index('--taps')+1]='2,6,10,14,18,22,26,30'
 positions=list(range(meta['prompt_tokens']-1,len(meta['input_tokens'])))
 cmd[cmd.index('--logit-positions')+1]=','.join(map(str,positions))
 cmd[cmd.index('--output-cap')+1]=str(len(positions))
 cmd[cmd.index('--max-output-mib')+1]='512'
 save(cell/'command.json',dict(argv=cmd,environment=original['environment'],ancestry=pin(old/'capture-command.json'),split='TRAIN' if name.startswith('fit') else 'consumed_DEV'))
 with (cell/'capture.log').open('w') as log:subprocess.run(cmd,env={**os.environ,**original['environment']},stdout=log,stderr=subprocess.STDOUT,check=True)
 new=json.load(open(cell/'capture/capture.json'))
 assert new['input_tokens']==meta['input_tokens'] and new['taps']==[2,6,10,14,18,22,26,30]
 labels=dict(zip(new['logit_positions'],new['greedy_ids']))
 assert all(labels[p]==v for p,v in zip(meta['logit_positions'],meta['greedy_ids']))
 records.append(dict(name=name,metadata=pin(cell/'capture/capture.json'),split='TRAIN' if name.startswith('fit') else 'consumed_DEV',files=[pin(p) for p in (cell/'capture').glob('*.f32')]))
 save(R/'progress.json',records);print('Captured',name,flush=True)
save(R/'result.json',dict(records=records,status='CAPTURED',taps=[2,6,10,14,18,22,26,30]))
