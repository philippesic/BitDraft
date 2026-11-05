import hashlib,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
P=Path('/home/philip/binary-eagle-decoding')
R=P/'runs/component-campaign-20261105-01'
B=P/'runs/precision-campaign-20261104-01/finalists/G4D8-learned-small'
OPT=P/'runs/execution-campaign-20261105-01/runtime-combined-last02'
TARGET=P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'
Q4=P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'

def pin(path):
 p=Path(path)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p.resolve()),sha256=h,bytes=p.stat().st_size)

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)

def pause():
 if json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']:
  raise RuntimeError('GPU pause requested')

def identity():
 return dict(pid=os.getpid(),stat=Path('/proc/self/stat').read_text(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text(),pgid=os.getpgrp())
