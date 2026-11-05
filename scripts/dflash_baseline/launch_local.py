"""Launch one supervised local lane task and write birth/source/stop receipts."""
import argparse,hashlib,json,shlex,subprocess,time,os
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');ROOT=Path(__file__).resolve().parents[2]
def launch(run,command):
 snapshot=P/'runs/precision-campaign-20261104-01/source-snapshots'/run;snapshot.mkdir(parents=True,exist_ok=False)
 files=[]
 for directory in ('scripts/dflash_baseline','src/w1a1_eagle','scripts/w1a8_generalization','scripts/contender_pipeline','scripts/data_cost_transfer'):
  files.extend((p,p.relative_to(ROOT)) for p in (ROOT/directory).glob('*') if p.is_file() and p.suffix in ('.py','.cpp','.cu','.cuh'))
 files.append((ROOT/'experiments/w1a8-standing-20261102/baseline.json',Path('experiments/w1a8-standing-20261102/baseline.json')))
 gguf=P/'third_party/llama.cpp/gguf-py/gguf';files.extend((p,Path('third_party/llama.cpp/gguf-py/gguf')/p.name) for p in gguf.glob('*.py'));pins={}
 for source,relative in files:
  before=source.stat();data=source.read_bytes();after=source.stat()
  if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('source changed during snapshot')
  target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);target.chmod(0o444);pins[str(target)]=hashlib.sha256(data).hexdigest()
 selection=dict(original_root=str(ROOT),source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT),'diff','--binary'])).hexdigest(),files=pins,scope='Immutable owned source snapshot, not a Git/device checkout; canonical weights/data/runtime remain in place.')
 (snapshot/'source-freeze.json').write_text(json.dumps(selection,indent=2)+'\n')
 command=[str(snapshot/Path(x).relative_to(ROOT)) if str(x).startswith(str(ROOT)+'/') else str(x) for x in command]
 argv=[str(P/'.venv/bin/python'),'scripts/remote_job.py',run,'--stop-grace-seconds','120','--',*map(str,command)]
 subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s',run,'-c',str(P),'exec '+shlex.join(argv)],check=True)
 path=P/'runs'/run/'state.json'
 for _ in range(100):
  try:
   state=json.loads(path.read_text())
   if 'pid' in state:break
  except (OSError,ValueError):pass
  time.sleep(.1)
 else:raise RuntimeError('supervisor did not publish running identity')
 receipt=dict(source_snapshot=str(snapshot),source_freeze=selection,supervisor_source_sha256=hashlib.sha256((P/'scripts/remote_job.py').read_bytes()).hexdigest(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=state,processes={},source={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},stop='Verify boot/starttime/cmdline against this receipt before signaling. TERM supervisor, allow120seconds for update-boundary save, verify state terminal and process group absent. Preserve artifacts.')
 for key in ('supervisor_pid','pid'):
  proc=Path('/proc')/str(state[key]);receipt['processes'][key]=dict(pid=state[key],stat=(proc/'stat').read_text(),cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
 (path.parent/'ownership.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args();launch(a.run,a.command)
