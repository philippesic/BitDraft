"""Launch one supervised local lane task and write birth/source/stop receipts."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding')
def launch(run,command):
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
 receipt=dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=state,processes={},source={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},stop='Verify boot/starttime/cmdline against this receipt before signaling. TERM supervisor, allow120seconds for update-boundary save, verify state terminal and process group absent. Preserve artifacts.')
 for key in ('supervisor_pid','pid'):
  proc=Path('/proc')/str(state[key]);receipt['processes'][key]=dict(pid=state[key],stat=(proc/'stat').read_text(),cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
 (path.parent/'ownership.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args();launch(a.run,a.command)
