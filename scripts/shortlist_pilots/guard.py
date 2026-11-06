"""Fresh finite pilot custody, input pins, resource checks and child supervision."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, pathlib, shutil, signal, subprocess, sys, time

def sha(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    c=json.loads(pathlib.Path(a.config).read_text());cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    assert cmd and c['authority']=='user_full_send_shortlist_pilots_20261026'
    assert c['max_train_seconds']<900 and c['wall_seconds']>0
    out=pathlib.Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    root=pathlib.Path('/home/philip/binary-eagle-decoding');controls=pathlib.Path.home()/'.config/binary-eagle-decoding'
    def check():
        v=json.loads((controls/'gpu-control.json').read_text());assert v['rtx5080']['pause_requested'] is False and v['rtx5080'].get('stop_all_requested',False) is False
        assert v['rtx2080ti']['pause_requested'] is True
        assert not any(pathlib.Path(x).exists() for x in c['stop_paths'])
        assert shutil.disk_usage(root).free>=c['minimum_wsl_free_bytes']
        assert shutil.disk_usage('/mnt/c').free>=c['minimum_windows_free_bytes']
        mem={l.split(':')[0]:int(l.split()[1])*1024 for l in pathlib.Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable']>1024**3
    started=time.time();check()
    owner={'pid':os.getpid(),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'authority':c['authority'],'output':str(out)}
    lock=(controls/'rtx5080-campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);prior=lock.read();lock.seek(0);lock.truncate();json.dump(owner,lock);lock.flush();os.fsync(lock.fileno())
    (out/'previous-lock.txt').write_text(prior)
    def contexts():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
    assert contexts()=='','foreign GPU context before launch'
    refs=c['inputs'];assert all(sha(x['path'])==x['sha256'] for x in refs),'input pin mismatch'
    paid={x['path']:sha(x['path']) for x in c['paid_barriers']}
    assert all(paid[x['path']]==x['sha256'] for x in c['paid_barriers'])
    for x in c['paid_barriers']:
        if x['path'].endswith('latest.json'):assert pathlib.Path(x['path']).stat().st_size<20*1024**2
    env=os.environ.copy();env.update(c.get('env',{}));env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
    (out/'entry.json').write_text(json.dumps({'owner':owner,'config':c,'command':cmd,'input_pins_checked':True,'started':started},indent=2))
    child=None;rc=None;error=None
    try:
        with (out/'child.log').open('wb') as log:
            child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                check()
                if time.time()-started>c['wall_seconds']:raise TimeoutError('pilot wall cap')
                time.sleep(.5)
            rc=child.returncode
    except BaseException as exc:
        error=repr(exc)
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        raise
    finally:
        preserved=all(sha(p)==h for p,h in paid.items())
        result={'returncode':rc,'error':error,'wall_seconds':time.time()-started,'paid_state_preserved':preserved,'child_exited':child is not None and child.poll() is not None,'contexts_after':contexts()}
        (out/'result.json').write_text(json.dumps(result,indent=2));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert rc==0 and preserved
    return 0

if __name__=='__main__':sys.exit(main())
