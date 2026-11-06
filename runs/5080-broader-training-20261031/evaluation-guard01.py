"""Fresh finite pilot custody, input pins, resource checks and child supervision."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, math, os, pathlib, shutil, signal, subprocess, sys, time

def sha(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def identity(p):
    s=pathlib.Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def validate_resource_caps(c):
    if sys.flags.optimize:raise ValueError('original assertion guards require an unoptimized interpreter')
    if not (math.isfinite(c['max_train_seconds']) and 0<=c['max_train_seconds']<900 and math.isfinite(c['wall_seconds']) and c['wall_seconds']>0):raise ValueError('finite bounded train/wall caps required')
    if c.get('deadline_unix') is not None and not math.isfinite(c['deadline_unix']):raise ValueError('finite absolute custody deadline required')

def observe_progress(paths, previous):
    """Keep the last observed progress across an atomic publication gap.

    DrvFS can briefly report a replaced file absent. Forgetting an observation
    at that instant falsely declares an actively updating child stalled.
    Missing files do not advance time; a genuinely stalled child still expires
    at the original threshold. Other metadata errors remain failures.
    """
    observed=previous
    for path in paths:
        try:stamp=pathlib.Path(path).stat().st_mtime
        except FileNotFoundError:continue
        observed=max(observed,stamp)
    return observed

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    c=json.loads(pathlib.Path(a.config).read_text());cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    assert cmd and c['authority'] in ('user_full_send_shortlist_pilots_20261026', 'user_overnight_quality_20261027', 'user_training_system_20261027','user_unattended_training_preparation_20261028', 'user_5080_broader_training_20261031')
    validate_resource_caps(c)
    check_pause=lambda:None
    out=pathlib.Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    root=pathlib.Path(c.get('project_root','/home/philip/binary-eagle-decoding'));controls=pathlib.Path.home()/'.config/binary-eagle-decoding'
    worker=None
    if c.get('device_worker'):
        # Optional dispatch uses the same guard, with an authenticated pool and
        # per-adapter proof. It cannot turn a paused worker on.
        from training_system.workers import validate_pool,eligibility
        from training_system.io import authenticate,read
        binding=c['device_worker'];authenticate(binding['pool']);pool=validate_pool(read(binding['pool']['path']))
        worker=next(w for w in pool['workers'] if w['id']==binding['worker_id'])
        if binding.get('qualification_trial'):
            from training_system.device_proofs import validate_trial
            trial=validate_trial(pool,worker,binding['qualification_trial'],binding['adapter_state_key'],binding['precision'])
            assert c.get('deadline_unix') is not None and c['deadline_unix']<=trial['deadline_unix'],'trial guard deadline exceeds declaration'
            assert 0<=c['max_train_seconds']<=60,'trial TRAIN ceiling exceeds declaration'
        else:assert eligibility(worker,binding['adapter_state_key'],binding['precision'],worker.get('qualification'))['eligible'],'worker not qualified'
        assert worker['host_machine_id']==pathlib.Path('/etc/machine-id').read_text().strip(),'wrong worker host'
        assert worker['gpu_uuid'] in subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split(),'wrong worker physical GPU'
    def check():
        check_pause()
        if c.get('deadline_unix') is not None and time.time()>=c['deadline_unix']:raise TimeoutError('finite absolute GPU custody deadline')
        v=json.loads((controls/'gpu-control.json').read_text());kind=worker['kind'] if worker else 'rtx5080'
        assert v[kind]['pause_requested'] is False and v[kind].get('stop_all_requested',False) is False,'worker paused or stopped'
        if worker is None:assert v['rtx2080ti']['pause_requested'] is True
        assert not any(pathlib.Path(x).exists() for x in c['stop_paths'])
        assert shutil.disk_usage(root).free>=c['minimum_wsl_free_bytes']
        assert shutil.disk_usage(c.get('host_storage_path','/mnt/c')).free>=c['minimum_windows_free_bytes']
        mem={l.split(':')[0]:int(l.split()[1])*1024 for l in pathlib.Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable']>1024**3
    started=time.time();check()
    owner={'pid':os.getpid(),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'authority':c['authority'],'output':str(out)}
    lockpath=controls/('gpu-'+worker['gpu_uuid']+'.lock' if worker and worker['kind']!='rtx5080' else 'rtx5080-campaign.lock')
    lock=lockpath.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);prior=lock.read();lock.seek(0);lock.truncate();json.dump(owner,lock);lock.flush();os.fsync(lock.fileno())
    (out/'previous-lock.txt').write_text(prior)
    def contexts():
        command=['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']
        if worker:command+=['--id='+worker['gpu_uuid']]
        return subprocess.check_output(command,text=True,timeout=10).strip()
    assert contexts()=='','foreign GPU context before launch'
    cache_enabled=c.get('authenticated_cache',False)
    assert not cache_enabled or c['authority'] in ('user_training_system_20261027','user_unattended_training_preparation_20261028', 'user_5080_broader_training_20261031')
    authenticated={}
    def checked_hash(ref):
        name=str(pathlib.Path(ref['path']).resolve());before=identity(name)
        if cache_enabled and name in authenticated:
            entry=authenticated[name];assert entry['identity']==before,'input identity changed';value=entry['sha256']
        else:
            value=sha(name);assert identity(name)==before,'input changed while hashing'
            authenticated[name]={'sha256':value,'identity':before}
        assert value==ref['sha256'],'input pin mismatch: '+name
        return value
    refs=c['inputs'];assert all(checked_hash(x)==x['sha256'] for x in refs),'input pin mismatch'
    paid={x['path']:checked_hash(x) for x in c['paid_barriers']}
    for x in c['paid_barriers']:
        if x['path'].endswith('latest.json'):assert pathlib.Path(x['path']).stat().st_size<20*1024**2
    env=os.environ.copy();env.update(c.get('env',{}));env.update(CUDA_VISIBLE_DEVICES=worker['gpu_uuid'] if worker else '0',PYTHONDONTWRITEBYTECODE='1')
    env.pop('TRAINING_LANE_AUTH_CACHE',None);env.pop('TRAINING_LANE_AUTH_CACHE_SHA256',None)
    if cache_enabled:
        manifest=out/'authenticated-inputs.json'
        manifest.write_text(json.dumps({'schema':'guard_authenticated_inputs_v1','owner':{k:owner[k] for k in ('pid','boot_id','start_ticks')},'files':authenticated},sort_keys=True))
        env.update(TRAINING_LANE_AUTH_CACHE=str(manifest),TRAINING_LANE_AUTH_CACHE_SHA256=sha(manifest))
    (out/'entry.json').write_text(json.dumps({'owner':owner,'config':c,'command':cmd,'input_pins_checked':True,'started':started},indent=2))
    child=None;rc=None;error=None;last_progress=started
    def interrupted(signum, frame):
        raise KeyboardInterrupt('guard stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        check() # hashing may consume time; never start a child after expiry
        with (out/'child.log').open('wb') as log:
            child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                check()
                if time.time()-started>c['wall_seconds']:raise TimeoutError('pilot wall cap')
                # Process liveness alone does not establish forward progress.
                # Optional tracked files include trainer progress/checkpoint and
                # native helper output. Startup gets its own explicit allowance.
                observed=observe_progress(c.get('progress_paths',[]),last_progress)
                last_progress=observed
                if c.get('stall_seconds') and time.time()-max(observed,started+c.get('startup_seconds',120))>c['stall_seconds']:
                    raise TimeoutError('owned child progress stalled')
                heartbeat={'owner':owner,'child_pid':child.pid,'observed_unix':time.time(),'last_progress_unix':observed}
                temp=out/'.heartbeat.tmp';temp.write_text(json.dumps(heartbeat));os.replace(temp,out/'heartbeat.json')
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
        preserved=(all(identity(p)==entry['identity'] for p,entry in authenticated.items()) if cache_enabled else all(sha(p)==h for p,h in paid.items()))
        result={'returncode':rc,'error':error,'wall_seconds':time.time()-started,'paid_state_preserved':preserved,'integrity_method':'entry_SHA256_and_exact_file_identity_boundaries' if cache_enabled else 'fresh_SHA256_at_entry_and_release','child_exited':child is not None and child.poll() is not None,'contexts_after':contexts()}
        (out/'result.json').write_text(json.dumps(result,indent=2));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert rc==0 and preserved
    return 0

if __name__=='__main__':sys.exit(main())
