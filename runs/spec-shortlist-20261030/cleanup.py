from pathlib import Path
import json,subprocess,shutil,os,time,fcntl
P=Path(__file__).resolve().parent;R=P.parent.parent
assert P==Path('/home/philip/binary-eagle-decoding/runs/spec-shortlist-20261030')
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
supervisors=[R/'runs'/('spec-shortlist-20261030-'+name) for name in ['initial','final-bench','parity']]
for p in supervisors:
 s=json.loads((p/'state.json').read_text());assert s['status']=='finished' and s['exit_code']==0
 for k in ['pid','supervisor_pid']:assert not Path('/proc',str(s[k])).exists()
def size(p):return int(subprocess.check_output(['du','-B1','-s',str(p)],text=True).split()[0])
def free():
 v={'wsl':shutil.disk_usage(P).free,'windows_via_mount':shutil.disk_usage('/mnt/c').free}
 try:v['windows_native_C']=int(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-Volume -DriveLetter C).SizeRemaining'],text=True,timeout=30).strip())
 except Exception as e:v['windows_native_error']=repr(e)
 return v
inventory=[{'path':str(x),'allocated_bytes':size(x),'reason':reason} for x,reason in [(P/'models','reusable pinned model and conversion-input cache'),(P/'runtime','pinned upstream source and conversion tools'),(P/'build/bin','reusable exact native runtime binaries and shared libraries'),(P/'measurements','request-level scientific evidence')]]
candidates=[x for x in (P/'build').iterdir() if x.name not in ['bin','CMakeCache.txt']]
for p in supervisors:
 candidates.extend(p/x for x in ['tmp','cache','cuda','torch','huggingface'] if (p/x).exists())
candidates.extend((P/'runtime').rglob('__pycache__'))
candidates.extend([P/'__pycache__'] if (P/'__pycache__').exists() else [])
rows=[]
for p in candidates:
 assert not p.is_symlink() and p.stat().st_uid==os.getuid();assert p.resolve().is_relative_to(R/'runs')
 rows.append({'path':str(p),'allocated_bytes':size(p),'category':'rebuildable task scratch; owned processes stopped'})
before=free()
for p in candidates:
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
after=free();receipt={'at':time.time(),'before':before,'after':after,'observed_delta':{k:after[k]-v for k,v in before.items() if isinstance(v,int) and isinstance(after.get(k),int)},'removed':rows,'retained':inventory,'supervisors_verified_finished':True,'gpu_empty':True,'lane_lock_free':True,'windows_compaction_performed':False}
(P/'storage-cleanup.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
subprocess.run([str(P/'build/bin/llama-server'),'--version'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
