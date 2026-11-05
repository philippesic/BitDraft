"""Bounded cleanup of superseded owned build caches, never source/models/evidence."""
from pathlib import Path
import json,os,hashlib,shutil,subprocess,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/execution-campaign-20261105-01'
assert json.load(open(R/'release.json'))['all_owned_groups_absent']
def windows():return int(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-Volume -DriveLetter C).SizeRemaining'],text=True).strip())
def space():return dict(wsl_free_bytes=shutil.disk_usage(P).free,windows_native_C_free_bytes=windows(),windows_mounted_C_free_bytes=shutil.disk_usage('/mnt/c').free)
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
names=['runtime-silu','runtime-residual','runtime-fused','runtime-silu02','runtime-residual02','runtime-narrow','runtime-head','runtime-silu03','runtime-generic','runtime-combined','runtime-combined-head','runtime-combined-last','runtime-silu-debug','runtime-lut']
selected=[]
for name in names:
 d=R/name
 assert d.is_dir() and not d.is_symlink() and (d/'build.json').exists()
 if (d/'bin').exists():selected.extend((d/'bin').iterdir())
 selected.extend(f for f in d.iterdir() if f.is_file() and not f.is_symlink() and (f.name.endswith('.o') or f.name.endswith('.o.d')))
# A duplicate output blob is recoverable byte-for-byte from the retained original.
a=R/'joint-overflow-checks/original-maxclip.bin';b=R/'joint-overflow-checks/optimized-maxclip.bin';assert pin(a)['sha256']==pin(b)['sha256'];duplicate=dict(original=pin(a),duplicate=pin(b));selected.append(b)
selected.extend(p for p in (R/'post-checks').glob('*.sass') if p.is_file() and not p.is_symlink())
# Recheck no recorded group has become live before touching caches.
for state in (P/'runs').glob('execution-*-20261105-*/state.json'):
 d=json.load(open(state))
 try:os.killpg(d['pgid'],0)
 except ProcessLookupError:continue
 raise RuntimeError('recorded group exists; reauthenticate before cleanup')
before=space();removed=[]
for p in selected:
 assert R in p.parents and not p.is_dir()
 if not p.exists() and not p.is_symlink():continue
 removed.append(dict(path=str(p),kind='owned symlink only' if p.is_symlink() else 'reproducible cache / redundant scratch',bytes=0 if p.is_symlink() else p.stat().st_size))
 p.unlink()
after=space();inventory=[]
for d in sorted(R.iterdir()):
 if d.is_symlink():continue
 inventory.append(dict(path=str(d),bytes=size(d) if d.is_dir() else d.stat().st_size,category='evidence/source/protected recovery receipts' if not d.name.startswith('runtime-') else 'qualified runtime/reproducible cache and immutable source',owner='execution campaign',retention='preserve through parent review/integration; sources and measurements retained even where superseded binaries were pruned'))
record=dict(before=before,after=after,measured_delta={k:after[k]-before[k] for k in before},logical_removed_bytes=sum(x['bytes'] for x in removed),removed=removed,duplicate_proof=duplicate,retained=inventory,protected_external=['original precision checkpoints and GGUF archives','shared raw GGUF dictionary basis','frozen runtime04/base objects/models','unrelated canonical checkout files/settings/shared caches'],interpretation='WSL free and native Windows free measured separately. Logical deletion is not a Windows/VHDX reclaim estimate. No compaction or protected-data deletion.',cleanup_time_unix_ns=time.time_ns());(R/'retained-storage.json').write_text(json.dumps(record,indent=2)+'\n');print({k:record[k] for k in ['before','after','measured_delta','logical_removed_bytes']})
