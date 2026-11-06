"""Authenticate post-reboot campaign state without rewriting stale job receipts."""
import hashlib,json,os,subprocess,time
from .common import *
from .archive import blocks

def main():
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 destination=R/'restart-reconciliation'/str(time.time_ns());destination.mkdir(parents=True)
 pause();prior={}
 for name in ('component-recovery-combinations-20261105-02','component-post-recovery-20261105-02'):
  path=ROOT/'runs'/name/'state.json';state=json.load(open(path));prior[name]=dict(receipt=pin(path),state=state)
 old_owner=json.load(open(R/'recovery-combinations-owner.json'))
 if old_owner['owner']['boot_id'].strip()==boot:raise ValueError('expected a different predecessor boot')
 apps=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
 if apps.strip():raise RuntimeError('GPU already has compute owner: '+apps)
 # Authenticate each immutable dependency once, then every reconstructed stream.
 checked={};rows=[]
 for receipt in sorted((R/'arms').glob('*/model-*.gguf.slices.json')):
  record=json.load(open(receipt))
  for ref in [record['archive'],*record['dependencies']]:
   if ref['path'] not in checked:checked[ref['path']]=pin(ref['path'])
   if checked[ref['path']]!=ref:raise ValueError('archive dependency differs '+ref['path'])
  h=hashlib.sha256();size=0
  for block in blocks(record):h.update(block);size+=len(block)
  if h.hexdigest()!=record['original']['sha256'] or size!=record['original']['bytes']:raise ValueError('reconstructed native differs '+str(receipt))
  if record.get('repacked_from'):
   oldref=record['repacked_from']['receipt']
   if pin(oldref['path'])!=oldref:raise ValueError('saved repack receipt differs')
   old=json.load(open(oldref['path']))
   if old['original']!=record['original']:raise ValueError('repack changed logical native identity')
  rows.append(dict(receipt=pin(receipt),original=record['original'],storage_version=record.get('storage_version',0)))
  print('ARCHIVE_VERIFIED',receipt.parent.name,receipt.name,flush=True)
 # Csmall never reached its zero readout or a single optimizer update. Preserve
 # the complete failed setup under a relocation receipt before a fresh attempt.
 incomplete=R/'arms/Csmall';relocation=None
 if incomplete.exists() and not (incomplete/'result.json').exists():
  owner=json.load(open(incomplete/'ownership.json'))
  if owner['boot_id'].strip()==boot or any((incomplete/n).exists() for n in ('updates.jsonl','readouts.json','recovery.json','live.json')):raise ValueError('Csmall has paid progress: explicit resume required')
  files={str(p.relative_to(incomplete)):pin(p) for p in incomplete.rglob('*') if p.is_file()}
  target=destination/'Csmall-interrupted-setup';incomplete.rename(target)
  relocation=dict(original=str(incomplete),preserved=str(target),files=files,reason='Old-boot setup only; no optimizer updates or checkpoint/readout receipt. All setup files preserved.')
 completed=[]
 for p in sorted((R/'arms').glob('*/result.json')):
  d=json.load(open(p))
  if d['status']=='COMPLETE' and d['progress']['updates']==4096:completed.append(dict(arm=p.parent.name,result=pin(p)))
 receipt=dict(status='RESTART_AUTHENTICATED_SERIAL_RESUME_ALLOWED',owner=identity(),previous_owner=old_owner,predecessors=prior,archives=rows,dependencies_verified=len(checked),completed=completed,incomplete_setup_relocation=relocation,partial_files=[str(p) for p in R.rglob('*') if p.is_file() and ('.partial' in p.name or p.name.endswith('.hydrating'))],hardware=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),source=pin(__file__),policy='Old running receipts preserved verbatim; old boot is terminal by reboot, never clean success. Complete native streams and repack original identities verified. No paid training rerun or prior artifact deletion.')
 save(destination/'verified.json',receipt);save(R/'restart-reconciliation.json',receipt)
 print('RESTART_AUTHENTICATED',destination,flush=True)

if __name__=='__main__':main()
