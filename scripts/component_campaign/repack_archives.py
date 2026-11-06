"""Losslessly repack only completed NEW campaign snapshots; preserve original bits."""
import argparse,copy,hashlib,json,os,shutil,time
from .common import *
from .archive import paths,hydrate,release,archive,authenticate,reference,blocks

def repack(p):
 _,receipt=paths(p);old=json.load(open(receipt));authenticate(old)
 if old.get('storage','').startswith('Content-addressed'):return None
 if old['original']['path']!=str(p.resolve()) or p.parent.parent!=R/'arms':raise ValueError('only NEW owned arm snapshots can be repacked')
 state=json.load(open(p.parent/'result.json'))
 if state['status']!='COMPLETE':raise ValueError('do not repack an active trainer')
 directory=R/'archive-repacked'/p.parent.name/(p.stem+'-'+str(time.time_ns()));directory.mkdir(parents=True,exist_ok=False)
 save(directory/'prior-receipt.json',old);created=hydrate(p);before=pin(p);staged=directory/'model.gguf'
 try:
  staged.hardlink_to(p);new=archive(staged);replacement=copy.deepcopy(new);replacement['original']=old['original']
  h=hashlib.sha256();size=0
  for part in blocks(replacement):h.update(part);size+=len(part)
  if h.hexdigest()!=before['sha256'] or size!=before['bytes']:raise ValueError('repacked actual native byte stream differs')
  if json.load(open(receipt))!=old or pin(p)!=before:raise ValueError('snapshot changed before repack publication')
  # The new immutable payloads exist and passed a whole-stream digest before
  # the single atomic metadata switch. Prior payload is kept until revalidation.
  replacement['repacked_from']=dict(receipt=pin(directory/'prior-receipt.json'),prior_container=old['archive'],scope='NEW campaign storage only; original GGUF bytes unchanged')
  save(receipt,replacement);authenticate(replacement)
  if reference(p)!=old['original']:raise ValueError('repack logical-model reference changed')
  oldfile=Path(old['archive']['path'])
  if oldfile.parent!=p.parent or oldfile.is_symlink() or pin(oldfile)!=old['archive']:raise ValueError('unexpected prior payload ownership')
  oldfile.unlink()
  result=dict(status='EXACT_NATIVE_BYTES_REPACK_PASS',original=before,receipt=pin(receipt),prior=old['archive'],new_container=replacement['archive'],coded_blob_bytes=sum(x['bytes'] for x in replacement['dependencies'] if '/archive-blobs/' in x['path']),prior_container_retired=True,source=pin(__file__),helper=pin(ROOT/'scripts/component_campaign/archive.py'))
  save(directory/'verified.json',result);print('REPACKED',p.parent.name,p.name,old['archive']['bytes'],result['coded_blob_bytes'],flush=True);return result
 finally:release(p,created)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arms',nargs='*');a=ap.parse_args();before={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')};rows=[]
 directories=[R/'arms'/n for n in a.arms] if a.arms else sorted((R/'arms').iterdir())
 for directory in directories:
  if not (directory/'result.json').exists() or json.load(open(directory/'result.json'))['status']!='COMPLETE':continue
  for receipt in sorted(directory.glob('model-*.gguf.slices.json')):
   pause();p=receipt.with_suffix('').with_suffix('');row=repack(p)
   if row:rows.append(row)
 out=R/'archive-repack-results';out.mkdir(exist_ok=True);path=out/(str(time.time_ns())+'.json');save(path,dict(status='COMPLETED_NEW_NATIVE_ARCHIVES_REPACKED',rows=rows,free_before=before,free_after={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')},owner=identity(),policy='Every original GGUF digest verified before atomic receipt replacement; only old NEW compressed containers retired. Raw bases, original paid sources/checkpoints, active trainers and every native state retained. Shared blobs immutable. Free-space deltas can include concurrent checkpoint writes.'))
 print('REPACK_BATCH_COMPLETE',len(rows),path,flush=True)

if __name__=='__main__':main()
