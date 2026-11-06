"""Keep exactly selected NEW full frontier states after authenticated reproduction."""
import argparse,json,sys,subprocess
from .common import *
from .archive import reference

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm');ap.add_argument('--ordinal',type=int,choices=[1,2],required=True);a=ap.parse_args()
 source=R/'arms'/a.arm;result=json.load(open(source/'result.json'))
 updates=result['progress']['updates']
 if result['status']!='COMPLETE' or updates<4096:raise ValueError('frontier must be fully trained')
 outname=a.arm+'-confirm';out=R/'arms'/outname
 directory=Path('/mnt/c/binary-eagle-component-20261105-01');pins=directory/'frontier-retained.json';retained=json.load(open(pins)) if pins.exists() else {}
 if len(retained)>=a.ordinal:
  if not any(x.get('original_arm')==a.arm for x in retained.values()):raise ValueError('frontier retention already has a different selection')
  return
 if len(retained)!=a.ordinal-1:raise ValueError('retain frontiers in declared order')
 if not (out/'result.json').exists():
  # The second state cannot create a third full rolling checkpoint in flight.
  argv=[sys.executable,'-m','scripts.component_campaign.train',str(source/'spec.json'),'--updates',str(updates),'--out-name',outname]
  if a.ordinal==2:argv+=['--checkpoint-points']
  subprocess.run(argv,cwd=ROOT,check=True)
 confirmed=json.load(open(out/'result.json'));old=reference(source/f'model-{updates}.gguf');new=reference(out/f'model-{updates}.gguf')
 if old['sha256']!=new['sha256']:raise ValueError('replayed same-history frontier differs; investigate before treating it as reproduction')
 recovery=json.load(open(out/'recovery.json'))
 if pin(recovery['archive'])['sha256']!=recovery['archive_sha256']:raise ValueError('frontier full recovery state changed')
 retained[recovery['archive']]=dict(**recovery,original_arm=a.arm,updates=updates,confirmed_result=pin(out/'result.json'),native=old,same_history_reproduction=True)
 save(pins,retained);save(R/'frontier-retention.json',dict(states=retained,source=pin(__file__),policy='Original paid B0 protected. Two exact full additional4096states, with masters/Adam/RNG/cursor. Same-history reproduction is not an independent training seed. Native readouts retained separately.'))
 print('FRONTIER_RETAINED',a.arm,flush=True)

if __name__=='__main__':main()
