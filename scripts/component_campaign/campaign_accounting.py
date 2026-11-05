"""Account paid physical work separately from exact-forward logical controls."""
import json,os,shutil
from pathlib import Path
from .common import *

def main():
 rows=[]
 for folder in sorted((R/'arms').iterdir()):
  path=folder/'result.json'
  if not path.exists():continue
  data=json.load(open(path));copy_control=data.get('reused_training') or data.get('training_reused') or folder.name in ('H6','E6')
  rows.append(dict(arm=folder.name,updates=data['progress']['updates'],physical_updates=0 if copy_control else data.get('updates_this_process',data['progress']['updates']),actual_update_seconds=0 if copy_control else data['progress'].get('update_seconds',0),process_wall_seconds=data.get('process_wall_seconds'),unique_label_positions=data.get('unique_label_positions'),supervised_positions=data.get('supervised_positions'),full_state_retained=Path(data.get('recovery',{}).get('archive','/nonexistent')).exists(),result=pin(path)))
 # H0 resumed pilot; update_seconds is cumulative, while physical_updates is process-local.
 actual=sum(x['physical_updates'] for x in rows)
 seconds=sum(x['actual_update_seconds'] for x in rows)
 pilot=next((x for x in rows if x['arm']=='H0-pilot'),None)
 if pilot and any(x['arm']=='H0' for x in rows):seconds-=pilot['actual_update_seconds']
 checkpoints=[]
 for folder in (R/'arms').iterdir():
  path=folder/'recovery-cost.jsonl'
  if path.exists():checkpoints += [dict(arm=folder.name,**json.loads(line)) for line in path.read_text().splitlines()]
 readouts=[p for p in (R/'arms').glob('*/readouts.json')]
 measurements=[p for p in R.rglob('result.json') if p.parent.name.startswith(('smoke-','context-zero-')) or p.parent.parent.name=='measurements']
 save(R/'accounting.json',dict(arms=rows,physical_updates_completed=actual,cumulative_actual_update_seconds=seconds,checkpoint_save_seconds=sum(x['seconds'] for x in checkpoints),checkpoint_saves=checkpoints,native_measurement_results=[pin(p) for p in measurements],free_bytes={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')},policy='Physical updates count resumed pilot once, copied exactQ6 controls zero. Total wall includes setup/native/checkpoint separately; cumulative step time does not claim campaign elapsed time. All4096 labels are repeated exposure to1762unique positions until another admitted recipe is explicitly labeled.'))
 print('ACCOUNTED',actual,'physical updates',seconds,'update seconds',flush=True)

if __name__=='__main__':main()
