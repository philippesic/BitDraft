"""Reconcile update accounting and distinct training exposure without model loads."""
import json
import math
import time
from .common import *


def audit():
    path=OUT/'updates.jsonl'
    windows=set();labels=set();sequences=set();times=[];last=None
    if path.exists():
        with path.open() as stream:
            for index,line in enumerate(stream,1):
                try: row=json.loads(line)
                except json.JSONDecodeError:
                    assert not stream.read(), 'Malformed interior update row'
                    break  # A concurrent writer may expose an incomplete last row.
                assert row['update']==index
                assert all(math.isfinite(row[k]) for k in ('loss','gradient_norm','seconds','cumulative_seconds'))
                assert row['seconds']>0
                times.append(row['seconds']);last=row
                key=(row['sequence'],row['anchor']);windows.add(key);sequences.add(row['sequence'])
                labels.update((row['sequence'],row['anchor']+j) for j in range(15))
    seconds=math.fsum(times)
    if last: assert abs(seconds-last['cumulative_seconds'])<1e-5
    result_path=OUT/'result.json'
    checkpointed=json.load(open(result_path)).get('updates',0) if result_path.exists() else 0
    result=dict(status='PASS',checked_unix=time.time(),completed_updates=len(times),
        actual_update_seconds=seconds,checkpointed_updates=checkpointed,
        uncheckpointed_completed_updates=max(0,len(times)-checkpointed),
        training_wall_seconds=last['training_seconds'] if last else 0,
        distinct_sequences=len(sequences),distinct_windows=len(windows),distinct_label_positions=len(labels),
        supervised_positions=15*len(times),nominal_window_epochs=len(times)/1568435,
        total_training_sequences=15647,total_training_windows=1568435,
        note='Distinct position identities within sequences; overlapping windows and repeated epochs are exposure, not fresh labels. Rejected rollback exposures remain separate in recovery-incidents.')
    save(REPORT/'audit.json',result)
    return result


if __name__=='__main__': print(json.dumps(audit(),indent=2))
