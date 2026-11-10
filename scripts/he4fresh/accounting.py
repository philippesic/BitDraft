"""Recover rejected exposure/time even after repeated rollback to one checkpoint."""
import json
import math
import os
import time

from .common import OUT, R, save


def reconcile_log(progress, path=None, incidents=None):
    path = OUT/'updates.jsonl' if path is None else path
    incidents = R/'recovery-incidents' if incidents is None else incidents
    previous = [json.loads(p.read_text()) for p in incidents.glob('*/accounting.json')]
    rejected_wall = math.fsum(max(0., row['training_seconds']) for row in previous)
    progress['training_seconds'] += max(0., rejected_wall - progress.get('discarded_training_seconds', 0.))
    progress['discarded_training_seconds'] = max(rejected_wall, progress.get('discarded_training_seconds', 0.))
    progress['discarded_update_seconds'] = max(math.fsum(row['update_seconds'] for row in previous), progress.get('discarded_update_seconds', 0.))
    progress['discarded_updates'] = max(sum(row['discarded_updates'] for row in previous), progress.get('discarded_updates', 0))
    if not path.exists():
        assert progress['updates'] == 0
        return
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) >= progress['updates'], 'Durable checkpoint update log is incomplete'
    durable = lines[:progress['updates']]
    assert [json.loads(line)['update'] for line in durable] == list(range(1, progress['updates']+1))
    extra = lines[progress['updates']:]
    if not extra:
        return
    incident = incidents/str(time.time_ns()); incident.mkdir(parents=True)
    (incident/'uncheckpointed-updates.jsonl').write_bytes(b''.join(extra))
    rows = []
    for line in extra:
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    lost_wall = max(0., max((x.get('training_seconds', 0.) for x in rows), default=progress['training_seconds'])-progress['training_seconds'])
    lost_update = math.fsum(row['seconds'] for row in rows)
    progress['training_seconds'] += lost_wall
    progress['discarded_training_seconds'] += lost_wall
    progress['discarded_update_seconds'] += lost_update
    progress['discarded_updates'] += len(rows)
    save(incident/'accounting.json', dict(discarded_updates=len(rows), update_seconds=lost_update, training_seconds=lost_wall,
        checkpoint_updates=progress['updates'], reason='Exact checkpoint rollback; rejected exposures preserved and cumulatively charged'))
    with path.open('wb') as stream:
        stream.write(b''.join(durable)); stream.flush(); os.fsync(stream.fileno())
