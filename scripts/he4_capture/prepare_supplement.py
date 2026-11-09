"""Replay the fixed admission policy without touching the overnight inputs."""
import collections
import json
import pathlib
import subprocess
import sys

P = pathlib.Path(sys.argv[1])
OUT = P / 'supplemental'
OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('prepare.py')), str(P), str(OUT/'replay')], check=True)
read = lambda p: json.loads(p.read_text())
old = read(P/'prompt-pool.json')
base = read(P/'manifest.json')
replay = read(OUT/'replay/prompt-pool.json')
old_ids = {r['id'] for r in old['records']}
captured_ids = {r['id'] for r in base['records']}
captured_hashes = {read(pathlib.Path(r['receipt']['path']).parent.parent/'source.json').get('rendered_sha256') for r in base['records']}
rows, exclusions = [], list(replay['rejected'])
for row in replay['records']:
    reason = ('already in original prompt pool' if row['id'] in old_ids else
              'already captured id' if row['id'] in captured_ids else
              'already captured rendered text' if row['rendered_sha256'] in captured_hashes else
              'not historical TRAIN' if row['pool'] != 'historical_optimizer' or row['split'] != 'train' else None)
    if reason:
        exclusions.append(dict(id=row['id'], pool=row['pool'], reason=reason))
    else:
        rows.append(row)
selector = read(P/'runs/dspark-a8-full-pool-20261006-01/plans/role-selector.json')['selection']
for r in selector:
    if r['split'] != 'train':
        exclusions.append(dict(id=r['prompt_id'], reason='non-TRAIN role', split=r['split'], shard=r['shard'], row=r['row']))
for id in replay['prior_modern_used']:
    exclusions.append(dict(id=id, reason='modern captured in earlier campaign'))
assert len(rows) == len({r['id'] for r in rows})
assert not ({r['id'] for r in rows} & (old_ids | captured_ids))
assert all(r['pool']=='historical_optimizer' and r['split']=='train' for r in rows)
report = dict(records=rows, sequences=len(rows), fixed_historical_admitted=replay['historical'], original_historical_admitted=old['historical'], bug_fix='Discard None as well as empty identity keys; retain nonempty identity/component exclusions', exclusions=exclusions, exclusion_counts=dict(collections.Counter(r['reason'] for r in exclusions)), prior_modern_used=replay['prior_modern_used'])
(OUT/'prompt-pool.json').write_text(json.dumps(report, indent=2)+'\n')
# The original manifest is frozen here; combined output is always written elsewhere.
if (OUT/'base-manifest.json').exists():
    assert read(OUT/'base-manifest.json') == base
else:
    (OUT/'base-manifest.json').write_bytes((P/'manifest.json').read_bytes())
print(json.dumps({k:v for k,v in report.items() if k not in ('records','exclusions','prior_modern_used')}, indent=2))
