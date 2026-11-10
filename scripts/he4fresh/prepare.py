"""Freeze the new recipe and its source closure without opening a CUDA context."""
import copy
import json
import shutil
import subprocess
from .common import *


def main():
    assert not (R/'gpu-owner.json').exists(), 'Do not replace a launched recipe'
    assert DEST.is_mount() and TEACHER.is_mount()
    assert shutil.disk_usage(DEST).free > 100 * 2**30
    OUT.mkdir(exist_ok=True)
    previous = json.load(open(P/'runs/he4broad-20261108-01/recipe.json'))
    config = copy.deepcopy(previous['config'])
    config['data'] = pin(DEST/'index.json')
    assert config['data']['sha256'] == 'd703d61f47eee8e695bbd09293c15bbae57184d078b98931ff4a2d5828bdbd73'
    config.update(parent=config['source'],budget_seconds=BUDGET)
    spec = copy.deepcopy(previous['spec']); spec['training_data'] = config['data']
    for ref in (config['source'],config['data'],config['evaluation']['target']): authenticate(ref)
    save(R/'recipe.json',dict(config=config,spec=spec,
        authorization='Philip, November 10: fresh HE4, newest combined data, up to 12 hours, hourly native evaluations',
        readout_training_wall_seconds=POINTS,update_seconds_cap=BUDGET,
        training_wall_seconds_cap=BUDGET,
        clock='Training wall includes remote materialization and hashing plus updates; excludes model setup/checkpoint/export/native evaluation. GPU update seconds are synchronized and separately reported. Stop at either cap; one complete iteration may cross a boundary.',
        order='All 1568435 windows globally shuffled once with random.Random(8101); cursor preserved',
        storage='Read-only SSHFS teacher; separate writable SSHFS archive on pstack-2080; no bulk local copy'))
    oldproof = json.load(open(P/'runs/he4broad-20261108-01/q4-matrix-proof.json'))
    authenticate(oldproof['basis']); authenticate(oldproof['original']['source'])
    save(R/'q4-matrix-proof.json',oldproof)
    paths = list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
    for directory in ('component_campaign','dflash_baseline','execution_campaign','he4fresh','he4broad'):
        paths += list((ROOT/'scripts'/directory).glob('*.py'))
    paths.append(ROOT/'experiments/w1a8-standing-20261102/baseline.json')
    for path in paths:
        destination = R/'source'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    save(R/'source/source-manifest.json',dict(head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=[pin(R/'source'/p.relative_to(ROOT)) for p in paths]))
    save(R/'prepared.json',dict(status='PREPARED',recipe=pin(R/'recipe.json'),
        source=pin(R/'source/source-manifest.json'),archive_free_bytes=shutil.disk_usage(DEST).free))
    print('PREPARED',flush=True)


if __name__=='__main__': main()
