"""Freeze this run's Python closure and write an existing contender-pipeline plan."""
import json
import shutil
import subprocess
from pathlib import Path
from scripts.component_campaign.common import P, ROOT, Q4, TARGET, identity, pin, save
from scripts.he4_continuation.train import R, OUT, COMPONENT, BUDGET, POINTS
from scripts.he4_continuation.workflow import RUNTIME


def main():
    R.mkdir(exist_ok=False)
    source = R / 'source'
    paths = [*list((ROOT / 'src').rglob('*.py')), *list((ROOT / 'scripts').glob('*.py'))]
    for name in ('component_campaign', 'dflash_baseline', 'execution_campaign', 'he4_continuation'):
        paths += list((ROOT / 'scripts' / name).glob('*.py'))
    paths.append(ROOT / 'experiments/w1a8-standing-20261102/baseline.json')
    for path in paths:
        dest = source / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    files = [pin(source / path.relative_to(ROOT)) for path in paths]
    save(source / 'source-manifest.json', dict(owner=identity(), head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), files=files))
    retained = json.load(open(COMPONENT / 'frontier-retention.json'))
    parent = retained['states']['/mnt/c/binary-eagle-component-20261105-01/HE4-confirm-4096.pt.bsz']
    refs = [pin(COMPONENT / 'frontier-retention.json'), pin(RUNTIME / 'build.json'), pin(Q4), pin(TARGET),
            pin(parent['archive']), pin(parent['native']['path']), pin(parent['confirmed_result']['path'])]
    assert refs[-3]['sha256'] == parent['archive_sha256'] and refs[-3]['bytes'] == parent['archive_bytes']
    assert refs[-2] == parent['native'] and refs[-1] == parent['confirmed_result']
    plan = dict(schema='contender_plan_v1', budget_seconds=37800, candidate_budgets={'HE4':37800}, grace_seconds=900,
                gpu=dict(control=str(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'),
                         lock=str(P/'runs/contender-gpu.lock')),
                inputs=[*refs, *files, pin(source/'source-manifest.json')], jobs=[dict(
                    id='he4-overnight',stage=0,candidate='HE4',depends_on=[],timeout_seconds=36000,
                    argv=['/usr/bin/env','PYTHONPATH='+str(source)+':'+str(source/'src'),str(P/'.venv/bin/python'),'-m','scripts.he4_continuation.workflow'], cwd=str(source),
                    checks=[dict(path=str(R/'workflow-result.json'), keys=['status'],equals='COMPLETE')])])
    save(R / 'plan.json', plan)
    save(P / 'experiments/he4-continuation-20261107/manifest.json', dict(
        status='PREPARED', run=str(R), plan=pin(R/'plan.json'), parent=parent,
        update_budget_seconds=BUDGET, readout_update_seconds=[0,*POINTS],
        checkpoint_policy='All eight new full states retained; complete-update SIGTERM; immutable parent preserved.',
        training_policy='Exact HE4 masters, Adam, RNG, data order/cursor and recipe; no other arm trained.',
        native_policy='Existing component native runner/runtime04: final target, external Q4, HE4 parent, continuation; consumed DEV3, 256-token reserve, 2 warmups, 5 alternating reps. Intermediate 64-token smoke panels after trainer release.',
        accounting='21600 synchronized forward/backward/optimizer seconds, one complete-update overshoot maximum. Setup, qualification, serialization and native evaluation excluded and reported. 10h wall timeout is safety envelope, not extra optimizer authorization.',
        source=pin(source/'source-manifest.json'), owner=identity()))
    print(R / 'plan.json')


if __name__ == '__main__':
    main()
