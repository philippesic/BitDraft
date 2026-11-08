"""Run the bounded HE4 trajectory, recovery proof, and serialized native panels."""
import json
import subprocess
import sys
from scripts.component_campaign.common import P, ROOT, Q4, TARGET, identity, pause, pin, save
from scripts.he4_continuation.train import R, OUT, COMPONENT

RUNTIME = COMPONENT / 'runtime04'


def run(module, arguments, label):
    pause()
    command = [sys.executable, '-m', module, *map(str, arguments)]
    save(R / (label + '-command.json'), dict(argv=command, cwd=str(ROOT), owner=identity()))
    # Stay in the contender job's supervised process group.
    with (R / (label + '.log')).open('x') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def panel(label, arms, reps, warmups, cap=None):
    directory = R / 'panels'
    directory.mkdir(exist_ok=True)
    save(directory / (label + '-arms.json'), arms)
    save(directory / (label + '-cache.json'), {k: ['f16', 'f16'] for k, v in arms.items() if v})
    args = ['--arms', directory / (label + '-arms.json'), '--draft-cache-map', directory / (label + '-cache.json'),
            '--runtime', RUNTIME, '--target', TARGET, '--depth', '2', '--cohort', 'capture3',
            '--cohort-label', 'consumed-DEV3-HE4-continuation' + ('-smoke64' if cap else '-256'),
            '--out', R / label, '--reps', reps, '--warmups', warmups]
    if cap:
        args += ['--output-cap', cap]
    run('scripts.component_campaign.native', args, label + '-worker')


def main():
    save(R / 'workflow-owner.json', identity())
    try:
        run('scripts.he4_continuation.train', [], 'training')
        result = json.load(open(OUT / 'result.json'))
        if result['status'] != 'COMPLETE':
            save(R / 'workflow-result.json', dict(status=result['status'], training=pin(OUT / 'result.json')))
            return
        run('scripts.he4_continuation.verify_final', [], 'final-recovery-worker')
        # The trainer has exited. Native servers now own this same locked lane.
        panel('native-final', dict(target=None, q4=str(Q4), parent=str(OUT / 'model-0.gguf'),
                                  after=result['readouts'][-1]['model']['path']), 5, 2)
        for point in (1800, 7200, 14400):
            panel('native-plus' + str(point), dict(target=None, parent=str(OUT / 'model-0.gguf'),
                                                 continuation=str(OUT / f'model-{point}.gguf')), 1, 0, 64)
        run('scripts.he4_continuation.publish', [], 'publish-worker')
        save(R / 'workflow-result.json', dict(status='COMPLETE', training=pin(OUT / 'result.json'),
            final_recovery=pin(OUT / 'final-recovery/result.json'), native=pin(R / 'native-final/result.json'),
            published=pin(P / 'experiments/he4-continuation-20261107/results.json')))
    except BaseException as error:
        save(R / 'workflow-result.json', dict(status='STOPPED_WITH_ERROR', error=repr(error), owner=identity(),
            recovery=json.load(open(OUT / 'recovery.json')) if (OUT / 'recovery.json').exists() else None))
        raise


if __name__ == '__main__':
    main()
