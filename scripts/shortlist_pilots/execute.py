"""Run explicitly prepared pilot jobs serially under the existing sole-GPU guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path('/home/philip/binary-eagle-decoding')
    run = Path(plan['run_root'])
    python = str(root/'.venv/bin/python')
    base = json.loads((run/'fidelity-guard02-config.json').read_text())
    results = []
    for job in plan['jobs']:
        for command in job.get('prepare', []):
            subprocess.run(command, cwd=root, check=True)
        guard = dict(base)
        guard.update(output=str(run/(job['id']+'-guard')),
                     max_train_seconds=job['max_train_seconds'],
                     wall_seconds=job.get('wall_seconds', 1200))
        guard['env'] = {**base['env'], **job.get('environment', {})}
        guard['inputs'] = [v for v in base['inputs'] if
                           not v['path'].startswith(str(root/'scripts/shortlist_pilots'))
                           and v['path'] != str(run/'fidelity-config.json')]
        pins = json.loads((run/'source-manifest.json').read_text())
        guard['inputs'] += [{'path':k, 'sha256':v} for k,v in pins.items()]
        guard['inputs'] += job.get('inputs', [])
        config = run/(job['id']+'-guard-config.json')
        with config.open('x') as stream:
            json.dump(guard, stream, indent=2)
        script = root/'scripts/shortlist_pilots/guard.py'
        command = [python, '-B', str(run/'source/remote_job.py'),
                   '--output-root', str(run/'supervision'),
                   '--command-root', str(root/'scripts/shortlist_pilots'),
                   '--command-script', str(script), job['id'], '--',
                   python, '-B', str(script), '--config', str(config), '--', *job['argv']]
        launch = {'command':command, 'job':job, 'unix':time.time(),
                  'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                  'interop':os.environ.get('WSL_INTEROP')}
        (run/(job['id']+'-launch.json')).write_text(json.dumps(launch, indent=2)+'\n')
        completed = subprocess.run(command, cwd=root)
        result = {'id':job['id'], 'returncode':completed.returncode}
        for name in job.get('readouts', []):
            path = Path(name)
            if path.is_file():
                result[str(path)] = json.loads(path.read_text())
        receipt = Path(guard['output'])/'result.json'
        if receipt.exists():
            result['guard'] = json.loads(receipt.read_text())
        if completed.returncode:
            log = Path(guard['output'])/'child.log'
            if log.exists():
                result['failure_tail'] = log.read_text(errors='replace')[-10000:]
        results.append(result)
        (run/(plan['id']+'-results.json')).write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
