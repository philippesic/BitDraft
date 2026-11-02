"""Revalidate the target runtime and smoke-test new matrix exports and harness."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from run_matrix import pin, read, sha, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    freeze = read(run / 'screen-freeze.json')
    freeze['scope'] = 'Development pilot: native runtime regression plus target/MTP/corrected EAGLE BF16/Q4_0; no held-out or ranking claim'
    freeze['arms'] = [arm for arm in freeze['arms'] if arm['family'] == 'target_only'
                      or (arm['family'] == 'mtp' and arm['depth'] == 3)
                      or (arm['family'] == 'eagle3' and arm['dtype'] in ('BF16', 'Q4_0') and arm['depth'] == 3)]
    freeze['sources'].append(pin(__file__))
    freeze['pins'].append(pin(__file__))
    write(run / 'pilot-freeze.json', freeze)
    inputs = run / 'pilot-inputs'
    inputs.mkdir(exist_ok=False)
    for name in ('requests.json', 'quality-cases.json', 'quality-controls.json'):
        (inputs / name).write_bytes((run / 'screen-inputs' / name).read_bytes())
    write(inputs / 'receipt.json', {'manifest_sha256': sha(run / 'pilot-freeze.json'),
                                  'files': {file.name: sha(file) for file in inputs.glob('*.json')}})
    lane = (Path.home() / '.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
    fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    control_path = Path.home() / '.config/binary-eagle-decoding/gpu-control.json'
    native = run / 'native-gates-01'
    native.mkdir(exist_ok=False)
    records = []
    variants = [('original_negative', next(iter(freeze['targets'])))] + [('quant4group', target) for target in freeze['targets']]
    for variant, target in variants:
        control = read(control_path)['rtx5080']
        assert not control.get('pause_requested') and not control.get('stop_all_requested') and not (run / 'STOP').exists()
        directory = native / (variant + '-' + target)
        directory.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'DSPARK_', 'DFLASH_', 'EAGLE_', 'LLAMA_ARG_')) and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
        env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', LD_LIBRARY_PATH=str(project / 'runs/spec-shortlist-20261030/build/bin'))
        if variant == 'quant4group':
            env.update(GGML_CUDA_DECODE_CONSISTENT='32', LD_LIBRARY_PATH=freeze['library_path'])
        command = [str(project / 'runs/upstream-consistency-20261101/test-grouped-parity'), '-m', freeze['targets'][target]['path'],
                   '-ngl', '99', '-t', '4', '-fa', 'on', '--test', 'spec-parity', '--logs', str(directory / 'logs')]
        write(directory / 'command.json', {'command': command, 'environment': {k: v for k, v in env.items() if k.startswith(('GGML_', 'LD_', 'CUDA_', 'OMP_'))}})
        began = time.perf_counter()
        with (directory / 'stdout.log').open('w') as log:
            process = subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
        contents = '\n'.join(file.read_text(errors='replace') for file in (directory / 'logs').glob('*.log'))
        comparisons = re.findall(r'spec parity: checks=(\d+) greedy_flips=(\d+) max_abs=([^\s]+)', contents)
        record = {'variant': variant, 'target_id': target, 'exit_code': process.returncode,
                  'wall_seconds': time.perf_counter() - began, 'checks': sum(int(row[0]) for row in comparisons),
                  'greedy_flips': sum(int(row[1]) for row in comparisons),
                  'max_abs': max((float(row[2]) for row in comparisons), default=None)}
        records.append(record)
        write(native / 'progress.json', records)
        print(record, flush=True)
        assert record['checks'] > 0
        assert process.returncode != 0 if variant == 'original_negative' else process.returncode == 0
    write(native / 'receipt.json', {'completed': True, 'invalid_control_rejected': True, 'records': records,
                                  'native_binary': pin(project / 'runs/upstream-consistency-20261101/test-grouped-parity'), 'training_seconds': 0})
    fcntl.flock(lane, fcntl.LOCK_UN)
    lane.close()
    command = [sys.executable, '-B', str(Path(__file__).with_name('run_matrix.py')), '--project', str(project), '--run', str(run),
               '--manifest', str(run / 'pilot-freeze.json'), '--inputs', str(inputs), '--phase', 'screen', '--output', 'pilot-01']
    result = subprocess.run(command, cwd=project)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    try:
        main()
    finally:
        subprocess.run(['tmux', '-L', 'binary-eagle-runtime', 'wait-for', '-S', 'qwen35-matrix-pilot-20261101-complete'], check=False)
