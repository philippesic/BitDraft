"""Serial, pinned Qwen3.5 baseline cells using the retained correctness runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def signature(result):
    return [result.get(key) for key in ('tokens', 'content', 'stop_type', 'stopping_word', 'truncated')]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--phase', choices=('screen', 'final'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    manifest = read(args.manifest)
    settings = manifest['settings']
    out = run / args.output
    out.mkdir(exist_ok=False)
    port = 18941
    endpoint = f'http://127.0.0.1:{port}'
    sys.path.insert(0, str(project / 'scripts'))
    from benchmark_native_eagle import available_port, counter_delta, metrics_text, wait_ready
    judge = load_module('matrix_quality_judge', manifest['judge']['path'])
    start = time.perf_counter()
    costs = {'training_seconds': 0, 'preflight_seconds': 0, 'startup_seconds': 0,
             'excluded_warmup_seconds': 0, 'performance_request_seconds': 0,
             'quality_request_seconds': 0}
    rows, records = [], []
    process = None

    def check_pause():
        state = read(Path.home() / '.config/binary-eagle-decoding/gpu-control.json')['rtx5080']
        if state.get('pause_requested') or state.get('stop_all_requested') or (run / 'STOP').exists():
            raise InterruptedError('GPU pause or experiment STOP requested')

    def request(body):
        request = urllib.request.Request(endpoint + '/completion', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        began = time.perf_counter()
        with urllib.request.urlopen(request, timeout=240) as response:
            raw = response.read()
        return json.loads(raw), time.perf_counter() - began

    def stop():
        nonlocal process
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            process = None

    def interrupted(*_):
        raise KeyboardInterrupt('supervisor interruption')

    signal.signal(signal.SIGTERM, interrupted)
    lane = (Path.home() / '.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
    fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_pause()
    assert available_port('127.0.0.1', port), 'port occupied'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    for ref in manifest['pins']:
        assert Path(ref['path']).stat().st_size == ref['bytes'] and sha(ref['path']) == ref['sha256'], ref['path']
    inputs_receipt = read(args.inputs / 'receipt.json')
    assert inputs_receipt['manifest_sha256'] == sha(args.manifest)
    for name, digest in inputs_receipt['files'].items():
        assert sha(args.inputs / name) == digest, name
    prompts = read(args.inputs / 'requests.json')
    cases = read(args.inputs / 'quality-cases.json')
    controls = read(args.inputs / 'quality-controls.json')
    assert len(controls) == len(cases)
    assert all(c['correct_control_passed'] and c['known_wrong_control_rejected'] for c in controls)
    repetitions = manifest['repetitions']
    if args.phase == 'final':
        assert repetitions >= 5 and len(prompts) == len(cases) == 24
        assert inputs_receipt['created_at_utc'] > manifest['frozen_at_utc']
    write(out / 'plan.json', {'manifest': pin(args.manifest), 'inputs': pin(args.inputs / 'receipt.json'),
                             'runner': pin(__file__), 'phase': args.phase,
                             'repetitions': repetitions, 'performance_prompts': len(prompts),
                             'quality_prompts': len(cases), 'training_seconds': 0})
    write(out / 'hardware.json', {'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,clocks.sm,power.draw', '--format=csv'], text=True),
                                'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                'hostname': os.uname().nodename})
    costs['preflight_seconds'] = time.perf_counter() - start
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(('GGML_', 'DSPARK_', 'DFLASH_', 'EAGLE_', 'W1AX_', 'LLAMA_ARG_'))
                   and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'NVIDIA_TF32_OVERRIDE')}
    environment.update(GGML_CUDA_DECODE_CONSISTENT='32', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4',
                       LD_LIBRARY_PATH=manifest['library_path'])
    error = None
    try:
        for target_id, target in manifest['targets'].items():
            target_arms = [arm for arm in manifest['arms'] if arm['target_id'] == target_id]
            assert target_arms[0]['family'] == 'target_only'
            for rep in range(repetitions):
                order = target_arms if rep % 2 == 0 else list(reversed(target_arms))
                # Alternate direction and rotate cells, keeping the same frozen cells.
                if rep:
                    offset = (rep * 3) % len(order)
                    order = order[offset:] + order[:offset]
                for arm in order:
                    check_pause()
                    cell = out / target_id / f"rep-{rep}-{arm['id']}"
                    cell.mkdir(parents=True)
                    record = {'target_id': target_id, 'arm_id': arm['id'], 'rep': rep, 'status': 'starting'}
                    command = [manifest['server']['path'], '-m', target['path'], '-ngl', '99', '-c', str(settings['context']),
                               '-np', '1', '-b', str(settings['batch']), '-ub', str(settings['ubatch']), '-t', '4',
                               '-fa', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                               '--host', '127.0.0.1', '--port', str(port), '--metrics', '--no-webui', '--no-warmup',
                               '--spec-type', arm['spec_type']]
                    if arm['family'] != 'target_only':
                        command += ['--spec-draft-n-max', str(arm['depth']), '--spec-draft-n-min', '0', '--spec-draft-p-min', '0', '--spec-draft-sampling', 'greedy']
                    if arm.get('draft'):
                        command += ['-md', arm['draft']['path'], '-ngld', '99', '--cache-type-k-draft', 'f16', '--cache-type-v-draft', 'f16']
                    write(cell / 'command.json', {'command': command, 'environment': {k: v for k, v in environment.items() if k.startswith(('GGML_', 'CUDA_', 'LD_', 'OMP_'))}})
                    cell_start = time.perf_counter()
                    with (cell / 'server.log').open('w') as log:
                        try:
                            began = time.perf_counter()
                            process = subprocess.Popen(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT)
                            write(cell / 'process.json', {'pid': process.pid, 'pgid': os.getpgid(process.pid),
                                                        'start_ticks': Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
                                                        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
                            wait_ready(process, endpoint, 180)
                            startup = time.perf_counter() - began
                            costs['startup_seconds'] += startup
                            maps = Path(f'/proc/{process.pid}/maps').read_text()
                            (cell / 'loaded-library-maps.txt').write_text(maps)
                            cuda = sorted({line.split()[-1] for line in maps.splitlines() if 'libggml-cuda.so' in line})
                            assert cuda == [manifest['cuda']['path']], cuda
                            write(cell / 'gpu.json', {'value': subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader'], text=True)})
                            for prompt in prompts[:2]:
                                _, seconds = request({'prompt': prompt['prompt'], 'n_predict': settings['output_limit'],
                                                      'temperature': 0, 'seed': 1234, 'cache_prompt': False, 'return_tokens': True})
                                costs['excluded_warmup_seconds'] += seconds
                            work = [('performance', prompt) for prompt in (prompts if rep % 2 == 0 else list(reversed(prompts)))]
                            if rep == 0:
                                work += [('quality', case) for case in cases]
                            for phase, item in work:
                                check_pause()
                                if phase == 'performance':
                                    prompt, maximum = item['prompt'], settings['output_limit']
                                else:
                                    prompt = ('<|im_start|>system\nFollow the requested answer format precisely.<|im_end|>\n'
                                              '<|im_start|>user\n' + item['prompt'] + '<|im_end|>\n'
                                              '<|im_start|>assistant\n<think>\n\n</think>\n\n')
                                    maximum = 384
                                before = metrics_text(endpoint)
                                result, seconds = request({'prompt': prompt, 'n_predict': maximum, 'temperature': 0,
                                                           'seed': 1234, 'cache_prompt': False, 'return_tokens': True, 'stream': False})
                                after = metrics_text(endpoint)
                                assert result['timings']['cache_n'] == 0 and result['tokens']
                                assert result['tokens_predicted'] == len(result['tokens'])
                                assert not result.get('truncated'), 'prompt context truncated'
                                counters = counter_delta(before, after)
                                assert all(type(v) is int and v >= 0 for v in counters.values()), counters
                                assert counters['accepted'] <= counters['proposed']
                                if arm['family'] != 'target_only':
                                    assert counters['accepted'] == result['timings']['draft_n_accepted']
                                    assert counters['proposed'] == result['timings']['draft_n']
                                costs[phase + '_request_seconds'] += seconds
                                row = {'target_id': target_id, 'arm_id': arm['id'], 'family': arm['family'], 'rep': rep,
                                       'phase': phase, 'prompt_id': item['id'], 'full_request_seconds': seconds,
                                       'output_tokens': len(result['tokens']), 'speculative': counters, 'result': result}
                                if phase == 'quality':
                                    row['judgement'] = judge.judge(item, result['content'])
                                rows.append(row)
                                with (out / 'requests.jsonl').open('a') as stream:
                                    stream.write(json.dumps(row) + '\n')
                            record.update(status='completed', startup_seconds=startup)
                        except (KeyboardInterrupt, InterruptedError):
                            raise
                        except Exception as exc:
                            record.update(status='failed', error=repr(exc))
                        finally:
                            stop()
                    record['wall_seconds'] = time.perf_counter() - cell_start
                    records.append(record)
                    write(out / 'progress.json', records)
                    print(json.dumps(record), flush=True)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        stop()
        costs['total_phase_wall_seconds'] = time.perf_counter() - start
        write(out / 'receipt.json', {'completed': error is None, 'error': error, 'records': records,
                                     'costs': costs, 'requests': len(rows),
                                     'scope': args.phase + ' matched baseline matrix, not a W1A8 result'})
        fcntl.flock(lane, fcntl.LOCK_UN)
        lane.close()


if __name__ == '__main__':
    main()
