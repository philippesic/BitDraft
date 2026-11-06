"""Post-confirmation diagnostic only; never retunes the frozen matrix."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/philip/binary-eagle-decoding')
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_native_eagle import wait_ready
sys.path.insert(0, str(ROOT / 'runs/upstream-consistency-20261101'))
import quality_judge

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    out = RUN / 'default-quality-01'
    out.mkdir(exist_ok=False)
    freeze = json.loads((RUN / 'final-freeze.json').read_text())
    cases = {c['id']: c for c in json.loads((RUN / 'final-inputs/quality-cases.json').read_text())}
    prompts = json.loads((RUN / 'final-inputs/requests.json').read_text())
    original = ROOT / 'runs/spec-shortlist-20261030/build/bin'
    cuda = (original / 'libggml-cuda.so.0.28.0').resolve()
    lane = (Path.home() / '.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
    fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    source_rows = [json.loads(line) for line in (RUN / 'final-01/requests.jsonl').read_text().splitlines()]
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'DSPARK_', 'DFLASH_', 'EAGLE_', 'W1AX_', 'LLAMA_ARG_')) and k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    environment.update(LD_LIBRARY_PATH=str(original), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4')
    rows, records = [], []
    server = None
    started = time.perf_counter()
    startup = 0.
    warmup = 0.
    def pause():
        control = json.loads((Path.home() / '.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']
        assert not control.get('pause_requested') and not control.get('stop_all_requested') and not (RUN / 'STOP').exists()
    def request(body):
        req = urllib.request.Request('http://127.0.0.1:18941/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        begin = time.perf_counter()
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.load(response)
        return result, time.perf_counter() - begin
    def stop():
        nonlocal server
        if server is not None:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=10)
            server = None
    try:
        for target in freeze['targets']:
            pause()
            arm = next(a for a in freeze['arms'] if a['target_id'] == target and a['family'] == 'target_only')
            cell = out / target
            cell.mkdir()
            source = RUN / 'final-01' / target / ('rep-0-' + arm['id']) / 'command.json'
            command = json.loads(source.read_text())['command']
            write(cell / 'command.json', {'command': command, 'environment': {k: environment[k] for k in ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')},
                'matched_final_cell': str(source), 'diagnostic_difference': 'original CUDA library and consistency environment unset; no speculative model'})
            with (cell / 'server.log').open('w') as log:
                try:
                    begin = time.perf_counter()
                    server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    wait_ready(server, 'http://127.0.0.1:18941', 180)
                    startup += time.perf_counter() - begin
                    maps = Path(f'/proc/{server.pid}/maps').read_text()
                    (cell / 'loaded-library-maps.txt').write_text(maps)
                    loaded = sorted({line.split()[-1] for line in maps.splitlines() if 'libggml-cuda.so' in line})
                    assert loaded == [str(cuda)]
                    for prompt in prompts[:2]:
                        _, seconds = request({'prompt': prompt['prompt'], 'n_predict': 256, 'temperature': 0, 'seed': 1234, 'cache_prompt': False, 'return_tokens': True})
                        warmup += seconds
                    # Replay the complete same rep-zero sequence before assessing
                    # quality, retaining prompt/history order as a control.
                    work = [r for r in source_rows if r['target_id'] == target and r['family'] == 'target_only' and r['rep'] == 0]
                    work += [{**r, 'diagnostic_rep': 1} for r in work if r['phase'] == 'quality']
                    for row in work:
                        pause()
                        maximum = 256 if row['phase'] == 'performance' else 384
                        result, seconds = request({'prompt': row['result']['prompt'], 'n_predict': maximum, 'temperature': 0,
                                                   'seed': 1234, 'cache_prompt': False, 'return_tokens': True, 'stream': False})
                        assert result['timings']['cache_n'] == 0 and not result['truncated']
                        assert result['prompt'] == row['result']['prompt']
                        output = {'target_id': target, 'phase': row['phase'], 'case_id': row['prompt_id'], 'rep': row.get('diagnostic_rep', 0),
                                  'result': result, 'request_seconds': seconds, 'matches_correctness_tokens': result['tokens'] == row['result']['tokens']}
                        if row['phase'] == 'quality':
                            output['judgement'] = quality_judge.judge(cases[row['prompt_id']], result['content'])
                        rows.append(output)
                        with (out / 'requests.jsonl').open('a') as stream:
                            stream.write(json.dumps(output) + '\n')
                    rr = [r for r in rows if r['target_id'] == target and r['phase'] == 'quality']
                    rep0 = {r['case_id']: r for r in rr if r['rep'] == 0}
                    assert all(r['result']['tokens'] == rep0[r['case_id']]['result']['tokens'] for r in rr)
                    records.append({'target_id': target, 'quality_passed': sum(r['judgement']['passed'] for r in rr if r['rep'] == 0), 'quality_cases': len(rep0),
                                    'quality_exact_to_correctness': sum(r['matches_correctness_tokens'] for r in rr if r['rep'] == 0),
                                    'repeat_identity': True})
                    write(out / 'progress.json', records)
                    print(records[-1], flush=True)
                finally:
                    stop()
        write(out / 'receipt.json', {'completed': True, 'records': records, 'training_seconds': 0,
            'startup_seconds': startup, 'excluded_warmup_seconds': warmup, 'request_seconds': sum(r['request_seconds'] for r in rows),
            'total_wall_seconds': time.perf_counter() - started,
            'scope': 'Post-confirmation original-target-only quality diagnostic; same frozen inputs, no selection changes or performance claim',
            'original_cuda': {'path': str(cuda), 'sha256': hashlib.sha256(cuda.read_bytes()).hexdigest()}})
    finally:
        stop()

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
