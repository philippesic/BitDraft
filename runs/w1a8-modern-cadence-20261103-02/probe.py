"""Bounded development diagnostic: same tokens, different verifier-like batches."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/w1a8-modern-cadence-20261103-02'
BUILD = P / 'runs/w1a8-modern-capture-build-20261103-08/build.json'
PILOT = P / 'runs/qwen35-correctness-matrix-20261101-01/pilot-inputs/requests.json'
REFERENCES = P / 'runs/qwen35-correctness-matrix-20261101-01/pilot-01/requests.jsonl'


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def authenticate(ref):
    assert pin(ref['path'])['sha256'] == ref['sha256'], ref['path']


def permission():
    control = read(Path.home() / '.config/binary-eagle-decoding/gpu-control.json')['rtx5080']
    assert not control.get('pause_requested') and not (R / 'STOP').exists()
    assert shutil.disk_usage(R).free > 2 * 1024**3, 'preserve 2 GiB storage floor'


def execute(argv, name, env):
    permission()
    (R / (name + '-command.json')).write_text(json.dumps({'argv': argv, 'environment': b['environment']}, indent=2))
    with (R / (name + '.log')).open('x') as stream:
        subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)


b = read(BUILD)
baseline = read(P / 'experiments/w1a8-standing-20261102/baseline.json')
target = baseline['rows'][0]['target_pin']
for ref in b['runtime_libraries'] + b['runtime_sources'] + b['included_local_headers'] + [
    b['cuda_overlay'], b['source'], b['builder'], b['executable'], b['baseline'], target
]:
    authenticate(ref)
clean = {k: v for k, v in os.environ.items() if not k.startswith(
    ('GGML_', 'DSPARK_', 'DFLASH_', 'EAGLE_', 'W1AX_', 'LLAMA_ARG_'))
    and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'NVIDIA_TF32_OVERRIDE')}
env = {**clean, **b['environment'], 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '4'}
prompts = {row['id']: row for row in read(PILOT)}
references = {}
for line in REFERENCES.open():
    row = json.loads(line)
    if row['target_id'] == '4B-Q4_K_M' and row['family'] == 'target_only' and row['phase'] == 'performance':
        references[row['prompt_id']] = row['result']['tokens'][:9]
freeze = {'build': pin(BUILD), 'target': target, 'pilot_inputs': pin(PILOT),
          'pilot_reference_requests': pin(REFERENCES), 'source': pin(__file__),
          'prompt_ids': [4], 'continuation_tokens': 8,
          'scope': 'Existing pilot development only; same fixed tokens, prefill512 and decode chunks1/2/5; no training, untouched release or throughput claim'}
(R / 'freeze.json').write_text(json.dumps(freeze, indent=2))
started = time.monotonic()
results = {}
for prompt_id in freeze['prompt_ids']:
    item = prompts[prompt_id]
    assert hashlib.sha256(item['prompt'].encode()).hexdigest() == item['sha256']
    prefix = 'prompt-%02d' % prompt_id
    raw = R / (prefix + '.txt')
    raw.write_text(item['prompt'])
    token_dir = R / (prefix + '-tokens')
    execute([b['executable']['path'], '--tokenize', '--model', target['path'],
             '--text-file', str(raw), '--out', str(token_dir)], prefix + '-tokenize', env)
    fixed = R / (prefix + '-continuation.ids')
    assert len(references[prompt_id]) == 9
    fixed.write_text('\n'.join(map(str, references[prompt_id][:8])) + '\n')
    tokens = [int(x) for x in (token_dir / 'tokens.ids').read_text().split()]
    cells = {}
    for name, chunk, features in [('control1', 1, False), ('features1', 1, True),
                                  ('features2', 2, True), ('features5', 5, True), ('repeat5', 5, True)]:
        out = R / (prefix + '-' + name)
        argv = [b['executable']['path'], '--model', target['path'], '--tokens', str(token_dir / 'tokens.ids'),
                '--out', str(out), *b['required_backend_argv'], '--decode-tokens', str(fixed),
                '--prefill-chunk', '512', '--decode-chunk', str(chunk), '--threads', '4', '--max-output-mib', '128']
        if not features:
            argv.append('--no-features')
        execute(argv, prefix + '-' + name, env)
        meta = read(out / 'capture.json')
        assert meta['input_tokens'] == tokens + references[prompt_id][:8]
        assert meta['feature_shape'] == [len(tokens) + 8, 2560] and meta['logits_shape'] == [9, 248320]
        assert meta['decode_chunk'] == chunk and meta['prefill_chunk'] == 512
        logits = np.fromfile(out / 'logits.f32', dtype='<f4').reshape(meta['logits_shape'])
        assert np.isfinite(logits).all() and logits.argmax(-1).tolist() == meta['greedy_ids']
        mapped = sorted({line.split()[-1] for line in (out / 'loaded-library-maps.txt').read_text().splitlines() if 'libggml-cuda.so' in line})
        assert mapped == [baseline['cuda']['path']]
        cells[name] = {'metadata': pin(out / 'capture.json'), 'argmax_ids': meta['greedy_ids'],
                       'server_reference_match': meta['greedy_ids'] == references[prompt_id],
                       'files': [pin(path) for path in sorted(out.glob('*.f32'))]}
        print(json.dumps({'prompt': prompt_id, 'case': name, 'matches_reference': cells[name]['server_reference_match']}), flush=True)
    assert cells['control1']['argmax_ids'] == cells['features1']['argmax_ids']
    assert cells['repeat5']['argmax_ids'] == cells['features5']['argmax_ids']
    numeric = {}
    for tap in [2, 16, 29]:
        arrays = {name: np.fromfile(R / (prefix + '-' + name) / ('layer-%d.f32' % tap), dtype='<f4').reshape(len(tokens) + 8, 2560)
                  for name in ['features1', 'features2', 'features5', 'repeat5']}
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert np.array_equal(arrays['features5'], arrays['repeat5'])
        assert all(np.array_equal(arrays['features1'][:len(tokens)], a[:len(tokens)]) for a in arrays.values())
        ref = arrays['features1'][len(tokens):].astype(np.float64)
        numeric[str(tap)] = {}
        for name in ['features2', 'features5']:
            values = arrays[name][len(tokens):].astype(np.float64)
            difference = ref - values
            numeric[str(tap)][name] = {'max_abs': float(np.abs(difference).max()),
                                      'relative_rms': float(np.linalg.norm(difference) / np.linalg.norm(ref))}
    results[str(prompt_id)] = {'token_count': len(tokens), 'cells': cells, 'continuation_numeric': numeric,
                               'prefill_byteexact': True, 'repeat5_byteexact_features': True}
result = {'status': 'COMPLETE', 'freeze': pin(R / 'freeze.json'), 'scope': freeze['scope'], 'results': results,
          'all_cadence_labels_match_server_reference': all(c['server_reference_match'] for p in results.values() for c in p['cells'].values()),
          'wall_seconds': time.monotonic() - started, 'training_updates': 0}
(R / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'status': result['status'], 'all_cadence_labels_match_server_reference': result['all_cadence_labels_match_server_reference'], 'wall_seconds': result['wall_seconds']}), flush=True)
