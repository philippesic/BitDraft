"""Actual native-parent36-group trace collection; GPU owner launches via remote_job.

Only --help/CPU source checks are preparation. Execution requires --allow-cuda.
Each frozen group gets a fresh server and trace directory. Preserve all emitted
native successes, without claiming CPU actor equivalence or performance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/w1a8-modern-actual-parent-capture-20261103-01'
ROSTER = P / 'runs/w1a8-modern-generalization-data-20261103-01/roster02.json'
CONDITIONAL = P / 'runs/w1a8-modern-generalization-capture-20261103-02'
TRACE_BUILD = P / 'runs/w1a8-modern-trace-build-20261103-02/build.json'
CACHE = P / 'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'
MAP = P / 'runs/w1a8-modern-native-replay-20261103-01/draft-map.ids'
PER_GROUP = 16 * 1024**2
TRACE_CAP = 64 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response_signature(value):
    keys = ('tokens', 'content', 'stop', 'stop_type', 'stopping_word', 'truncated',
            'tokens_predicted', 'has_new_line')
    require(all(key in value for key in keys), 'raw token/text/stop response fields missing')
    require(isinstance(value['tokens'], list) and value['tokens']
            and all(type(token) is int and 0 <= token < 248320 for token in value['tokens'])
            and len(value['tokens']) == value['tokens_predicted'], 'invalid raw completion token IDs')
    return {key: value[key] for key in keys}


def trace_audit(cell, process_id, trace_build, map_ref, parser, utils, response):
    trace = cell / 'trace'
    manifest = read(trace / 'manifest.json')
    require(manifest['schema'] == 'w1a8_actual_native_actor_trace_v2'
            and manifest['pid'] == process_id and manifest['cap_bytes'] == TRACE_CAP
            and manifest['required_llama'] == trace_build['environment']['W1A8_TRACE_LLAMA']
            and manifest['draft_map'] == map_ref['path']
            and manifest['operation_configuration_changed'] is False
            and manifest['model_opened_by_interposer'] is False, 'corrected v2 trace manifest differs')
    files = sorted(path for path in trace.iterdir() if path.is_file())
    require(all(not path.is_symlink() and not path.name.startswith('.') for path in files),
            'trace contains a temporary or substituted file')
    actual_bytes = sum(path.stat().st_size for path in files)
    require(actual_bytes <= TRACE_CAP, 'per-process trace exceeded64MiB cap')
    events = [json.loads(line) for line in (trace / 'calls.jsonl').read_text().splitlines() if line]
    require(events and [event['event'] for event in events] == list(range(1, len(events) + 1)),
            'missing/reordered native trace events')
    before = {}
    after = set()
    native_proposals = []
    variants = {}
    for event in events:
        call = event['call']
        if event['phase'] == 'before':
            require(call not in before, 'duplicate native before call')
            before[call] = event
        else:
            require(event['phase'] == 'after' and call in before and call not in after
                    and event['result'] == 0, 'failed/unpaired native call')
            after.add(call)
            source = before[call]
            if source['architecture'] != 'eagle3' or source['process_type_name'] != 'decode':
                continue
            for selected in event.get('selected', []):
                batch_row = source['batch']['rows'][selected['batch_row']]
                # Model initialization can decode unconditioned placeholder tokens.
                # Actual EAGLE proposals have the supplied feature embedding.
                if not batch_row['has_embd']:
                    continue
                variant = selected['logits_mode']
                require(variant in ('backend_explicit_candidate_pairs', 'dense_target_vocab_mapped_subset'),
                        'ambiguous/old decoder-logit trace schema')
                variants[variant] = variants.get(variant, 0) + 1
                native_proposals.append(selected['argmax_target_token'])
    require(set(before) == after, 'native call trace ended before completion')
    text = (cell / 'server.log').read_text()
    audit = parser.audit_trace(text, response, expected_depth=2)
    lines, _, _ = parser._request_lines(text, None, None)
    logged = []
    for _, line in lines:
        candidate = parser.CANDIDATE.search(line)
        if candidate and int(candidate[2]) == 0:
            logged.append(int(candidate[4]))
    require(native_proposals == logged, 'actual native output proposals differ from serving debug decisions')
    audit['inputs'] = {'log': utils.pin(cell / 'server.log'), 'response': utils.pin(cell / 'response.json'),
                       'parser': utils.pin(Path(parser.__file__))}
    audit['native_output_pair_audit'] = {'matched_debug_proposals': len(logged), 'logits_modes': variants,
                                       'trace_bytes': actual_bytes, 'trace_files': [utils.pin(path) for path in files]}
    utils.atomic(cell / 'protection.json', audit)
    return audit


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--roster', type=Path, default=ROSTER)
    cli.add_argument('--conditional', type=Path, default=CONDITIONAL)
    cli.add_argument('--build', type=Path, default=TRACE_BUILD)
    cli.add_argument('--output', type=Path, default=R / 'capture01')
    cli.add_argument('--allow-cuda', action='store_true')
    args = cli.parse_args()
    require(args.allow_cuda, 'sole GPU owner must explicitly pass --allow-cuda')
    require(not args.output.exists(), 'actual-parent output directory already exists')
    require(os.environ.get('CUDA_CACHE_PATH') and Path(os.environ['CUDA_CACHE_PATH']).resolve() == CACHE.resolve()
            and CACHE.is_dir(), 'root COMMAND must override CUDA_CACHE_PATH to the existing shared cache')
    roster = read(args.roster)
    require(roster['schema'] == 'w1a8_modern_phase1_generalization_roster_v2'
            and roster['status'] == 'FROZEN_INPUTS_AND_BOUNDED_GEOMETRY_NO_LABELS'
            and len(roster['rows']) == 36 and roster['rows'] == roster['roster'], 'frozen36-group roster02 required')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()
    source_root = args.output / 'source'
    source_root.mkdir()
    created = True
    originals = [Path(__file__), P / 'scripts/w1a8_generalization/modern_collect.py',
                 P / 'scripts/w1a8_generalization/modern_serving_trace.py']
    for original in originals:
        shutil.copyfile(original, source_root / original.name)
    utils = load_module('actual_parent_collect_utils', source_root / 'modern_collect.py')
    parser = load_module('actual_parent_trace_parser', source_root / 'modern_serving_trace.py')
    for original in originals:
        require(utils.pin(original)['sha256'] == utils.pin(source_root / original.name)['sha256'],
                'source changed while snapshotting')
    worker = utils.Collector(args)
    worker.created_output = created
    source_pins = [utils.pin(source_root / path.name) for path in originals]
    cleaning = False

    def stop(signum, frame):
        if not cleaning:
            raise utils.StopRequested(f'received signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        worker.stage = 'authenticate_actual_parent_inputs'
        worker.permission()
        worker.budget(36 * PER_GROUP)
        conditional_audit = read(args.conditional / 'root-independent-audit.json')
        summary = read(args.conditional / 'summary.json')
        require(conditional_audit['status'] == 'ROOT_RAW_CAPTURE_AUDIT_PASS'
                and summary['status'] == 'COLLECTION_COMPLETE_WITH_ACCOUNTED_EXCLUSIONS'
                and utils.pin(args.roster)['sha256'] == summary['roster']['sha256'],
                'conditional collection/roster was not independently qualified')
        initial = read(P / 'runs/w1a8-modern-native-admission-20261103-01/freeze.json')
        trace_build = read(args.build)
        require(trace_build['output_cap_bytes'] == TRACE_CAP, 'trace cap differs from64MiB')
        require(utils.pin(trace_build['source']['path'])['sha256']
                == '1c09194a1f54240f31e51e06c7647d52878214c2f25336b1fcfbf8075cb38f12',
                'corrected interposer v2 source required')
        references = [initial['baseline'], initial['target'], initial['w1a8_initialization'],
                      trace_build['candidate_build'], trace_build['candidate_source_pins'],
                      trace_build['source'], trace_build['builder'], trace_build['interposer'], trace_build['compiler']]
        references += trace_build['candidate_binaries'] + trace_build['headers']
        references += [utils.pin(args.roster), utils.pin(args.build), utils.pin(MAP),
                       utils.pin(P / 'runs/w1a8-modern-native-admission-20261103-01/freeze.json'),
                       utils.pin(args.conditional / 'root-independent-audit.json'),
                       utils.pin(args.conditional / 'summary.json')]
        for ref in references:
            worker.checked(ref)
        require(initial['candidate_build'] == trace_build['candidate_build'], 'native parent runtime pins differ')
        candidate = read(trace_build['candidate_build']['path'])
        for ref in candidate['binaries']:
            worker.checked(ref)
        server = next(ref for ref in candidate['binaries'] if Path(ref['path']).name == 'llama-server')
        map_ref = utils.pin(MAP)
        map_values = [int(value) for value in MAP.read_text().split()]
        require(len(map_values) == len(set(map_values)) == 50000
                and min(map_values) >= 0 and max(map_values) < 248320, 'absolute native draft map invalid')
        map_raw = b''.join(value.to_bytes(8, 'little', signed=True) for value in map_values)
        native_export = read(P / 'runs/w1a8-modern-packed-init-20261103-01/export.json')
        require(hashlib.sha256(map_raw).hexdigest() == native_export['preserved_tensors']['d2t']['raw_sha256']
                and native_export['output']['sha256'] == initial['w1a8_initialization']['sha256'],
                'trace map or parent export differs')
        worker.checked(utils.pin(P / 'runs/w1a8-modern-packed-init-20261103-01/export.json'))
        groups = roster['rows']
        require(len({row['id'] for row in groups}) == 36 and len({row['group_id'] for row in groups}) == 36,
                'unique36whole groups required')
        requests = {}
        for row in groups:
            require(re.fullmatch('[A-Za-z0-9_-]+', row['id']), 'safe roster ID required')
            request_path = args.conditional / 'serving' / row['id'] / 'request.json'
            response_path = request_path.with_name('response.json')
            request = read(request_path)
            expected = read(response_path)
            require(request == {'prompt': Path(row['rendered_text_path']).read_bytes().decode('utf-8'),
                    'n_predict': 80, 'temperature': 0, 'seed': 1234, 'cache_prompt': False,
                    'return_tokens': True, 'stream': False}, 'exact frozen80-token request differs')
            response_signature(expected)
            require(expected['truncated'] is False and expected['tokens_evaluated'] == row['native_prompt_tokens'],
                    'conditional target prompt was truncated or retokenized')
            for path in (request_path, response_path, Path(row['rendered_text_path'])):
                worker.checked(utils.pin(path))
            requests[row['id']] = request, expected
        worker.budget(36 * PER_GROUP)
        utils.atomic(args.output / 'freeze.json', {'source_pins': source_pins,
            'input_pins': list(worker.authenticated.values()), 'roster': utils.pin(args.roster),
            'conditional': str(args.conditional.resolve()), 'trace_build': utils.pin(args.build),
            'parent': initial['w1a8_initialization'], 'target': initial['target'], 'depth': 2,
            'groups': [row['id'] for row in groups], 'cache_reused': str(CACHE.resolve()),
            'per_group_predicted_bytes': PER_GROUP, 'trace_per_process_cap_bytes': TRACE_CAP,
            'storage_floor_bytes': utils.FLOOR, 'extra_reserve_bytes': utils.RESERVE,
            'scope': 'Actual native-parent emitted successes across all80returned tokens; no CPU actor equivalence or performance qualification'})
        clean = {key: value for key, value in os.environ.items()
                 if not key.startswith(('GGML_', 'DSPARK_', 'DFLASH_', 'EAGLE_', 'W1AX_', 'LLAMA_', 'W1A8_'))
                 and key not in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'NVIDIA_TF32_OVERRIDE')}
        environment = {**clean, **trace_build['environment'], 'GGML_CUDA_DECODE_CONSISTENT': '32',
                       'GGML_W1AX_ACT_BITS': '8', 'LLAMA_TRACE': '1', 'CUDA_VISIBLE_DEVICES': '0',
                       'OMP_NUM_THREADS': '4', 'CUDA_CACHE_PATH': str(CACHE.resolve()), 'W1A8_TRACE_MAP': map_ref['path']}
        failures = 0
        for index, row in enumerate(groups):
            worker.stage = 'actual_parent_' + row['id']
            worker.permission()
            worker.budget((len(groups) - index) * PER_GROUP)
            cell = args.output / row['id']
            cell.mkdir()
            trace = cell / 'trace'
            trace.mkdir()
            env = {**environment, 'W1A8_TRACE_DIR': str(trace)}
            request, expected = requests[row['id']]
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            argv = [server['path'], '-m', initial['target']['path'], '-ngl', '99', '-c', '4096', '-np', '1',
                    '-b', '512', '-ub', '512', '-t', '4', '-fa', 'on', '--cache-type-k', 'f16',
                    '--cache-type-v', 'f16', '--host', '127.0.0.1', '--port', str(port), '--metrics',
                    '--no-webui', '--no-warmup', '--spec-type', 'draft-eagle3', '-md', initial['w1a8_initialization']['path'],
                    '-ngld', '99', '--cache-type-k-draft', 'f16', '--cache-type-v-draft', 'f16',
                    '--spec-draft-n-max', '2', '--spec-draft-n-min', '0', '--spec-draft-p-min', '0',
                    '--spec-draft-sampling', 'greedy', '-lv', '5']
            utils.atomic(cell / 'command.json', {'argv': argv, 'environment': {key: value for key, value in env.items()
                if key.startswith(('LD_', 'W1A8_', 'LLAMA_', 'GGML_', 'CUDA_', 'OMP_'))}})
            utils.atomic(cell / 'request.json', request)
            utils.atomic(cell / 'target-reference.json', {'request': utils.pin(args.conditional / 'serving' / row['id'] / 'request.json'),
                'response': utils.pin(args.conditional / 'serving' / row['id'] / 'response.json')})
            began = time.perf_counter()
            result = None
            try:
                with (cell / 'server.log').open('x') as log:
                    worker.active = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                    worker.active_identity = utils.process_identity(worker.active)
                    process_id = worker.active.pid
                    utils.atomic(cell / 'identity.json', worker.active_identity)
                    worker.progress()
                    for _ in range(240):
                        worker.permission()
                        require(worker.active.poll() is None, 'native parent server exited during startup')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1):
                                break
                        except (urllib.error.URLError, TimeoutError):
                            time.sleep(.1)
                    else:
                        raise TimeoutError('native parent startup timed out')
                    maps = (Path('/proc') / str(process_id) / 'maps').read_text()
                    (cell / 'loaded-library-maps.txt').write_text(maps)
                    mapped = {line.split()[-1] for line in maps.splitlines() if any(name in line for name in ('libggml', 'libllama', 'libmtmd', 'libw1a8-modern-trace'))}
                    allowed = {str(Path(ref['path']).resolve()) for ref in candidate['binaries']} | {trace_build['interposer']['path']}
                    require(mapped <= allowed and trace_build['interposer']['path'] in mapped
                            and trace_build['environment']['W1A8_TRACE_LLAMA'] in mapped
                            and any('libggml-cuda.so' in path for path in mapped), 'actual native library maps differ')
                    for path in mapped:
                        worker.checked(next(ref for ref in candidate['binaries'] + [trace_build['interposer']] if str(Path(ref['path']).resolve()) == path))
                    native_request = urllib.request.Request(f'http://127.0.0.1:{port}/completion',
                        data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
                    worker.permission()
                    with urllib.request.urlopen(native_request, timeout=120) as response:
                        observed = json.load(response)
                    utils.atomic(cell / 'response.json', observed)
                    require(response_signature(observed) == response_signature(expected),
                            'actual parent differs from frozen target tokens/text/stop/truncation')
                    worker.reap(cell / 'release.json')
                audit = trace_audit(cell, process_id, trace_build, map_ref, parser, utils, observed)
                result = {'id': row['id'], 'group_id': row['group_id'], 'role': row['role'], 'domain': row['domain'],
                    'status': 'ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS', 'token_text_stop_parity': True,
                    'response_tokens': len(observed['tokens']), 'draft_proposed': audit['draft_proposed'],
                    'draft_verified_accepted': audit['draft_verified_accepted'],
                    'emitted_draft_successes': audit['emitted_draft_successes'],
                    'protection': utils.pin(cell / 'protection.json'),
                    'trace_bytes': audit['native_output_pair_audit']['trace_bytes'],
                    'source_identity': row['source_ancestry'], 'cpu_native_actor_equivalence_proven': False}
                failures = 0
            except (ValueError, OSError, TimeoutError, subprocess.TimeoutExpired) as error:
                result = {'id': row['id'], 'group_id': row['group_id'], 'role': row['role'], 'domain': row['domain'],
                    'status': 'ACTUAL_PARENT_TRACE_FAILED_RETAINED', 'error': repr(error),
                    'cpu_native_actor_equivalence_proven': False}
                failures += 1
            finally:
                worker.reap(cell / 'release.json')
            result['wall_seconds'] = time.perf_counter() - began
            utils.atomic(cell / 'result.json', result)
            worker.groups.append(result)
            worker.costs.append({'stage': 'actual_parent_' + row['id'], 'wall_seconds': result['wall_seconds'], 'status': result['status']})
            worker.progress()
            print(json.dumps(result), flush=True)
            require(failures < 3, 'three consecutive actual-parent failures; inspect preserved inputs and trace')
        worker.stage = 'complete'
        utils.atomic(args.output / 'summary.json', {'status': 'ACTUAL_PARENT_COLLECTION_COMPLETE_WITH_ACCOUNTED_FAILURES',
            'groups': worker.groups, 'costs': worker.costs, 'wall_seconds': time.perf_counter() - worker.started,
            'actual_native_protections': sum(group.get('emitted_draft_successes', 0) for group in worker.groups),
            'free_bytes_after': shutil.disk_usage(args.output).free, 'training_updates': 0,
            'cpu_native_actor_equivalence_proven': False,
            'scope': 'Actual emitted native-parent successes retained across entire response; no speed or generalization claim'})
        worker.progress()
    except BaseException as error:
        cleaning = True
        worker.reap(args.output / 'interrupted-release.json')
        utils.atomic(args.output / 'failure.json', {'stage': worker.stage, 'error': repr(error),
            'groups': worker.groups, 'costs': worker.costs, 'wall_seconds': time.perf_counter() - worker.started})
        raise
    finally:
        cleaning = True
        worker.reap(args.output / 'final-release.json')


if __name__ == '__main__':
    main()
