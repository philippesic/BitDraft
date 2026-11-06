"""One original CPU keeper creation/readiness/cleanup; never owns GPU or config."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/home/philip/binary-eagle-decoding')
HERE = Path(__file__).resolve().parent
CONFIG_SHA = 'e98d6c9dfe0fb3be7038808a66535bbc6e89da4031d6a86538eea177409eb09a'
RUNTIME_SHA = '7d64b4a7af99bcde872180fd50ea1750c27925cf8bb499f09f266626a9a6fe90'
CANDIDATE_SHA = 'c7171a00d6e1fbcc702ee8c90f80c1041577545be60705a42c6e712a6ef7520d'
BOOT = '808f6149-6c87-47f4-890d-3a9aebe15dca'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def birth(pid):
    return int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])

assert os.getuid() == 1000 and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT
assert digest(ROOT / 'configs/training.json') == CONFIG_SHA
assert digest(ROOT / 'src/w1a1_eagle/direct_training_runtime.py') == RUNTIME_SHA
assert digest(HERE / 'ensure_environment.py') == CANDIDATE_SHA
sys.path.insert(0, str(ROOT / 'src'))
from w1a1_eagle import direct_training_runtime as runtime
spec = importlib.util.spec_from_file_location('keeper_boundary_candidate', HERE / 'ensure_environment.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
config, _ = runtime.metadata(ROOT / 'configs/training.json')
lane = runtime.checked(config['lane'])
original_keeper = runtime.checked(config['keeper'])
original_environment = runtime.keeper_environment(config['keeper'])
operation = HERE / 'operation'
operation.mkdir(mode=0o700, exist_ok=False)


def authorize():
    runtime.current_control(lane)
    assert digest(ROOT / 'configs/training.json') == CONFIG_SHA
    assert digest(ROOT / 'src/w1a1_eagle/direct_training_runtime.py') == RUNTIME_SHA
    assert runtime.keeper_environment(config['keeper']) == original_environment


def stop_new_keeper():
    metadata_path = operation / 'keeper/keeper.json'
    if not metadata_path.exists():
        return {'status': 'no_published_keeper', 'partial_operation_preserved': True}
    value, _ = runtime.metadata(metadata_path)
    pid = value['pid']
    assert value['schema'] == 'bounded_wsl_interop_keeper_v2'
    assert value['boot_id'] == BOOT and pid != original_keeper['pid']
    assert Path('/proc', str(pid)).stat().st_uid == os.getuid()
    assert birth(pid) == value['birth_ticks']
    argv = Path('/proc', str(pid), 'cmdline').read_bytes().split(bytes([0]))
    assert str(ROOT / 'scripts/wsl_interop_keeper_20261011.py').encode() in argv
    assert str(operation / 'keeper').encode() in argv
    stop = Path(value['stop_file'])
    assert stop == operation / 'keeper/STOP'
    descriptor = os.pidfd_open(pid, 0)
    try:
        assert birth(pid) == value['birth_ticks']
        with stop.open('x') as stream:
            stream.write('owned CPU keeper boundary check complete\n')
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state, _ = runtime.metadata(metadata_path)
        try:
            raw = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
            own_live = int(raw[19]) == value['birth_ticks'] and raw[0] != 'Z'
        except FileNotFoundError:
            own_live = False
        if state.get('status') == 'stopped' and not own_live:
            return {'status': 'stopped', 'pid': pid, 'birth_ticks': value['birth_ticks'],
                    'boot_id': BOOT, 'kernel_identity_present': False,
                    'keeper': {'path': str(metadata_path),
                    'sha256': runtime.metadata(metadata_path)[1]}}
        time.sleep(.25)
    raise TimeoutError('own new CPU keeper did not publish stopped metadata')

result = {'schema': 'actual_inline_keeper_boundary_check_v1', 'original_config': config['keeper'],
          'scope': 'Actual Windows/WSL CPU keeper boundary only; no GPU/data/TRAIN/config change.',
          'gpu_queried': False, 'model_loaded': False, 'training_started': False}
primary = None
try:
    authorize()
    # A private missing locator exercises the real stale-keeper branch without
    # editing or extending the live job's existing keeper/configuration.
    selected = dict(config, keeper={'path': str(operation / 'absent-keeper.json'), 'sha256': '0' * 64})
    keeper, environment, previous = helper.ensure_keeper(
        selected, operation, ROOT, authorize, runtime.keeper_environment)
    observed = runtime.checked(keeper)
    assert observed['pid'] != original_keeper['pid']
    result.update(status='PASS', created_keeper=keeper, created_metadata=observed,
                  environment=environment, previous_failure=previous)
except BaseException as error:
    primary = error
    result.update(status='FAILED', type=type(error).__name__, reason=str(error))
finally:
    try:
        result['owned_cpu_cleanup'] = stop_new_keeper()
        assert result['owned_cpu_cleanup']['status'] == 'stopped' or primary is not None
        authorize()
        result['original_keeper_unchanged'] = runtime.checked(config['keeper']) == original_keeper
        result['original_config_unchanged'] = digest(ROOT / 'configs/training.json') == CONFIG_SHA
    except BaseException as cleanup:
        result['cleanup_failure'] = {'type': type(cleanup).__name__, 'reason': str(cleanup)}
        if primary is None:
            primary = cleanup
            result['status'] = 'FAILED'
    runtime.write_new(HERE / 'result.json', result)
    print(json.dumps({'result': str(HERE / 'result.json'), 'sha256': digest(HERE / 'result.json'),
                      'bytes': (HERE / 'result.json').stat().st_size, 'record': result}), flush=True)
if primary is not None:
    raise primary
