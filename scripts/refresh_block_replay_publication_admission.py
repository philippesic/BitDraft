#!/usr/bin/env python3
"""Refresh only a replay publication's admission locator; never issue admission.

Pure stdlib metadata consumer. A pinned report/receipt must come from an actual
full audit by the frozen reader. Pins authenticate bytes, not who ran the audit.
The operator must establish that provenance before invoking this tool.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

READER_SHA256 = 'd8531872a1641c7b35e822107a3186abf9e16542cabbe83e6c0dfc0bdfcd5544'
HISTORICAL_CAPTURE_LABEL = 'module:scripts/capture_nine_model_train_data.py'
HISTORICAL_CAPTURE_SHA256 = '75b5d06aade349379df209b77f1e227f2b9f4084e54281819e8511ffb601be65'
CURRENT_CAPTURE_SHA256 = '7355cc2a7f655247f36c6904872da406630b08ce947cf32640c28f908fc30c0b'
CURRENT_CAPTURE_GIT_OBJECT = 'a0371af5713208f9726e7a9f6ddb966e8bebc3a4'
STAT_KEYS = {'path', 'bytes', 'mtime_ns', 'ctime_ns', 'device', 'inode'}
RECORD_KEYS = STAT_KEYS | {'sha256'}
NATIVE_KEYS = ('target_sha256', 'producer_binary_sha256', 'producer_source_revision',
               'client_source_sha256', 'target_precision', 'kv_type', 'tap_ids',
               'tokens', 'decode_history', 'features_shape', 'logits_indices',
               'logits_shape', 'logits_mode')
GENERATION_KEYS = ('tokens', 'decode_history', 'chain_ancestry', 'target_sha256',
                   'producer_binary_sha256', 'producer_source_revision',
                   'client_source_sha256', 'hardware', 'kv_type', 'target_precision', 'tap_ids')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def hash_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p, 'absolute canonical nonsymlink path required: ' + str(p))
    return p


def fingerprint(path):
    p = canonical(path)
    s = p.lstat()
    require(stat.S_ISREG(s.st_mode), 'regular file required: ' + str(p))
    return {'path': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns,
            'ctime_ns': s.st_ctime_ns, 'device': s.st_dev, 'inode': s.st_ino}


def locator(pair):
    return {'path': str(canonical(pair[0])), 'sha256': pair[1]}


def unique_json(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON key: ' + k)
        result[k] = v
    return result


class Inputs:
    """Pin small controls and payload stat snapshots, rechecking before commit."""
    def __init__(self):
        self.controls = {}
        self.payloads = {}

    def read(self, pin):
        require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'} and hash_ok(pin['sha256']),
                'exact path/SHA256 locator required')
        p = canonical(pin['path'])
        before = fingerprint(p)
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            s = os.fstat(fd)
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and (s.st_dev, s.st_ino) == (before['device'], before['inode']),
                    'control changed while opening')
            with os.fdopen(fd, 'rb', closefd=False) as f:
                data = f.read()
        finally:
            os.close(fd)
        require(before == fingerprint(p) and digest(data) == pin['sha256'], 'control SHA/stat pin mismatch: ' + str(p))
        require(str(p) not in self.controls or self.controls[str(p)] == (pin, before), 'conflicting control pin')
        self.controls[str(p)] = (dict(pin), before)
        return data

    def json(self, pin):
        return json.loads(self.read(pin), object_pairs_hook=unique_json)

    def recheck(self):
        for pin, before in list(self.controls.values()):
            require(fingerprint(pin['path']) == before, 'control changed before publication')
            self.read(pin)
        for p, wanted in self.payloads.items():
            require(fingerprint(p) == wanted, 'payload stat changed before publication: ' + p)


def records(value, manifest_sha):
    require(set(value) == {'schema', 'manifest_sha256', 'source_sha256', 'artifacts'}
            and value['schema'] == 'block_data_completed_admission_v1'
            and value['manifest_sha256'] == manifest_sha and value['source_sha256'] == READER_SHA256,
            'completed admission source/manifest/schema mismatch')
    require(isinstance(value['artifacts'], list) and value['artifacts'], 'empty artifact inventory')
    result = {}
    for r in value['artifacts']:
        require(set(r) == RECORD_KEYS and hash_ok(r['sha256']) and r['path'] not in result,
                'malformed/duplicate artifact inventory')
        canonical(r['path'])
        require(all(type(r[k]) is int and r[k] >= 0 for k in STAT_KEYS - {'path'}), 'invalid stat identity')
        result[r['path']] = r
    return result



DEVICE_VIEW_PROFILE = 'original_trial_admission_device_view_v1'
OLD_TRIAL_ADMISSION_SHA256 = '1fb5938ccd34a8166537e4ed0e032ca83bd2e45daa6c795fffb035a62ab15c18'


def validate_device_admission_values(view, old, new, report):
    """Only a genuine sibling full audit may replace physical reader arguments."""
    require(view.get('schema') == DEVICE_VIEW_PROFILE
            and view.get('old_admission', {}).get('sha256') == OLD_TRIAL_ADMISSION_SHA256
            and view.get('old_device') == 51 and view.get('new_device') == 61,
            'exact original0410 device51-to61 view required')
    manifest_sha = view['manifest']['sha256']
    before, after = records(old, manifest_sha), records(new, manifest_sha)
    require(len(before) == len(after) == 68 and before.keys() == after.keys(),
            'complete singleton trial artifact inventory differs')
    require(view['old_admission']['path'] != view['new_admission']['path']
            and view['old_admission']['sha256'] != view['new_admission']['sha256'],
            'preserve original admission and use a genuine NEW audit')
    for path, record in before.items():
        require(record['device'] == 51 and after[path] == {**record, 'device': 61},
                'fresh audit changed more than recorded device identity')
    require(report.get('schema') == 'block_data_preparation_v1'
            and report.get('source_sha256') == READER_SHA256
            and report.get('manifest_sha256') == manifest_sha
            and report.get('manifest_path') == view['manifest']['path']
            and report.get('completed_admission') == view['new_admission']
            and report.get('family') == 'dspark'
            and report.get('profiles', {}).get('hard_ce_target_prefix') == 'PASS'
            and report.get('profiles', {}).get('exact_soft_teacher_target_prefix') == 'PASS',
            'genuine original-reader full audit report required')
    return after


def load_device_admission_view(pin, *, current_stats=True):
    inputs = Inputs();view = inputs.json(pin)
    old = inputs.json(view['old_admission'])  # Retain exact original digest comparison FIRST.
    new, report = inputs.json(view['new_admission']), inputs.json(view['audit_report'])
    inputs.json(view['manifest'])
    after = validate_device_admission_values(view, old, new, report)
    if current_stats:
        for path, record in after.items():
            wanted = {k:v for k,v in record.items() if k != 'sha256'}
            require(fingerprint(path) == wanted, 'current artifact changed after fresh full audit')
            inputs.payloads[path] = wanted
    inputs.recheck()
    return view



def device_admission_arguments(view, args, kwargs):
    """Translate exactly two reader arguments; all scientific settings survive."""
    if kwargs.get('admission_path') is None or str(Path(kwargs['admission_path'])) != view['old_admission']['path']:
        return kwargs
    require(kwargs.get('admission_sha256') == view['old_admission']['sha256'],
            'caller original trial admission digest differs')
    require(args and str(Path(args[0])) == view['manifest']['path']
            and kwargs.get('expected_sha256') == view['manifest']['sha256'],
            'view is restricted to the original pinned trial manifest')
    return {**kwargs,'admission_path':view['new_admission']['path'],
            'admission_sha256':view['new_admission']['sha256']}


def install_device_admission_view(block_data, pin):
    require(digest(Path(block_data.__file__).read_bytes()) == READER_SHA256,
            'original source937 BlockDataset required')
    view = load_device_admission_view(pin)
    original = block_data.BlockDataset.__init__
    def initialize(dataset, *args, **kwargs):
        admission = kwargs.get('admission_path')
        if admission is not None and str(Path(admission)) == view['old_admission']['path']:
            require(kwargs.get('admission_sha256') == view['old_admission']['sha256'],
                    'caller original trial admission digest differs')
            current = load_device_admission_view(pin)  # No cached trust or extra payload hash.
            require(current == view, 'pinned device admission view changed')
            kwargs = device_admission_arguments(view,args,kwargs)
        return original(dataset, *args, **kwargs)
    block_data.BlockDataset.__init__ = initialize
    return view


def refresh(args):
    inputs = Inputs()
    pins = {name: locator(getattr(args, name)) for name in
            ('old_publication', 'old_admission', 'new_admission', 'audit_report',
             'old_replay', 'join', 'seal', 'plan', 'stat_census')}
    output = canonical(args.output)
    require(not output.exists(), 'output already exists; preserve immutable history')
    canonical(output.parent)
    require(output.parent.is_dir(), 'output parent must exist')
    parent_stat = output.parent.stat()
    def check_output_parent():
        now = canonical(output.parent).stat()
        require(stat.S_ISDIR(now.st_mode) and (now.st_dev, now.st_ino)
                == (parent_stat.st_dev, parent_stat.st_ino), 'output parent replaced')
    lock = canonical(args.lock)
    fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        def check_lock():
            s, current = os.fstat(fd), lock.lstat()
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1
                    and (s.st_dev, s.st_ino, s.st_uid, stat.S_IMODE(s.st_mode))
                    == (args.lock_device, args.lock_inode, args.lock_uid, 0o644)
                    and (current.st_dev, current.st_ino, current.st_uid, current.st_mode, current.st_nlink)
                    == (s.st_dev, s.st_ino, s.st_uid, s.st_mode, s.st_nlink)
                    and canonical(lock) == lock, 'canonical lock identity/mode/owner changed')
        check_lock()
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_lock()
        publication, old, new, report, replay, join, seal, plan, census = (
            inputs.json(pins[n]) for n in pins)
        plan_sha, shard = publication['plan_sha256'], publication['shard']
        require(publication.get('schema') == 'block_shard_publication_v1' and publication.get('status') == 'PASS',
                'historical publication required')
        require(publication['admission'] == pins['old_admission']
                and publication['reconstruction_join'] == pins['join'], 'publication locators differ from external pins')
        manifest_pin = publication['manifest']
        manifest = inputs.json(manifest_pin)
        require(identity(plan) == plan_sha and plan['source'] == replay['source'], 'canonical plan/source mismatch')
        require(replay.get('schema') == 'block_shard_replay_admission_v1' and replay.get('status') == 'PASS'
                and replay['trial_publication'] == pins['old_publication']
                and replay['plan_sha256'] == plan_sha and replay['shard'] == shard
                and set(replay['domains']) == {'prose', 'code', 'reasoning'}, 'historical replay binding mismatch')
        expected = replay['original_identity']
        require(expected and expected == replay['replay_identity'] == seal['identity'] == join['original_identity']
                and set(expected) == set(plan['shards'][shard]), 'original tensor seal/membership mismatch')
        require(seal['schema'] == 'block_shard_original_tensor_seal_v1' and seal['plan_sha256'] == plan_sha
                and seal['shard'] == shard, 'seal schema/plan/shard mismatch')
        require(join['schema'] == 'block_shard_reconstruction_join_v1' and join['status'] == 'PASS'
                and join['plan_sha256'] == plan_sha and join['shard'] == shard
                and join['original_metadata'] == replay['original_metadata'] and join['producer'] == replay['producer']
                and join['tensor_identity_equal'] is True and join['historical_receipt_bytes_restored'] is True
                and set(join['fresh_receipts']) == set(expected) == set(join['measured_tensor_joins']),
                'reconstruction provenance mismatch')
        inputs.read(replay['producer'])
        source_root = canonical(args.source_root)
        require(plan['source']['module:src/w1a1_eagle/block_data.py'] == READER_SHA256,
                'frozen original reader required')
        capture_source_history = None
        for label, sha in plan['source'].items():
            if label.startswith('module:'):
                p = canonical(source_root / label.removeprefix('module:'))
                require(p.is_relative_to(source_root), 'source module escaped root')
                if label == HISTORICAL_CAPTURE_LABEL:
                    # This capture module produced historical ancestry; metadata
                    # refresh executes neither its old nor current version. Bind
                    # both explicitly, without rewriting the immutable plan.
                    require(sha == HISTORICAL_CAPTURE_SHA256
                            and args.historical_capture_source is not None,
                            'explicit original historical capture source required')
                    historical = locator(args.historical_capture_source)
                    require(historical['sha256'] == sha and historical['path'] != str(p),
                            'historical capture locator must bind original plan bytes separately')
                    current = {'path': str(p), 'sha256': CURRENT_CAPTURE_SHA256}
                    inputs.read(historical)
                    inputs.read(current)
                    capture_source_history = {
                        'plan_source_label': label,
                        'historical': historical,
                        'historical_role': 'original capture/data ancestry; never executed by refresh',
                        'current_authorized_source': current,
                        'current_authorized_git_object': CURRENT_CAPTURE_GIT_OBJECT,
                        'current_role': 'authorized original S stage; never executed by refresh',
                        'reason': 'metadata-only refresh reads original audit/replay evidence; retains historical capture pin',
                        'plan_source_pin_unchanged': True,
                    }
                else:
                    inputs.read({'path': str(p), 'sha256': sha})
        require(args.historical_capture_source is None or capture_source_history is not None,
                'historical capture override applies only to the declared capture module')
        require(replay['producer']['sha256'] == plan['source']['module:scripts/replay_dspark_shard.py'],
                'replay producer source mismatch')
        inputs.json(publication['request'])
        metadata = inputs.json(replay['original_metadata'])
        require(metadata['schema'] == 'block_shard_original_metadata_v1' and metadata['plan_sha256'] == plan_sha
                and metadata['shard'] == shard and metadata['identity'] == expected, 'original metadata mismatch')
        for pin in metadata['files'].values():
            inputs.read(pin)
        old_records, new_records = records(old, manifest_pin['sha256']), records(new, manifest_pin['sha256'])
        require(old_records.keys() == new_records.keys(), 'complete old/new artifact inventory differs')
        require(type(args.old_device) is int and type(args.new_device) is int and args.old_device != args.new_device,
                'explicit distinct device transition required')
        require(census['schema'] == 'bounded_admission_stat_census_v1' and census['manifest'] == manifest_pin
                and census['admission'] == {**pins['old_admission'], 'manifest_sha256': manifest_pin['sha256'],
                                           'source_sha256': READER_SHA256}
                and census['publication_directory'] == publication['directory']
                and census['record_count'] == len(old_records), 'census original bindings mismatch')
        changed = set()
        for entry in census['mismatches']:
            path = entry['path']
            require(path in old_records and path not in changed and entry['mismatched_fields'] == ['device'],
                    'census undeclared/nondevice/duplicate change')
            before = {k: v for k, v in old_records[path].items() if k not in ('path', 'sha256')}
            require(before['device'] == args.old_device and entry['recorded'] == before
                    and entry['actual'] == {**before, 'device': args.new_device}, 'census stat transition mismatch')
            changed.add(path)
        require(changed and census['mismatch_count'] == len(changed)
                and census['matching_record_count'] == len(old_records) - len(changed)
                and census['mismatch_counts_by_field'] == {'device': len(changed)}
                and census['device_pairs'] == {str((args.old_device, args.new_device)): len(changed)},
                'census complete counts mismatch')
        for path, record in old_records.items():
            wanted = {**record, 'device': args.new_device} if path in changed else record
            require(new_records[path] == wanted, 'new admission permits only declared device changes')
            fp = {k: v for k, v in wanted.items() if k != 'sha256'}
            require(fingerprint(path) == fp, 'current artifact stat mismatch')
            inputs.payloads[path] = fp
        inventory = {}
        base = canonical(manifest_pin['path']).parent
        def artifact(record):
            require(isinstance(record, dict) and set(record) in ({'path', 'sha256'}, {'path', 'sha256', 'shape', 'dtype'})
                    and hash_ok(record['sha256']), 'invalid manifest artifact locator')
            p = canonical(base / record['path'])
            pin = {'path': str(p), 'sha256': record['sha256']}
            require(str(p) not in inventory or inventory[str(p)] == pin['sha256'], 'conflicting payload SHA')
            inventory[str(p)] = pin['sha256']
            require(str(p) in new_records and new_records[str(p)]['sha256'] == pin['sha256'],
                    'manifest/ancestry artifact absent or SHA differs from admission')
            return pin
        inputs.json(artifact(manifest['producer']['receipt']))
        inputs.json(artifact(manifest['train_inventory']))
        require(all(manifest.get(k) == v for k, v in plan['geometry'].items()), 'manifest geometry mismatch')
        actual_identity, domains = {}, set()
        for chain in manifest['chains']:
            cid = chain['chain_id']
            require(cid in expected and cid not in actual_identity, 'manifest duplicate/unknown chain')
            require(all(chain[k] == plan['chains'][cid][k] for k in ('prompt_id', 'prompt_sha256', 'domain', 'split')),
                    'original chain source/roles mismatch')
            domains.add(chain['domain'])
            payload = {k: artifact(chain[k]) for k in ('tokens', 'features', 'logits', 'native_receipt')}
            actual_identity[cid] = {'artifacts': {k: v['sha256'] for k, v in payload.items()},
                                    'anchors': chain['anchors'], 'prompt_length': chain['prompt_length'],
                                    'logits_indices': chain.get('logits_indices')}
            native = inputs.json(payload['native_receipt'])
            fresh = inputs.json(join['fresh_receipts'][cid])
            require(all(native.get(k) == fresh.get(k) for k in NATIVE_KEYS), 'historical/fresh native semantics mismatch')
            require(all(fresh['files'][k]['sha256'] == payload[k]['sha256'] for k in ('features', 'logits')),
                    'fresh native tensor SHA mismatch')
            require(join['measured_tensor_joins'][cid] == {k + '_sha256': payload[k]['sha256']
                                                         for k in ('tokens', 'features', 'logits')}, 'measured tensor joins mismatch')
            if 'generation_source' in native:
                generation = inputs.json(artifact(native['generation_source']))
                require(generation.get('schema') == 'block_native_teacher_request_v1'
                        and generation.get('optimizer_updates') == 0 and generation.get('complete') is True
                        and generation.get('logits_mode') == 'none'
                        and generation.get('teacher_context_reset_between_requests') is True
                        and all(generation.get(k) == native.get(k) for k in GENERATION_KEYS), 'generation ancestry mismatch')
                for file in generation['files'].values():
                    artifact({'path': file['path'], 'sha256': file['sha256']})
        require(actual_identity == expected and domains == {'prose', 'code', 'reasoning'}, 'complete chain/seal/domain mismatch')
        require(inventory == {p: r['sha256'] for p, r in new_records.items()}, 'complete reader artifact inventory mismatch')
        require(report['schema'] == 'block_data_preparation_v1' and report['manifest_sha256'] == manifest_pin['sha256']
                and report['manifest_path'] == manifest_pin['path'] and report['completed_admission'] == pins['new_admission']
                and report['source_sha256'] == READER_SHA256 and report['family'] == manifest['family']
                and report['layout'] == manifest['layout']
                and report['producer_receipt_sha256'] == manifest['producer']['receipt']['sha256']
                and report['storage']['chains'] == len(expected)
                and report['storage']['blocks'] == sum(len(c['anchors']) for c in manifest['chains'])
                and report['profiles']['hard_ce_target_prefix'] == 'PASS'
                and report['profiles']['exact_soft_teacher_target_prefix'] == 'PASS', 'actual full-audit report bindings mismatch')
        refreshed = {**publication, 'admission': pins['new_admission']}
        pub_bytes = (json.dumps(refreshed, indent=2, sort_keys=True) + '\n').encode()
        pub_pin = {'path': str(output / 'publication.json'), 'sha256': digest(pub_bytes)}
        generator = {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).read_bytes())}
        inputs.read(generator)
        provenance = {'schema': 'block_replay_publication_admission_refresh_v1',
                      'inputs': pins, 'publication': pub_pin, 'generator': generator,
                      'capture_source_history': capture_source_history,
                      'owner': args.owner, 'plan_sha256': plan_sha, 'shard': shard,
                      'admission_change_only': True, 'new_native_captures': 0,
                      'historical_capture_provenance_retained': True,
                      'artifact_count': len(inventory), 'changed_device_paths': sorted(changed),
                      'old_device': args.old_device, 'new_device': args.new_device,
                      'lock': {'path': str(lock), 'device': args.lock_device, 'inode': args.lock_inode,
                               'uid': args.lock_uid, 'mode': '0644'},
                      'audit_trust': 'externally pinned genuine original-reader full audit; not rerun here',
                      'consumer_acceptance': 'PENDING: original-source admit-replay and validate_replay_admission'}
        require(isinstance(args.owner, str) and args.owner.strip(), 'explicit owner required')
        inputs.recheck()
        check_lock()
        # mkdir is exclusive. Readers must use the returned pins, never discover
        # partial files. Failure retains evidence; an existing directory is never
        # overwritten. This is not a whole-filesystem compare-and-swap.
        check_output_parent()
        output.mkdir(mode=0o755)
        output_stat = output.stat()
        def check_output():
            check_output_parent()
            now = canonical(output).stat()
            require(stat.S_ISDIR(now.st_mode) and (now.st_dev, now.st_ino)
                    == (output_stat.st_dev, output_stat.st_ino), 'output directory replaced')
        for name, data in [('publication.json', pub_bytes), ('provenance.json',
                           (json.dumps(provenance, indent=2, sort_keys=True) + '\n').encode())]:
            check_output()
            temp = output / ('.' + name + '.tmp')
            with temp.open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            inputs.recheck()
            check_lock()
            check_output()
            os.rename(temp, output / name)
        inputs.recheck()
        check_lock()
        check_output()
        return {'publication': pub_pin, 'provenance': {'path': str(output / 'provenance.json'),
                                                     'sha256': digest((output / 'provenance.json').read_bytes())}}
    finally:
        os.close(fd)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old-publication', 'old-admission', 'new-admission', 'audit-report',
                 'old-replay', 'join', 'seal', 'plan', 'stat-census'):
        p.add_argument('--' + name, nargs=2, metavar=('PATH', 'SHA256'), required=True)
    p.add_argument('--historical-capture-source', nargs=2, metavar=('PATH', 'SHA256'),
                   help='separately retained exact historical capture module; required when present in frozen plan')
    for name in ('source-root', 'lock', 'output', 'owner'):
        p.add_argument('--' + name, required=True)
    for name in ('lock-device', 'lock-inode', 'lock-uid', 'old-device', 'new-device'):
        p.add_argument('--' + name, type=int, required=True)
    return p


def main():
    args = parser().parse_args()
    try:
        print(json.dumps(refresh(args), sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as e:
        raise SystemExit('replay publication refresh refused: ' + str(e)) from e


if __name__ == '__main__':
    main()
