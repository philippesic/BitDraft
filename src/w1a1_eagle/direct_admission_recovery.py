"""Explicit current-boot ORIGINAL completed-admission arguments, not admission issuance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

READER = 'd8531872a1641c7b35e822107a3186abf9e16542cabbe83e6c0dfc0bdfcd5544'
PROFILE = 'original_completed_admission_current_boot_view_v1'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def helper(record):
    raw = Path(record['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == record['sha256'], 'existing admission helper source differs')
    spec = importlib.util.spec_from_file_location('existing_admission_refresh', record['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.READER_SHA256 == READER, 'original reader helper differs')
    return module


def validate_values(view, entry, old, new, report, tools):
    manifest = entry['manifest']
    before, after = tools.records(old, manifest['sha256']), tools.records(new, manifest['sha256'])
    require(before.keys() == after.keys(), 'complete original artifact inventory differs')
    require(entry['old_admission']['path'] != entry['new_admission']['path'], 'genuine sibling audit required; preserve original')
    changed = []
    for path, record in before.items():
        actual = after[path]
        require(all(actual[key] == value for key, value in record.items() if key != 'device'),
                'fresh original audit changed nondevice field or expected SHA')
        if actual['device'] != record['device']:
            changed.append({'path': path, 'old_device': record['device'], 'new_device': actual['device']})
    require(changed == entry['device_changes'], 'explicit device-change census differs')
    require(report.get('schema') == 'block_data_preparation_v1'
            and report.get('source_sha256') == READER
            and report.get('manifest_sha256') == manifest['sha256']
            and report.get('manifest_path') == manifest['path']
            and report.get('completed_admission') == entry['new_admission']
            and report.get('family') == 'dspark'
            and report.get('profiles', {}).get('hard_ce_target_prefix') == 'PASS'
            and report.get('profiles', {}).get('exact_soft_teacher_target_prefix') == 'PASS',
            'genuine original full audit report differs')
    return after


def load(record, tools, *, selected=None):
    inputs = tools.Inputs()
    view = inputs.json(record)
    require(view.get('schema') == PROFILE and view.get('reader_sha256') == READER
            and view.get('boot_id') == boot_id(), 'exact current boot/reader view required')
    entries = view.get('entries')
    require(isinstance(entries, list) and entries, 'explicit nonempty original admission allowlist required')
    seen = set()
    for entry in entries:
        for role in ('manifest', 'old_admission', 'new_admission', 'audit_report'):
            locator = entry[role]
            require(isinstance(locator, dict) and set(locator) == {'path', 'sha256'}
                    and tools.hash_ok(locator['sha256']), 'exact selector locator required')
            tools.canonical(locator['path'])
        require(isinstance(entry['device_changes'], list), 'explicit device census required')
        key = (entry['old_admission']['path'], entry['old_admission']['sha256'],
               entry['manifest']['path'], entry['manifest']['sha256'])
        require(key not in seen, 'duplicate original admission/manifest view')
        seen.add(key)
        # Installing a selector grants no data validity. Retired entries remain
        # structurally authenticated, but only an EXACT consumed tuple reads
        # its old/new audit controls and current payload stat identities.
        if selected is None or key != selected:
            continue
        old = inputs.json(entry['old_admission'])  # Original digest first.
        new = inputs.json(entry['new_admission'])
        report = inputs.json(entry['audit_report'])
        inputs.json(entry['manifest'])
        after = validate_values(view, entry, old, new, report, tools)
        for path, value in after.items():
            expected = {k: v for k, v in value.items() if k != 'sha256'}
            require(tools.fingerprint(path) == expected, 'current artifact differs from genuine audit')
            inputs.payloads[path] = expected
    require(selected is None or selected in seen, 'selected admission absent from allowlist')
    inputs.recheck()
    return view


def install(block_data, record, existing_helper):
    tools = helper(existing_helper)
    source = Path(block_data.__file__).read_bytes()
    require(hashlib.sha256(source).hexdigest() == READER, 'original BlockDataset reader source required')
    original = block_data.BlockDataset.__init__
    top = compile(source, str(block_data.__file__), 'exec', dont_inherit=True)
    cls = next(c for c in top.co_consts if isinstance(c, types.CodeType) and c.co_name == 'BlockDataset')
    expected = next(c for c in cls.co_consts if isinstance(c, types.CodeType) and c.co_name == '__init__')
    require(isinstance(original, types.FunctionType) and original.__globals__ is block_data.__dict__
            and original.__code__ == expected, 'original reader constructor callable differs')
    view = load(record, tools)
    entries = {(entry['old_admission']['path'], entry['old_admission']['sha256'],
                entry['manifest']['path'], entry['manifest']['sha256']): entry
               for entry in view['entries']}

    def initialize(dataset, *args, **kwargs):
        admission = kwargs.get('admission_path')
        key = (str(Path(admission)), kwargs.get('admission_sha256'),
               str(Path(args[0])) if args else None, kwargs.get('expected_sha256')) if admission is not None else None
        entry = entries.get(key)
        if entry is not None:
            require(kwargs.get('admission_sha256') == entry['old_admission']['sha256'], 'caller original admission SHA differs')
            require(args and str(Path(args[0])) == entry['manifest']['path']
                    and kwargs.get('expected_sha256') == entry['manifest']['sha256'], 'exact original manifest required')
            require(load(record, tools, selected=key) == view, 'current boot view changed')
            kwargs = {**kwargs, 'admission_path': entry['new_admission']['path'],
                      'admission_sha256': entry['new_admission']['sha256']}
        return original(dataset, *args, **kwargs)
    block_data.BlockDataset.__init__ = initialize
    return view
