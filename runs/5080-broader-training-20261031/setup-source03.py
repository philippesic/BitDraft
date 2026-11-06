"""CPU-only preparation. Does not admit, launch, or qualify a fresh restart."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
S2 = R / 'source02'
S3 = R / 'source03'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pin(p):
    return {'path': str(p), 'sha256': sha(p)}


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


assert len(sys.argv) == 3
assert sha(S2 / 'manifest.json') == sys.argv[1]
assert sha(R / 'fresh_base.py') == sys.argv[2]
original_manifest = json.loads((S2 / 'manifest.json').read_text())
for entry in original_manifest['files']:
    assert sha(S2 / entry['path']) == entry['sha256']
assert not S3.exists()
# Preserve the source-layout symlinks; never recurse into retained runs/models.
shutil.copytree(S2, S3, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
for entry in original_manifest['files']:
    assert sha(S3 / entry['path']) == entry['sha256']
shutil.copyfile(R / 'fresh_base.py', S3 / 'scripts/training_system/fresh_base.py')
shutil.copyfile(R / 'controller02/lane.py', S3 / 'scripts/training_system/lane.py')
shutil.copyfile(R / 'controller02/guard.py', S3 / 'scripts/shortlist_pilots/guard.py')
p = S3 / 'scripts/training_system/train.py'
s = p.read_text()
s = replace_once(s, '    validate_options(spec)\n',
                 "    validate_options(spec)\n    if spec.get('fresh_initializer'):\n        from training_system.fresh_base import validate as validate_fresh\n        validate_fresh(spec)\n")
s = replace_once(s, "    if spec.get('paired_plugin',{}).get('continuation_contract'):\n",
                 "    if spec.get('fresh_initializer'):\n        from training_system.fresh_base import load as load_fresh\n        model,optimizer,baseline,metadata=load_fresh(setup,spec)\n    elif spec.get('paired_plugin',{}).get('continuation_contract'):\n")
s = replace_once(s, "    if not spec.get('model_cache'):\n        from training_system.model_cache import validate_frozen\n",
                 "    if not spec.get('model_cache') and not spec.get('fresh_initializer'):\n        from training_system.model_cache import validate_frozen\n")
s = replace_once(s, "        phases['full_job_wall'] = time.perf_counter()-invocation_start\n",
                 "        if spec.get('fresh_initializer'):\n            fresh=metadata['fresh_initializer']\n            result['exposure'].update(inherited_initializer={k:fresh[k] for k in ['released_model','native_model','ordinary_donor']},inherited_immediate_branch_updates=0,inherited_immediate_branch_TRAIN_seconds=0.)\n        phases['full_job_wall'] = time.perf_counter()-invocation_start\n")
p.write_text(s)
p = S3 / 'scripts/training_system/describe.py'
s = p.read_text()
anchor = '    # Match the durable JSON representation, including tuple options and integer\n'
addition = """    if metadata.get('fresh_initializer'):
        fresh=metadata['fresh_initializer']
        value['initializer']={'fresh_initializer':spec['fresh_initializer'],'released_model':fresh['released_model'],'native_model':fresh['native_model'],'ordinary_donor':fresh['ordinary_donor'],'base_optimizer_steps':optimizer._lane_base_steps,'optimizer_imported':False,'cursor_imported':False,'paid_checkpoint_loaded':False,'rng_seed':42,'guard_anchor':'fresh fork zero'}
        value['comparison']['initializer']={'released_model':fresh['released_model']['sha256'],'native_model':fresh['native_model']['sha256'],'ordinary_donor':fresh['ordinary_donor']['sha256'],'all48_fresh':True,'optimizer_imported':False,'cursor_imported':False,'rng_seed':42}
        value['inputs']=[spec['fresh_initializer'],fresh['released_model'],fresh['native_model'],fresh['ordinary_donor'],fresh['architecture_contract'],data.input_pin,data.order_pin]
"""
s = replace_once(s, anchor, addition + anchor)
p.write_text(s)
manifest = json.loads((S2 / 'manifest.json').read_text())
manifest.update(base_manifest=pin(S2 / 'manifest.json'),
                change='Conditional full48 released-BF16 restart factory and explicit ancestry, unchanged production step/loss/checkpoint guards; operational lane/guard bytes from already tested controller02. UNQUALIFIED: no CUDA work launched by this preparation.')
manifest['files'].append({'path': 'scripts/training_system/fresh_base.py'})
for f in manifest['files']:
    p = S3 / f['path']
    f.update(bytes=p.stat().st_size, sha256=sha(p))
(S3 / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
diffs = {}
for relative in ['scripts/training_system/train.py', 'scripts/training_system/describe.py',
                 'scripts/training_system/lane.py', 'scripts/shortlist_pilots/guard.py']:
    result = subprocess.run(['diff', '-u', str(S2 / relative), str(S3 / relative)],
                            capture_output=True, text=True)
    assert result.returncode in [0, 1]
    diffs[relative] = result.stdout
receipt = {'status': 'PREPARED_UNQUALIFIED_NOT_LAUNCHED', 'source': pin(S3 / 'manifest.json'),
           'factory': pin(S3 / 'scripts/training_system/fresh_base.py'), 'diffs': diffs}
with (R / 'source03-preparation.json').open('x') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps({k: v for k, v in receipt.items() if k != 'diffs'}))
