from pathlib import Path
import hashlib, json, os, shutil, subprocess, time, uuid

P = Path('/home/philip/binary-eagle-decoding')
R = P/'runs/w1a8-modern-protected-20261103-01'
C = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts')
roots = [C/'w1a8-head-resident-reference4320-01', C/'w1a8-selected-head-false4320-01', C/'w1a8-added-replay-pair-20261103-01/false4320']
models = [root/'model.gguf' for root in roots]
base = P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf'
def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)
def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
started = time.monotonic(); before = shutil.disk_usage(C).free
assert json.loads((R/'natural-replay-confirmation01/quiet-window.json').read_text())['status'] == 'COMPLETE_ROOT_RAW_AUDITED_RELEASED'
for proof in ['head-false4320-parity02.json', 'added-replay-false4320-parity.json']:
    assert json.loads((R/proof).read_text())['all_exact']
results = [json.loads((root/'result.json').read_text()) for root in roots]
expected = results[0]['model']; base_pin = pin(base)
assert base_pin['sha256'] == '6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048'
assert expected['sha256'] == '8b6161eec9d93f823835350ccecb25becb418a23c91fa7ea976f6a506ab176d9'
identities = [model.stat() for model in models]
assert len({(s.st_dev,s.st_ino) for s in identities}) == 1 and identities[0].st_nlink == 3
checkpoints = [result['checkpoint'] for result in results]
for model, result, checkpoint in zip(models, results, checkpoints):
    assert not model.is_symlink() and model.stat().st_uid == os.getuid()
    assert pin(model) == result['model'] and pin(checkpoint['path']) == checkpoint
active = json.loads((R/'natural-replay-confirmation01/models.json').read_text())
assert not {str(m) for m in models} & {a['model']['path'] for a in active['arms'] if a['model']}
for process in Path('/proc').iterdir():
    if not process.name.isdigit(): continue
    try:
        for fd in (process/'fd').iterdir():
            try:
                s = fd.stat()
                assert (s.st_dev,s.st_ino) != (identities[0].st_dev,identities[0].st_ino), 'export has live file handle'
            except (OSError, FileNotFoundError): pass
    except (OSError, FileNotFoundError): pass
folder = roots[0]/'compact-native-export'; folder.mkdir(exist_ok=False)
ranges = []; cursor = offset = 0
with base.open('rb') as left, models[0].open('rb') as right, (folder/'native-scales.bin').open('xb') as out:
    while True:
        a,b = left.read(4096),right.read(4096); assert len(a)==len(b)
        if not a: break
        if a != b:
            out.write(b); ranges.append(dict(model_offset=offset,payload_offset=cursor,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())); cursor += len(b)
        offset += len(a)
    out.flush(); os.fsync(out.fileno())
assert 0 < cursor < 1024**2
manifest = dict(schema='lossless_w1a8_native_export_chunk_archive_v1',base=base_pin,original=expected,ranges=ranges,payload_sha256=pin(folder/'native-scales.bin')['sha256'],paid_checkpoints=checkpoints,reason='Completed4320 migration references superseded by5760 controls. Preserve all paid states and exact export bytes; archive three aliases of one derived materialization only.')
write(folder/'manifest.json',manifest); shutil.copyfile(R/'archive-head-export.py',folder/'restore.py')
scratch = Path('/dev/shm')/('w1a8-4320-roundtrip-'+uuid.uuid4().hex)
try:
    subprocess.run([str(P/'.venv/bin/python'),str(folder/'restore.py'),'restore',str(folder/'manifest.json'),str(scratch)],check=True,stdout=subprocess.DEVNULL)
    restored=pin(scratch); assert restored['sha256']==expected['sha256'] and restored['bytes']==expected['bytes']
finally: scratch.unlink(missing_ok=True)
receipts=[]
for model,result,checkpoint in zip(models,results,checkpoints):
    receipt=dict(status='EXACT_COMPLETED4320_EXPORT_ARCHIVE_ROUNDTRIP_PASS',original=result['model'],archive_manifest=pin(folder/'manifest.json'),paid_checkpoint_unchanged=checkpoint,all_unique_export_bytes_preserved=True,payload_bytes=cursor,restore_command=[str(P/'.venv/bin/python'),str(folder/'restore.py'),'restore',str(folder/'manifest.json'),str(model)])
    write(model.with_suffix('.gguf.archived.json'),receipt);receipts.append(receipt)
for model,result,checkpoint in zip(models,results,checkpoints):
    assert pin(model)==result['model'] and pin(checkpoint['path'])==checkpoint
assert pin(base)==base_pin
for model in models: model.unlink()
report=dict(status='THREE_COMPLETED4320_ALIASES_LOSSLESSLY_ARCHIVED',receipts=receipts,before_free_bytes=before,after_free_bytes=shutil.disk_usage(C).free,paid_checkpoints_deleted=0,protected_live_and5760_model_paths_unchanged=True,wall_seconds=time.monotonic()-started)
write(R/'completed4320-export-archives.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='receipts'}))
