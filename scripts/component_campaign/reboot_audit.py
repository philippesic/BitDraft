"""Read-only post-reboot verification; never hydrate, train, reclaim or adopt jobs."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .archive import blocks
from .common import B, R, ROOT, identity, pin, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    owner = identity()
    save(args.out / 'owner.json', dict(owner=owner, source=pin(__file__),
         scope='CPU-only streaming reads; no hydration, training, deletion or signalling.'))
    prior = json.loads((R / 'restart-reconciliation.json').read_text())
    originals = {row['original']['path']: row['original'] for row in prior['archives']}
    checked, streams, failures, history_damage = {}, [], [], []
    for receipt in sorted((R / 'arms').glob('*/model-*.gguf.slices.json')):
        try:
            record = json.loads(receipt.read_text())
            for ref in [record['archive'], *record['dependencies']]:
                if ref['path'] not in checked:
                    checked[ref['path']] = pin(ref['path'])
                if checked[ref['path']] != ref:
                    raise ValueError('dependency differs: ' + ref['path'])
            original = record['original']
            if originals.get(original['path']) != original:
                raise ValueError('original differs from prior authenticated reboot inventory')
            digest, size = hashlib.sha256(), 0
            for block in blocks(record):
                digest.update(block)
                size += len(block)
            if digest.hexdigest() != original['sha256'] or size != original['bytes']:
                raise ValueError('reconstructed native bytes differ')
            link = record.get('repacked_from', {}).get('receipt')
            if link:
                try:
                    if pin(link['path']) != link:
                        raise ValueError('prior repack receipt differs')
                    if json.loads(Path(link['path']).read_text())['original'] != original:
                        raise ValueError('prior repack original differs')
                except (OSError, ValueError) as exc:
                    history_damage.append(dict(receipt=str(receipt), link=link, error=str(exc),
                        recovery='Current complete native stream matches the original identity in the prior reboot audit; damaged history is preserved verbatim.'))
            streams.append(dict(receipt=pin(receipt), original=original,
                                storage_version=record.get('storage_version', 0)))
            print('NATIVE_STREAM_PASS', receipt.parent.name, receipt.name, flush=True)
        except Exception as exc:
            failures.append(dict(receipt=str(receipt), error=repr(exc)))
            print('NATIVE_STREAM_FAIL', receipt, repr(exc), flush=True)
    complete = []
    for path in sorted((R / 'arms').glob('*/result.json')):
        data = json.loads(path.read_text())
        if data['status'] == 'COMPLETE' and data['progress']['updates'] == 4096:
            reused = bool(data.get('reused_training') or data.get('training_reused') or path.parent.name in ('H6', 'E6'))
            complete.append(dict(arm=path.parent.name, physical_window=not reused,
                                 result=pin(path), progress=data['progress']))
    damaged_json = []
    for path in sorted(R.rglob('*.json')):
        try:
            json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            damaged_json.append(dict(file=pin(path), error=str(exc)))
    csmall = R / 'arms/Csmall'
    csmall_files = {str(p.relative_to(csmall)): pin(p) for p in sorted(csmall.rglob('*')) if p.is_file()}
    progress_files = [name for name in ('resolved.json', 'updates.jsonl', 'readouts.json', 'live.json', 'recovery.json', 'result.json') if (csmall / name).exists()]
    checkpoints = []
    # Validate archive digest and full decoded torch stream without a disk temporary.
    from .checkpoint import validate
    for path in (B / 'checkpoint.pt.bsz', Path('/mnt/c/binary-eagle-component-20261105-01/HE6R2-4096.pt.bsz')):
        try:
            metadata = json.loads(path.with_suffix('.json').read_text())
            actual = pin(path)
            if actual['sha256'] != metadata['archive_sha256'] or actual['bytes'] != metadata['archive_bytes']:
                raise ValueError('checkpoint archive differs')
            validate(path, metadata['raw_sha256'], metadata['raw_bytes'])
            checkpoints.append(dict(archive=actual, receipt=pin(path.with_suffix('.json')), status='FULL_DECODED_STREAM_DIGEST_PASS'))
            print('CHECKPOINT_STREAM_PASS', path.name, flush=True)
        except Exception as exc:
            failures.append(dict(checkpoint=str(path), error=repr(exc)))
    predecessors = {}
    for name in ('component-restart-recovery-20261105-01', 'component-restart-selection-20261105-01'):
        path = ROOT / 'runs' / name / 'state.json'
        predecessors[name] = dict(receipt=pin(path), state=json.loads(path.read_text()),
            disposition='Stale previous-boot receipt; unchanged, never adopted or labelled clean success.')
    free = {p: shutil.disk_usage(p).free for p in ('/', '/mnt/c')}
    required = 4_800_000_000
    result = dict(status='ARTIFACT_FAILURE_REQUIRES_RECOVERY' if failures else 'NATIVE_AND_CHECKPOINT_BYTES_VERIFIED_METADATA_DAMAGE_PRESERVED',
        owner=owner, source=pin(__file__), prior_reconciliation=pin(R / 'restart-reconciliation.json'),
        native_streams=streams, authenticated_dependencies=len(checked), failures=failures,
        damaged_repack_history=history_damage, invalid_json=damaged_json, checkpoints=checkpoints,
        completed_windows=complete, completed_physical_windows=sum(x['physical_window'] for x in complete),
        csmall=dict(files=csmall_files, progress_files=progress_files,
                    disposition='No durable optimizer progress evidenced; preserve setup before a new attempt.' if not progress_files else 'Explicit exact-cursor reconciliation required.'),
        predecessors=predecessors, free_bytes=free,
        training_preflight=dict(required_host_free_bytes=required, host_deficit_bytes=max(0, required-free['/mnt/c']),
                                qualified=free['/mnt/c'] >= required),
        hardware=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'], text=True),
        policy='No paid states, source, malformed evidence or migration data deleted or rewritten. No GPU work. Byte authentication is not a full optimizer/native-resume qualification or final campaign comparison.')
    save(args.out / 'result.json', result)
    print('REBOOT_AUDIT_COMPLETE', result['status'], len(streams), 'streams', len(failures), 'failures', flush=True)
    if failures:
        raise RuntimeError('Artifact failures preserved in audit result')


if __name__ == '__main__':
    main()
