"""Revalidate the affected retained state before resuming this boot's serial queue."""
import argparse
import json
import os
import shutil
import subprocess

from .common import R, ROOT, identity, pause, pin, save
from .checkpoint import validate
from .post_primary import run, wait_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', required=True, type=__import__('pathlib').Path)
    args = parser.parse_args()
    wait_release(args.after)
    pause()
    structure_path = R / 'qualification/HE6R2-post-reclaim-structure.json'
    structure = json.loads(structure_path.read_text())
    if (structure['status'] != 'CPU_MASTER_ADAM_RNG_CURSOR_STRUCTURE_PASS'
            or structure['owner']['boot_id'] != identity()['boot_id']):
        raise ValueError('Current-boot RAM structure qualification required')
    recovery = json.loads((R / 'arms/HE6R2/recovery.json').read_text())
    actual = pin(recovery['archive'])
    if actual != structure['checkpoint']:
        raise ValueError('Retained state changed after RAM qualification')
    validate(recovery['archive'], recovery['raw_sha256'], recovery['raw_bytes'])
    hardware = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'], text=True)
    apps = subprocess.check_output([
        '/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True)
    if apps.strip():
        raise RuntimeError('Another GPU owner is present: ' + apps)
    free = {p: shutil.disk_usage(p).free for p in ('/', '/mnt/c')}
    if min(free.values()) < 4_800_000_000:
        raise RuntimeError('Insufficient reclaimed headroom: ' + repr(free))
    if (R / 'arms/Csmall').exists():
        raise ValueError('Csmall setup/cursor must be reconciled before launch')
    save(R / 'post-reclaim-owner.json', dict(
        status='POST_RECLAIM_CHECKPOINT_STREAM_AND_RAM_STATE_PASS', owner=identity(),
        predecessor=pin(args.after), checkpoint=actual, structure=pin(structure_path),
        raw_sha256=recovery['raw_sha256'], raw_bytes=recovery['raw_bytes'],
        free_bytes=free, hardware=hardware, source=pin(__file__),
        prior_stream_audit=pin(R / 'second-reboot-audit-20261105-01/result.json'),
        policy='Previous boots are terminal by reboot, not clean success. No retired exploratory checkpoints recreated. CPU proof is not GPU resume qualification. All remaining training and evaluation uses one serial lane.'))
    print('POST_RECLAIM_CHECKPOINT_STREAM_AND_RAM_STATE_PASS', flush=True)
    run('recovery_combinations', '--after', args.after)


if __name__ == '__main__':
    main()
