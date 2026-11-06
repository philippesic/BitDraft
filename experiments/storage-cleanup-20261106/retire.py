"""Execute only the explicitly recorded November 6 checkpoint retirement plan.

This is a one-off receipt-backed cleanup, not an automatic retention policy.
It deliberately cannot select new paths or recursively delete directories.
"""
import json
import os
import shutil
import stat
from pathlib import Path

ROOT = Path('/home/philip/binary-eagle-decoding')
RUN = ROOT / 'runs/storage-cleanup-20261106-01'


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink)


def main():
    plan = json.loads((RUN / 'plan.json').read_text())
    kept = json.loads((RUN / 'protected-before.json').read_text())
    assert kept['status'] == 'ALL_FIVE_KEEP_STATES_HASH_PASS'
    protected = set()
    for entry in kept['records']:
        info = Path(entry['path']).stat()
        assert (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == (
            entry['device'], entry['inode'], entry['bytes'], entry['mtime_ns'])
        protected.add((info.st_dev, info.st_ino))
    for name in ('plan.json', 'protected-before.json'):
        with (RUN / name).open('rb') as stream:
            os.fsync(stream.fileno())
    journal = RUN / 'deleted.jsonl'
    assert not journal.exists(), 'Do not repeat a completed or partial retirement.'
    with journal.open('x') as log:
        directory = os.open(RUN, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        for entry in plan['candidates']:
            path = Path(entry['path'])
            assert path.is_absolute() and not path.is_symlink()
            expected = tuple(entry[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'nlink'))
            assert expected[:2] not in protected
            with_directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                info = os.stat(path.name, dir_fd=with_directory, follow_symlinks=False)
                assert stat.S_ISREG(info.st_mode) and identity(info) == expected
                log.write(json.dumps(dict(status='ABOUT_TO_RETIRE', **entry)) + '\n')
                log.flush()
                os.fsync(log.fileno())
                os.unlink(path.name, dir_fd=with_directory)
                os.fsync(with_directory)
            finally:
                os.close(with_directory)
            log.write(json.dumps(dict(status='RETIRED', **entry)) + '\n')
            log.flush()
            os.fsync(log.fileno())
            print('RETIRED', entry['storage'], entry['bytes'], path.name, flush=True)
    result = dict(
        status='ALL_34_SELECTED_OPTIMIZER_STATES_RETIRED',
        files=len(plan['candidates']),
        bytes_by_storage={kind: sum(x['bytes'] for x in plan['candidates'] if x['storage'] == kind)
                          for kind in ('WSL', 'WindowsC')},
        free_before=plan['before_free'],
        free_after={str(p): shutil.disk_usage(p).free for p in ('/', '/mnt/c')},
        policy='Selected superseded histories intentionally retired, not archived. '
               'Native exports/config/results/source remain; five current full states protected.')
    (RUN / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
