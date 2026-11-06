"""Bounded, read-only artifact inventory for a completed investigation's cleanup.

Categories are hints, never deletion decisions or proof of checkpoint validity.
"""
import argparse
import json
import os
from pathlib import Path


def inventory(root, limit=10000):
    root = Path(root).absolute()
    if limit < 1:
        raise ValueError('max entries must be positive')
    if any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise ValueError('Use a real run directory, not a symlink')
    rows, pending, count = [], [root], 0
    complete = True
    while pending and complete:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > limit:
                    complete = False
                    break
                p = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink():
                    kind = 'protected-link'
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(p)
                    continue
                elif not entry.is_file(follow_symlinks=False):
                    kind = 'protected-special'
                elif p.suffix in ('.pt', '.pth', '.safetensors', '.gguf', '.npz', '.bin', '.bsz', '.zst'):
                    kind = 'model-or-array-review-dependencies'
                elif p.suffix in ('.json', '.jsonl', '.csv', '.md', '.html', '.png', '.svg'):
                    kind = 'evidence-review-retention'
                else:
                    kind = 'unclassified-protect'
                rows.append({'path': str(p.relative_to(root)), 'category': kind,
                             'bytes': info.st_size, 'device': info.st_dev,
                             'inode': info.st_ino, 'mtime_ns': info.st_mtime_ns})
    rows.sort(key=lambda row: (-row['bytes'], row['path']))
    return {'root': str(root), 'complete': complete, 'entries_scanned': min(count, limit),
            'logical_bytes_listed': sum(row['bytes'] for row in rows),
            'limits': 'Metadata only; hardlinks may double-count. No free-space prediction, ownership, resume or deletion qualification.',
            'files': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory', type=Path)
    p.add_argument('--max-entries', type=int, default=10000)
    args = p.parse_args()
    try:
        result = inventory(args.run_directory, args.max_entries)
    except (OSError, ValueError) as error:
        p.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
