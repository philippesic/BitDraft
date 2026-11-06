#!/usr/bin/env python3
"""Read-only bounded native-evaluation metadata snapshot; never loads weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile

ROOT = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-20261015-04/lane/attempts/direct-dspark-f199bcd4a05c484a90ee94c495ebe3de/evaluations/c737bcf53f2f467d9032e98a418fd622')
RECEIPT_SHA = 'ea453c0084ece1ed4745ddc6de41292fe685acce24e96596409a68b45440b0e0'
REPORT_SHA = '90fae8ee27f12f01e89ffa7d7aa1c63c67486957c13ce6ce1745d3aae9357d24'
MAX_FILE = 16 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_FILES = 4096
METADATA = {'.json', '.jsonl', '.log', '.txt'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(value):
    return [getattr(value, key) for key in ('st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns')]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'canonical source path required: '+str(path))
    for ancestor in path.parents:
        require(not ancestor.is_symlink(), 'source ancestor symlink: '+str(ancestor))
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= MAX_FILE,
            'bounded regular source with one link required: '+str(path))
    descriptor=os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(descriptor,'rb') as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before),'source changed at open')
        raw=stream.read(MAX_FILE+1)
        require(identity(os.fstat(stream.fileno())) == identity(before),'source changed during read')
    require(identity(path.lstat()) == identity(before) and len(raw) == before.st_size,'source replaced during read')
    sha=digest(raw)
    require(expected is None or sha == expected,'source SHA differs: '+str(path))
    return raw, identity(before), sha


def locators(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            yield {'path':value['path'],'sha256':value['sha256']}
        for child in value.values():
            yield from locators(child)
    elif isinstance(value,list):
        for child in value:
            yield from locators(child)


def collect(root, output, receipt_sha, report_sha, extras=()):
    root=Path(root);output=Path(output)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(),'canonical existing evaluation root required')
    require(output.is_absolute() and output.resolve() == output and not output.exists(), 'fresh canonical receiver required')
    require(not output.is_relative_to(root) and not root.is_relative_to(output),'receiver/evaluation roots must be disjoint')
    receipt_path=root/'receipt.json';report_path=root/'evaluation/report.json'
    receipt_raw,_,_=read(receipt_path,receipt_sha);report_raw,_,_=read(report_path,report_sha)
    receipt=json.loads(receipt_raw);report=json.loads(report_raw)
    require(receipt.get('evaluation') == {'path':str(report_path),'sha256':report_sha},'receipt/report exact join differs')
    expected={str(receipt_path):receipt_sha,str(report_path):report_sha}
    selected=set()
    # Exact completed evaluation root only; arrays/models never enter selection.
    for directory, dirs, files in os.walk(root,followlinks=False):
        for name in dirs:
            require(not (Path(directory)/name).is_symlink(),'evaluation metadata tree contains symlink directory')
        for name in files:
            path=Path(directory)/name
            if path.suffix.lower() in METADATA:
                selected.add(str(path))
    # Small external scientific/input metadata needed to join this checkpoint.
    seeds=[receipt]
    for key in ('protocol','frozen_lane','producer_sources','research_continuation_policy','research_bindings'):
        if key in report:seeds.append(report[key])
    export_record=report.get('export_receipt')
    if export_record:
        export=json.loads(read(export_record['path'],export_record['sha256'])[0])
        for key in ('checkpoint_sidecar','config','export_source','frozen_lane','manifest','lane_state','supervisor_state'):
            if key in export:seeds.append(export[key])
    seeds.extend(extras)
    references={}
    for seed in seeds:
        for record in locators(seed):
            path=Path(record['path']);key=str(path)
            require(path.is_absolute(),'absolute source locator required')
            if key in references:require(references[key] == record['sha256'],'conflicting expected source digests')
            references[key]=record['sha256']
            if path.suffix.lower() in {'.json','.jsonl','.py'}:
                selected.add(key)
                if key in expected:require(expected[key] == record['sha256'],'conflicting selected source digests')
                expected[key]=record['sha256']
    require(len(selected) <= MAX_FILES,'metadata member count exceeds bound')
    # Bounds checked before snapshot creation. No stat/read on excluded payloads.
    total=0
    for name in selected:
        size=Path(name).lstat().st_size
        require(size <= MAX_FILE,'metadata source exceeds per-file bound: '+name)
        total+=size
    require(total <= MAX_TOTAL,'metadata packet exceeds aggregate bound')
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'snapshot';snapshot.mkdir()
    members=[]
    for name in sorted(selected):
        path=Path(name);raw,before,sha=read(path,expected.get(name))
        relative=path.relative_to('/');destination=snapshot/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        copied,_,copy_sha=read(destination)
        # Recheck ORIGINAL after snapshot write, preserving actual observed bytes.
        require(identity(path.lstat()) == before,'source changed after snapshot write')
        require(copy_sha == sha and copied == raw,'snapshot copy differs')
        members.append({'relative_path':str(relative),'source_path':name,'source_bytes':len(raw),
                        'snapshot_bytes':len(copied),'source_sha256':sha,'snapshot_sha256':copy_sha,
                        'source_mtime_ns':before[-2],'source_identity_before':before,
                        'source_identity_after':identity(path.lstat())})
    archive=output/'native-14400-evaluation-packet.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for member in members:
            path=snapshot/member['relative_path'];raw,_,sha=read(path,member['snapshot_sha256'])
            info=tarfile.TarInfo(member['relative_path']);info.size=len(raw);info.mode=0o600
            tar.addfile(info,io.BytesIO(raw))
    with archive.open('rb') as stream:
        h=hashlib.sha256()
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    manifest={'schema':'dspark_a8_evaluation_snapshot_v1','observed_at_utc':datetime.now(timezone.utc).isoformat(),
              'snapshot_root':str(snapshot),'member_count':len(members),'member_bytes':sum(m['source_bytes'] for m in members),
              'members':members,'bundle':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest()},
              'exact_evaluation_root':str(root),'receipt_sha256':receipt_sha,'report_sha256':report_sha,
              'excluded_payload_references':[{'path':p,'sha256':s,'reason':'model/tensor/array/binary payload not read'} for p,s in sorted(references.items()) if p not in selected],
              'closure_scope':'All bounded metadata/raw/trace files under exact evaluation root + explicit scientific/input metadata roles. Nested historical policy associations may reuse prior accepted evidence or require separately pinned extra-metadata; no full model authentication claimed.'}
    target=output/'source-manifest-v2.json'
    with target.open('x') as stream:json.dump(manifest,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return {'manifest':{'path':str(target),'sha256':digest(target.read_bytes()),'bytes':target.stat().st_size},
            'bundle':manifest['bundle'],'members':len(members),'member_bytes':manifest['member_bytes']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--evaluation-root',default=str(ROOT))
    parser.add_argument('--receipt-sha256',default=RECEIPT_SHA)
    parser.add_argument('--report-sha256',default=REPORT_SHA)
    parser.add_argument('--extra-metadata',action='append',default=[],help='explicit absolute PATH=SHA256, JSON/JSONL/PY only')
    args=parser.parse_args()
    extras=[]
    for value in args.extra_metadata:
        path,sep,sha=value.rpartition('=');require(sep and Path(path).suffix.lower() in {'.json','.jsonl','.py'},'bounded explicit metadata locator required')
        extras.append({'path':path,'sha256':sha})
    print(json.dumps(collect(args.evaluation_root,args.output,args.receipt_sha256,args.report_sha256,extras),sort_keys=True))

if __name__=='__main__':main()
