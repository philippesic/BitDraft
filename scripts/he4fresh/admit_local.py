"""Authenticate the local dataset once while its resumable download completes."""
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import P,R,OUT,DEST,TEACHER,REMOTE_TEACHER,MANIFEST_HASH,pin,save

WORK=R/'pause-fix-20261110'
STAGE=R/'local-teacher-transfer'
ARCHIVE=R/'local-archive-transfer'


def main():
    recipe=json.load(open(R/'recipe.json'))
    index_path=WORK/'index-copy.json'
    if not index_path.exists():
        subprocess.run(['scp','pstack-2080:/home/philip/he4broad-training-20261110-01/index.json',str(index_path)],check=True)
    index_pin=pin(index_path);assert index_pin['sha256']==recipe['config']['data']['sha256']
    index=json.load(open(index_path));assert index['manifest_sha256']==MANIFEST_HASH
    assert (index['sequences'],index['windows'])==(15647,1568435)
    journal=WORK/'local-data-verification.jsonl';done={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line);assert row['index_sha256']==index_pin['sha256'];done[row['row']]=row
    while len(done)<len(index['rows']):
        added=0
        for i,row in enumerate(index['rows']):
            if i in done:continue
            refs=[row['receipt']['metadata'],*row['receipt']['files']]
            paths=[STAGE/Path(ref['path']).relative_to(REMOTE_TEACHER) for ref in refs]
            if not all(p.is_file() and p.stat().st_size==ref['bytes'] for p,ref in zip(paths,refs)):continue
            arrays={};nbytes=0
            for path,ref in zip(paths,refs):
                raw=path.read_bytes()
                if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
                    save(WORK/'local-data-failure.json',dict(path=str(path),expected=ref,row=i))
                    raise ValueError('Local transfer checksum mismatch: '+str(path))
                nbytes+=len(raw)
                if path.name=='capture.json':assert json.loads(raw)==row['metadata']
                elif path.suffix=='.f32':
                    a=np.frombuffer(raw,dtype='<f4');assert a.size==int(np.prod(row['receipt']['feature_shape']))
                    assert np.isfinite(a).all()
                elif path.suffix=='.npy':arrays[path.name]=np.load(io.BytesIO(raw),allow_pickle=False)
            m=row['metadata']
            assert arrays['inputs.npy'].tolist()==m['input_tokens']
            assert arrays['label_positions.npy'].tolist()==m['logit_positions']
            assert arrays['labels.npy'].tolist()==m['greedy_ids']
            assert all(a.dtype==np.dtype('int32') for a in arrays.values())
            assert row['anchors']==m['logit_positions'][:-14]
            assert row['receipt']['no_post_eog_labels']
            record=dict(row=i,id=row['id'],files=len(refs),bytes=nbytes,index_sha256=index_pin['sha256'],
                status='HASH_ARRAY_ALIGNMENT_AND_FINITE_PASS',checked_unix=time.time())
            with journal.open('a') as out:
                out.write(json.dumps(record)+'\n');out.flush()
            done[i]=record;added+=1
            if added%100==0:progress(done,index)
        progress(done,index)
        if len(done)==len(index['rows']):break
        state=json.load(open(P/'runs/he4-localize-download-20261110/state.json'))
        if state['status'] in ('finished','failed','interrupted') and state.get('exit_code')!=0:
            raise RuntimeError('Dataset download stopped: '+str(state))
        if not added and state['status']=='finished':raise RuntimeError('Completed transfer has missing dataset files')
        time.sleep(10)
    while json.load(open(P/'runs/he4-localize-download-20261110/state.json'))['status'] not in ('finished','failed','interrupted'):
        time.sleep(10)
    assert json.load(open(P/'runs/he4-localize-download-20261110/state.json'))['exit_code']==0
    manifest=STAGE/'supplemental/combined-manifest.json'
    assert pin(manifest)['sha256']==MANIFEST_HASH
    save(WORK/'dataset-verified.json',dict(status='ALL_CAPTURE_FILES_VERIFIED',index=recipe['config']['data'],
        manifest_sha256=MANIFEST_HASH,sequences=len(done),windows=index['windows'],
        verified_files=sum(x['files'] for x in done.values()),verified_bytes=sum(x['bytes'] for x in done.values()),
        dataset_transfer=json.load(open(P/'runs/he4-localize-download-20261110/state.json')),
        verification_journal=str(journal),checked_unix=time.time()))
    # All current states, exports, live accounting and metadata are local too.
    ARCHIVE.mkdir(exist_ok=True)
    command=['rsync','-aH','--partial','--info=progress2','-e','ssh',
             'pstack-2080:/home/philip/he4broad-training-20261110-01/',str(ARCHIVE)+'/']
    subprocess.run(command,check=True)
    assert pin(ARCHIVE/'index.json')['sha256']==recipe['config']['data']['sha256']
    for receipt_path in ARCHIVE.glob('*.pt.json'):
        receipt=json.load(open(receipt_path));archive=ARCHIVE/Path(receipt['archive']).name
        assert archive.stat().st_size==receipt['archive_bytes']
        assert pin(archive)['sha256']==receipt['archive_sha256']
    saved=json.load(open(ARCHIVE/'candidate/result.json'));assert saved['status']=='PAUSED' and saved['updates']==20671
    # Check no still-live GPU owner before replacing only this run's mounts.
    from .health import alive
    assert not alive(json.load(open(R/'gpu-owner.json'))['owner'])
    for mount in (TEACHER,DEST):
        assert mount.is_mount()
        subprocess.run(['fusermount3','-u',str(mount)],check=True)
        mount.rmdir()
    STAGE.rename(TEACHER);ARCHIVE.rename(DEST)
    assert not TEACHER.is_mount() and not DEST.is_mount()
    assert TEACHER.stat().st_dev==DEST.stat().st_dev==P.stat().st_dev
    save(R/'local-data.json',dict(status='VERIFIED_LOCAL_READY',checked_unix=time.time(),
        index=recipe['config']['data'],manifest_sha256=MANIFEST_HASH,sequences=index['sequences'],windows=index['windows'],
        teacher=str(TEACHER),archive=str(DEST),source='pstack-2080:/home/philip/he4-capture-2080ti',
        verification=pin(WORK/'dataset-verified.json'),
        policy='One-time complete checksum/array/finite admission; runtime reads local feature bytes without repeat hashing or full scans.'))
    print('LOCAL_DATA_AND_ALL_CHECKPOINTS_ADOPTED',flush=True)


def progress(done,index):
    save(WORK/'local-data-progress.json',dict(checked_unix=time.time(),verified_sequences=len(done),
        total_sequences=len(index['rows']),verified_bytes=sum(x['bytes'] for x in done.values()),
        free_bytes=__import__('shutil').disk_usage(P).free))


if __name__=='__main__':main()
