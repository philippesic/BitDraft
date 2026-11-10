"""Bounded CPU replay of capture materialization, split into timed stages.

Does not import a model, initialize CUDA, change the live cache, or update data.
Read timing includes SSHFS/OS caching and cannot isolate physical network bytes.
"""
import argparse
import hashlib
import io
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .common import R, OUT, TEACHER, REMOTE_TEACHER, REPORT, pin, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    assert 1 <= args.samples <= 64
    torch.set_num_threads(2)
    recipe = json.load(open(R/'recipe.json'))
    assert pin(recipe['config']['data']['path']) == recipe['config']['data']
    index = json.load(open(recipe['config']['data']['path']))
    windows = [(i,a) for i,row in enumerate(index['rows']) for a in row['anchors']]
    order = list(range(len(windows))); random.Random(8101).shuffle(order)
    # A fixed window-weighted sample independent of the live worker cursor.
    picks = random.Random(810102).sample(range(len(order)), args.samples)
    records=[]
    taps=(2,6,10,14,18,22,26,30)
    for pick in picks:
        sequence, anchor = windows[order[pick]]
        row=index['rows'][sequence];receipt=row['receipt'];metadata=row['metadata']
        started=time.perf_counter();times={};arrays={};nbytes=0;files=[]
        t=time.perf_counter()
        labels=dict(zip(metadata['logit_positions'],metadata['greedy_ids'],strict=True))
        assert row['anchors']==metadata['logit_positions'][:-14]
        assert all(a+j-1 in labels for a in row['anchors'] for j in range(1,16))
        times['metadata_checks']=time.perf_counter()-t
        times['remote_file_reads']=times['sha256']=0.
        for ref in receipt['files']:
            path=TEACHER/Path(ref['path']).relative_to(REMOTE_TEACHER)
            t=time.perf_counter();raw=path.read_bytes();read_s=time.perf_counter()-t
            t=time.perf_counter()
            assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
            hash_s=time.perf_counter()-t
            times['remote_file_reads']+=read_s;times['sha256']+=hash_s;nbytes+=len(raw)
            files.append(dict(name=path.name,bytes=len(raw),read_seconds=read_s,hash_seconds=hash_s))
            if path.name.startswith('layer-') or path.suffix=='.npy':arrays[path.name]=raw
        t=time.perf_counter()
        positions=np.load(io.BytesIO(arrays['label_positions.npy']),allow_pickle=False)
        target=np.load(io.BytesIO(arrays['labels.npy']),allow_pickle=False)
        inputs=np.load(io.BytesIO(arrays['inputs.npy']),allow_pickle=False)
        assert positions.dtype==target.dtype==inputs.dtype==np.dtype('int32')
        assert positions.tolist()==metadata['logit_positions']
        assert target.tolist()==metadata['greedy_ids'] and inputs.tolist()==metadata['input_tokens']
        shape=tuple(receipt['feature_shape'])
        features=torch.from_numpy(np.concatenate([
            np.frombuffer(arrays[f'layer-{tap}.f32'],dtype='<f4').reshape(shape)
            for tap in taps],axis=1))
        times['array_parse_and_copy']=time.perf_counter()-t
        t=time.perf_counter();assert torch.isfinite(features).all()
        times['finite_feature_check']=time.perf_counter()-t
        elapsed=time.perf_counter()-started
        times['other_python']=max(0.,elapsed-sum(times.values()))
        records.append(dict(sequence=sequence,anchor=anchor,window_order_position=pick,
            bytes=nbytes,files=files,seconds=times,total_seconds=elapsed))
        del features,arrays,raw
    live=json.load(open(OUT/'live.json'))
    keys=list(records[0]['seconds'])
    totals={k:sum(x['seconds'][k] for x in records) for k in keys}
    total=sum(totals.values())
    result=dict(status='PASS',checked_unix=time.time(),source=pin(__file__),
        recipe=pin(R/'recipe.json'),data=recipe['config']['data'],samples=len(records),
        torch_threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized(),
        sampling='Fixed window-weighted sample seed 810102 from global seed-8101 order; fresh materialization for each.',
        records=records,total_seconds=total,total_bytes=sum(x['bytes'] for x in records),
        categories={k:dict(seconds=totals[k],fraction=totals[k]/total,
            mean_seconds=totals[k]/len(records),median_seconds=statistics.median(x['seconds'][k] for x in records)) for k in keys},
        actual_training=dict(updates=live['updates'],training_wall_seconds=live['training_seconds'],
            update_seconds=live['additional_update_seconds'],loader_seconds=live['data_io_seconds']),
        limits=['CPU replay concurrent with the healthy trainer; brief extra CPU/storage contention is possible.',
            'Remote-read wall time includes SSHFS, SSH, remote/local disk and page caches, and file-call latency; not a physical network-only measurement.',
            'Stage fractions describe these replayed samples, not direct instrumentation of historical training.',
            'No cache eviction, cold-cache guarantee, CUDA work, active source edit, or trainer restart.'])
    assert not result['cuda_initialized']
    save(REPORT/'loader-profile.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
