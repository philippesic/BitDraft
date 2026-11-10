"""Bounded authenticated materialization; no eager all-capture feature list."""
import io
import json
import hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from scripts.dflash_baseline.model import Capture, TAPS
from .common import *


class TeacherCapture(Capture):
    def __init__(self, row):
        receipt, metadata = row['receipt'], row['metadata']
        self.path = Path(receipt['metadata']['path']).parent
        self.meta = metadata
        self.tokens = metadata['input_tokens']
        self.labels = dict(zip(metadata['logit_positions'], metadata['greedy_ids'], strict=True))
        self.anchors = row['anchors']
        assert self.anchors == metadata['logit_positions'][:-14]
        assert all(a + j - 1 in self.labels for a in self.anchors for j in range(1,16))
        arrays = {}
        for ref in receipt['files']:
            path = TEACHER / Path(ref['path']).relative_to(REMOTE_TEACHER)
            raw = path.read_bytes()
            assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], path
            if path.name.startswith('layer-') or path.suffix == '.npy': arrays[path.name] = raw
        positions = np.load(io.BytesIO(arrays['label_positions.npy']),allow_pickle=False)
        labels = np.load(io.BytesIO(arrays['labels.npy']),allow_pickle=False)
        inputs = np.load(io.BytesIO(arrays['inputs.npy']),allow_pickle=False)
        assert positions.dtype == labels.dtype == inputs.dtype == np.dtype('int32')
        assert positions.tolist() == metadata['logit_positions']
        assert labels.tolist() == metadata['greedy_ids'] and inputs.tolist() == self.tokens
        shape = tuple(receipt['feature_shape'])
        self.features = torch.from_numpy(np.concatenate([
            np.frombuffer(arrays[f'layer-{tap}.f32'],dtype='<f4').reshape(shape)
            for tap in TAPS],axis=1))
        assert torch.isfinite(self.features).all()


class LazyTeachers:
    def __init__(self, rows):
        self.rows = rows
        self.cache = OrderedDict()

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        if index not in self.cache:
            # Bound CPU feature memory to two sequences, regardless of pool size.
            while len(self.cache) >= 2: self.cache.popitem(last=False)
            self.cache[index] = TeacherCapture(self.rows[index])
        self.cache.move_to_end(index)
        return self.cache[index]


def data(config):
    for ref in (config['source'],config['evaluation']['target'],config['data']): authenticate(ref)
    index = json.load(open(config['data']['path']))
    assert index['manifest_sha256'] == MANIFEST_HASH
    assert index['target_sha256'] == config['evaluation']['target']['sha256']
    assert (index['sequences'],index['windows']) == (15647,1568435)
    old = json.load(open(COMPONENT/'broader-train-capture/manifest.json'))
    devrows = [row for row in old['records'] if row['split']=='consumed_DEV']
    for row in devrows:
        for ref in (row['metadata'],*row['files']): authenticate(ref)
    dev = [Capture(Path(row['metadata']['path']).parent) for row in devrows]
    dev_prompts = {tuple(cap.tokens[:cap.meta['prompt_tokens']]) for cap in dev}
    assert not any(tuple(row['metadata']['input_tokens'][:row['metadata']['prompt_tokens']]) in dev_prompts
                   for row in index['rows']), 'TRAIN overlaps consumed DEV prompts'
    windows = [(i,a) for i,row in enumerate(index['rows']) for a in row['anchors']]
    assert len(windows) == index['windows']
    save(OUT/'data-proof.json',dict(status='PASS',sequences=len(index['rows']),
        windows=len(windows),unique_labels=index['unique_labels_in_windows'],
        authentication='Metadata/exclusions pinned; every materialized file hashed before exposure',
        dev_sequences=len(dev),consumed_dev_prompt_overlap=0,training_data=config['data'],heldout_opened=False))
    return LazyTeachers(index['rows']), dev, windows
