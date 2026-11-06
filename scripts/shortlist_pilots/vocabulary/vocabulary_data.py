"""Read-only adapters for current native packets and original TRAIN continuations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/w1a1_eagle').is_dir() and (p/'runs').is_dir())


def pin(path):
    p=Path(path)
    with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':d,'bytes':p.stat().st_size}


def checked(ref):
    p=Path(ref['path'])
    if pin(p)['sha256']!=ref['sha256']:raise ValueError('input hash mismatch: '+str(p))
    return p


def load_packet(path,arm):
    packet=json.loads(Path(path).read_text())
    if packet['schema']!='head_tree_native_v1' or packet['target_kv']!='F16' or packet['draft_body_rows']!=7 or packet['target_width_mask']!=3:raise ValueError('current fixed native packet required')
    cases=[c for c in packet['cases'] if c['arm']==arm]
    fit={c['group_id'] for c in cases if c['split']=='fit'};cal={c['group_id'] for c in cases if c['split']=='calibration'}
    if not fit or not cal or fit&cal or len(cases)!=len(packet['cases']):raise ValueError('FIT/CAL split and arm binding')
    for c in cases:
        if c['split'] not in ('fit','calibration') or len(c['labels'])!=7 or len(c['native_argmax'])!=7 or c['shape']!=[7,2560]:raise ValueError('native seven-row geometry')
    checked(packet['model']);return packet,cases,fit,cal


def hidden(case):
    h=np.fromfile(checked(case['hidden']),dtype='<f4').reshape(7,2560)
    if not np.isfinite(h).all():raise ValueError('finite actual normalized hidden required')
    return torch.from_numpy(h.copy())


def original_train_records(packet,fit_groups,calibration_groups):
    # Read the source-case metadata to locate FIT records, but never open a CAL
    # group's native teacher receipt or target continuation for fitting.
    source=json.loads(checked(packet['cases_source']).read_text());records=[];refs=[];seen=set()
    for c in source['cases']:
        group=c['group_id']
        if group not in fit_groups:continue
        if group in calibration_groups:raise ValueError('CAL source requested by FIT')
        if group in seen:continue
        p=checked(c['native_receipt']);r=json.loads(p.read_text())
        if (r['schema']!='block_native_teacher_request_v1' or not r['complete'] or
                r['chain_ancestry']['source_split']!='TRAIN' or r['target_sha256']!=packet['target']['sha256'] or
                r['target_kv_type']!='F16' or r['target_precision']!='F16' or
                r['prefix_contract']!='native_tokenized_prompt_then_target_only_greedy'):
            raise ValueError('authentic original TRAIN target continuation required')
        tokens=r['tokens'];prompt=r['prompt_length']
        if not 0<prompt<len(tokens) or tokens[c['anchor']+1:c['anchor']+8]!=c['labels']:raise ValueError('original target/case join')
        records.append({'group_id':group,'source_split':'TRAIN','target_tokens':tokens[prompt:]})
        refs.append(c['native_receipt']);seen.add(group)
    if seen!=set(fit_groups):raise ValueError('original target continuation missing for FIT group')
    return records,refs


def load_weights(path):
    # Reuse the pre-existing audited GGUF parser and Q4_K/Q6_K row decoder.
    # No dependency on or mutation of head_tree/native build files.
    directory=ROOT/'runs/w1a8-campaign-20261023/dynamic_head01'
    modules={}
    for name in ('reference','gguf_io','quant_rows'):
        spec=importlib.util.spec_from_file_location(name,directory/(name+'.py'))
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);modules[name]=m
    g=modules['gguf_io'].GGUF(path);Rows=modules['quant_rows'].StoredRows
    if g.get('general.architecture')!='dflash' or g.get('dflash.sample_from_anchor') is not True:raise ValueError('private DSpark anchor-first model required')
    if any(x in g.tensors for x in ('output.scale','markov_w2.scale','output.factor_down.weight','output.factor_up.weight')):raise ValueError('original unscaled full head required')
    weights={};storage={}
    for name,key in [('head','output.weight'),('m1','markov_w1.weight'),('m2','markov_w2.weight')]:
        t=g.tensors[key];rows=Rows(g,key,t.shape,t.kind)
        array=np.empty((rows.count,rows.width),dtype=np.float32)
        for start in range(0,rows.count,1024):array[start:start+1024]=rows.read_range(start,min(start+1024,rows.count))
        weights[name]=torch.from_numpy(array);storage[name]={'type':t.kind,'shape':list(t.shape),'native_bytes':t.nbytes,'decoded_F32_bytes':array.nbytes,'payload_sha256':rows.payload_sha256}
    if weights['head'].shape!=(151936,2560) or weights['m1'].shape!=(151936,256) or weights['m2'].shape!=(151936,256):raise ValueError('actual current model geometry required')
    return weights,storage
