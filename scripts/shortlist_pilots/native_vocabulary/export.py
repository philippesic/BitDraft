#!/usr/bin/env python3
"""Export native dynamic-r64/K1024 and TRAIN-only static-K4096 DSpark GGUFs.

Zero TRAIN updates. Dynamic original quantized output is retained ONLY for
indexed reads; static output is physically removed. Both gather/pack to F16.
Fresh destinations only. All other tensor payloads and metadata are verified.
"""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_n,'1')
import argparse,hashlib,importlib.util,json,struct,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'vocabulary'))
import vocabulary_data as data
import vocabulary_core as core
spec=importlib.util.spec_from_file_location('vocab_export_gguf',ROOT/'runs/w1a8-campaign-20261023/dynamic_head01/gguf_io.py')
g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)

def load_rows(source):
    directory=ROOT/'runs/w1a8-campaign-20261023/dynamic_head01'
    for name in ('reference','quant_rows'):
        sp=importlib.util.spec_from_file_location(name,directory/(name+'.py'));m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m)
    original=g.GGUF(source);t=original.tensors['output.weight']
    if t.shape!=(2560,151936) or t.kind not in (12,14):raise ValueError('actual original Q4_K/Q6_K full head required')
    return original,sys.modules['quant_rows'].StoredRows(original,t.name,t.shape,t.kind)

def f16bytes(value):
    array=np.asarray(value,dtype=np.float32)
    if not np.isfinite(array).all() or np.max(np.abs(array))>65504:raise ValueError('finite F16 range')
    return array.astype('<f2').tobytes()

def stream_export(source,expected,output,additions,metadata,remove=()):
    """Small independently testable raw-byte GGUF writer."""
    source=Path(source);output=Path(output)
    if output.exists() or source.resolve()==output.resolve():raise ValueError('fresh distinct output required')
    if g.sha(source)!=expected:raise ValueError('source pin')
    original=g.GGUF(source);fields=dict(original.fields)
    for key,kind,value in metadata:
        if key in fields:raise ValueError('metadata collision: '+key)
        fields[key]=(kind,value,g.kv(key,kind,value))
    preserved=[t for t in original.tensors.values() if t.name not in set(remove)];entries=[];offset=0
    names=set(original.tensors)
    for name,shape,kind,raw in additions:
        if name in names:raise ValueError('tensor collision: '+name)
        names.add(name)
    for t in preserved:
        entries.append((t.name,t.shape,t.kind,offset,t.nbytes,t));offset=g.align(offset+t.nbytes,original.alignment)
    for name,shape,kind,raw in additions:
        probe=g.Tensor(name,tuple(shape),kind,0,b'')
        if len(raw)!=probe.nbytes:raise ValueError('tensor payload length')
        entries.append((name,shape,kind,offset,len(raw),raw));offset=g.align(offset+len(raw),original.alignment)
    header=b'GGUF'+struct.pack('<IQQ',3,len(entries),len(fields))+b''.join(f[2] for f in fields.values())
    header+=b''.join(g.descriptor(n,s,k,o) for n,s,k,o,_,_ in entries);start=g.align(len(header),original.alignment)
    try:
        with output.open('xb') as out,source.open('rb') as inp:
            out.write(header);out.write(b'\0'*(start-out.tell()))
            for _,_,_,off,size,src in entries:
                out.write(b'\0'*(start+off-out.tell()))
                if isinstance(src,bytes):out.write(src)
                else:
                    inp.seek(original.data_start+src.offset)
                    while size:
                        chunk=inp.read(min(size,4*1024*1024))
                        if not chunk:raise ValueError('source truncated')
                        out.write(chunk);size-=len(chunk)
        new=g.GGUF(output)
        same={t.name:original.payload_sha(t)==new.payload_sha(new.tensors[t.name]) for t in preserved}
        if not all(same.values()) or g.sha(source)!=expected:raise ValueError('unchanged tensor/source check')
        return {'output':data.pin(output),'unchanged_tensors':len(same),'removed_tensors':list(remove),
                'added_tensors':{n:{'shape':list(s),'type':k,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,s,k,b in additions},
                'all_original_metadata_preserved':True}
    except BaseException:
        if output.exists():output.unlink()
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('dynamic','static','both'),default='both');p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=20261027);p.add_argument('--router-checkpoint',type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('fresh output directory required')
    packet,_,fit,cal=data.load_packet(a.packet,a.arm);source=data.checked(packet['model']);original,rows=load_rows(source)
    if any(n in original.tensors for n in ('output.scale','markov_w2.scale','d2t','output.factor_down.weight','output.factor_up.weight')) or any(n.startswith('vocab_') for n in original.tensors):raise ValueError('exclusive original full head/Markov model required')
    a.out.mkdir(parents=True);start=time.monotonic();result={'schema':'native_vocabulary_export_v1','packet':data.pin(a.packet),'model':packet['model'],'target':packet['target'],'target_kv':'F16','body_rows':7,'submitted_depth':5,'native_markov_rows':7,'training_updates':0,'arm':a.arm,'exports':{},'precision':'Original quantized head rows decoded then F16 selected projection; full original native Markov retained. This is an explicit F16 selected-head arithmetic exception, not original quantized-head parity.'}
    meta=[('dflash.vocab_reference',8,'indexed_F16_reference_v1'),('dflash.vocab_source_sha256',8,packet['model']['sha256'])]
    if a.mode in ('static','both'):
        records,refs=data.original_train_records(packet,fit,cal);counts,frequency=core.fit_frequencies(records,fit,cal,151936);ids=core.static_ids(counts,4096)
        packed=np.concatenate([rows(ids[i:i+1024]) for i in range(0,4096,1024)],axis=0)
        additions=[('vocab_static_head.weight',(2560,4096),1,f16bytes(packed)),('vocab_static_ids',(4096,),26,ids.astype('<i4').tobytes())]
        result['exports']['static']=stream_export(source,packet['model']['sha256'],a.out/'static-k4096.gguf',additions,meta+[('dflash.vocab_mode',8,'static-k4096')],remove=('output.weight',))
        result['frequency']={**frequency,'fit_groups':sorted(fit),'excluded_calibration_groups':sorted(cal),'TRAIN_receipts':refs,'zero_frequency_fill_positions':int(np.sum(counts[ids]==0))}
    if a.mode in ('dynamic','both'):
        prep=time.monotonic();device=torch.device(a.device)
        if device.type=='cuda':
            if not torch.cuda.is_available() or torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 5080':raise ValueError('qualified CUDA device required; no CPU fallback')
        elif device.type!='cpu':raise ValueError('explicit CPU or CUDA only')
        torch.set_num_threads(1);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        if a.router_checkpoint:
            state=torch.load(a.router_checkpoint,map_location='cpu',weights_only=True)
            if set(state)!={'down.weight','up.weight'}:raise ValueError('router checkpoint keys')
            down=state['down.weight'].float().numpy();up=state['up.weight'].float().numpy()
            provenance={'checkpoint':data.pin(a.router_checkpoint)}
        else:
            array=np.empty((151936,2560),np.float32)
            for i in range(0,len(array),1024):array[i:i+1024]=rows.read_range(i,min(i+1024,len(array)))
            with torch.no_grad():
                head=torch.from_numpy(array).to(device);_,_,right=torch.svd_lowrank(head,q=144,niter=1)
                basis=right[:,:64].contiguous();coefficients=head@basis
                down=basis.T.cpu().numpy().copy();up=coefficients.cpu().numpy().copy()
            provenance={'seed':a.seed,'q':144,'niter':1,'data':'global original head weights only; no hidden states or TRAIN/CAL labels','device':str(device),'note':'Same recipe as prior portable nested r64/128; fresh SVD realization, not claimed bit-identical prior uncaptured factors.'}
        if down.shape!=(64,2560) or up.shape!=(151936,64):raise ValueError('fixed native r64 geometry')
        checkpoint=a.out/'router-r64.pt';torch.save({'down.weight':torch.from_numpy(down).half(),'up.weight':torch.from_numpy(up).half()},checkpoint)
        additions=[('vocab_router_down.weight',(2560,64),1,f16bytes(down)),('vocab_router_up.weight',(64,151936),1,f16bytes(up))]
        result['exports']['dynamic']=stream_export(source,packet['model']['sha256'],a.out/'dynamic-r64-k1024.gguf',additions,meta+[('dflash.vocab_mode',8,'dynamic-r64-k1024')])
        result['router']={**provenance,'saved_checkpoint':data.pin(checkpoint),'preparation_wall_s':time.monotonic()-prep}
    result['wall_s']=time.monotonic()-start;result['required_next']='Native seven-row conditional trace/reference gate, unchanged control body parity, then matched brief full-request W1/Q4 controls. Export alone is no native quality/speed result.'
    (a.out/'export.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
