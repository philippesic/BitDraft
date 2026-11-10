"""Inspect immutable BSZ tensor storage without torch, CUDA or loading a model.

Run on the archive host to avoid transferring multi-GiB checkpoints over SSHFS.
Raw scratch is authenticated against the archive receipt and deleted on exit.
"""
import argparse
import collections
import hashlib
import json
import math
import pickle
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
try:
    import zstandard
    def decompress(data,size):
        return zstandard.ZstdDecompressor().decompress(data,max_output_size=size)
except ImportError:
    import ctypes
    import ctypes.util
    lib=ctypes.CDLL(ctypes.util.find_library('zstd') or 'libzstd.so.1')
    lib.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
    lib.ZSTD_decompress.restype=ctypes.c_size_t
    lib.ZSTD_isError.argtypes=[ctypes.c_size_t];lib.ZSTD_isError.restype=ctypes.c_uint
    def decompress(data,size):
        output=ctypes.create_string_buffer(size);source=ctypes.create_string_buffer(data)
        count=lib.ZSTD_decompress(output,size,source,len(data))
        assert not lib.ZSTD_isError(count) and count==size
        return output.raw[:count]


def tensor(storage,offset,size,stride,*_):
    return dict(tensor=True,storage=storage,offset=offset,shape=size,stride=stride)


class Metadata(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):
            return tensor
        if module=='torch' and name.endswith('Storage'):
            return name
        if module=='collections' and name=='OrderedDict':return collections.OrderedDict
        if module.startswith('numpy') or module=='_codecs':return super().find_class(module,name)
        raise ValueError('Unexpected metadata type '+module+'.'+name)

    def persistent_load(self,pid):
        assert pid[0]=='storage'
        return dict(dtype=pid[1],key=pid[2],location=pid[3],elements=pid[4])


def inspect(path,scratch):
    receipt=json.load(open(path.with_suffix('.json')))
    archive_digest=hashlib.sha256();raw_digest=hashlib.sha256();raw_count=0
    with tempfile.NamedTemporaryFile(prefix='checkpoint-audit-',suffix='.pt',dir=scratch) as raw:
        with path.open('rb') as src:
            def read(n):
                value=src.read(n);archive_digest.update(value);return value
            assert read(11)==b'DFLASHBSZ1\n'
            while header:=read(16):
                assert len(header)==16
                raw_size,coded_size=struct.unpack('<QQ',header);assert raw_size<=8<<20
                coded=read(coded_size);assert len(coded)==coded_size
                block=decompress(coded,raw_size)
                assert len(block)==raw_size
                n=len(block)//4*4
                block=np.frombuffer(block[:n],dtype=np.uint8).reshape(4,-1).T.copy().tobytes()+block[n:]
                raw_digest.update(block);raw_count+=len(block);raw.write(block)
        raw.flush()
        assert archive_digest.hexdigest()==receipt['archive_sha256']
        assert raw_digest.hexdigest()==receipt['raw_sha256'] and raw_count==receipt['raw_bytes']
        with zipfile.ZipFile(raw.name) as archive:
            name=next(n for n in archive.namelist() if n.endswith('/data.pkl'));prefix=name[:-len('data.pkl')]
            if prefix+'byteorder' in archive.namelist():assert archive.read(prefix+'byteorder')==b'little'
            with archive.open(name) as stream:state=Metadata(stream).load()
            def blocks(meta):
                assert meta['storage']['dtype']=='FloatStorage' and meta['offset']==0
                count=math.prod(meta['shape']);assert count==meta['storage']['elements']
                with archive.open(prefix+'data/'+meta['storage']['key']) as stream:
                    for data in iter(lambda:stream.read(4<<20),b''):
                        assert len(data)%4==0
                        yield np.frombuffer(data,dtype='<f4')
            parameters={};count=outside=nonfinite=0
            for key,value in state['parameters'].items():
                if not key.endswith('.latent_sign'):continue
                n=bad=dead=positive=0;maximum=0.;sum_abs=0.
                for a in blocks(value):
                    n+=a.size;bad+=int((~np.isfinite(a)).sum());absolute=np.abs(a)
                    dead+=int((absolute>1).sum());positive+=int((a>=0).sum())
                    maximum=max(maximum,float(absolute.max()));sum_abs+=float(absolute.sum(dtype=np.float64))
                parameters[key]=dict(elements=n,nonfinite=bad,outside_sign_ste_support=dead,
                    fraction_outside_sign_ste_support=dead/n,positive=positive,max_abs=maximum,mean_abs=sum_abs/n)
                count+=n;outside+=dead;nonfinite+=bad
            assert len(parameters)==18 and nonfinite==0
            scalar_elements=scalar_nonfinite=scale_elements=zero_scales=0;scalar_ranges={}
            for key,value in state['parameters'].items():
                if key.endswith('.latent_sign'):continue
                n=bad=zeros=0;minimum=float('inf');maximum=-float('inf')
                for a in blocks(value):
                    n+=a.size;bad+=int((~np.isfinite(a)).sum());zeros+=int((a==0).sum())
                    minimum=min(minimum,float(a.min()));maximum=max(maximum,float(a.max()))
                scalar_elements+=n;scalar_nonfinite+=bad
                if key.endswith('.scale'):scale_elements+=n;zero_scales+=zeros
                scalar_ranges[key]=dict(elements=n,nonfinite=bad,zeros=zeros,min=minimum,max=maximum)
            assert scalar_nonfinite==0
            steps=[]
            for opt in state['optimizer']['state'].values():
                steps.extend(float(x) for block in blocks(opt['step']) for x in block)
            assert set(steps)=={state['progress']['updates']} if steps else state['progress']['updates']==0
            # Fixed-head/tied head is a frozen buffer, deliberately outside this
            # master-parameter checkpoint; no model or CUDA import occurs here.
            return dict(status='AUTHENTICATED_CPU_STORAGE_INSPECTION_PASS',archive=str(path),
                archive_sha256=receipt['archive_sha256'],updates=state['progress']['updates'],
                config_hash=state['config_hash'],data=state['config']['data'],optimizer_steps=sorted(set(steps)),
                optimizer_groups=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay']) for g in state['optimizer']['param_groups']],
                cursor=state['progress']['cursor'],window_order_entries=len(state['progress']['order']),
                latent_elements=count,nonfinite_latents=nonfinite,outside_sign_ste_support=outside,
                fraction_outside_sign_ste_support=outside/count,projections=parameters,
                scalar_elements=scalar_elements,scalar_nonfinite=scalar_nonfinite,
                scale_elements=scale_elements,zero_scales=zero_scales,
                fraction_zero_scales=zero_scales/scale_elements,scalar_ranges=scalar_ranges,
                raw_sha_verified=True,compressed_sha_verified=True,no_torch_import=True,no_cuda_import=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('archives',type=Path,nargs='+');args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in args.archives:
        records.append(inspect(path,args.out.parent))
        args.out.write_text(json.dumps(dict(records=records),indent=2)+'\n')
        print(json.dumps({k:records[-1][k] for k in ('updates','latent_elements','fraction_outside_sign_ste_support','optimizer_steps')}),flush=True)
