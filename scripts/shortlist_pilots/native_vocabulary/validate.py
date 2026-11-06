#!/usr/bin/env python3
"""Local native syntax/anchor checks; never opens a model or GPU."""
import argparse,hashlib,json,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from patch import transform
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'runs/shortlist-throughput-20261027/base');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('fresh evidence required')
    source=ROOT/'runs/w1a8-openitems-closeout-20261023/tree_verification01/integration04/source';compiler=shutil.which('clang++') or shutil.which('g++');records=[];t=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='vocab-native-') as tmp:
        tmp=Path(tmp)
        for subtree in ('src','include','ggml/include'):
            for old in (source/subtree).rglob('*.h'):
                q=tmp/old.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,q)
        for old in a.base.rglob('*'):
            if old.is_file():
                q=tmp/old.relative_to(a.base);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,q)
        for rel in ('src/llama-arch.h','src/llama-arch.cpp','src/llama-model.h','src/llama-context.cpp','common/speculative.cpp','src/models/dflash.cpp'):
            (tmp/rel).write_text(transform(rel,(tmp/rel).read_text()))
        shutil.copyfile(HERE/'vocab_head.h',tmp/'src/vocab_head.h')
        inc=['-I'+str(q) for q in [tmp/'src/models',tmp/'src',tmp/'include',tmp/'ggml/include',*[source/x for x in ('include','common','ggml/include','ggml/src','vendor')]]]
        for rel in ('src/models/dflash.cpp','src/llama-arch.cpp'):
            cmd=[compiler,'-std=c++17','-fsyntax-only',*inc,str(tmp/rel)];r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            records.append({'argv':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
            if r.returncode:raise RuntimeError(json.dumps(records[-1]))
    value={'status':'SYNTAX_PASS_NO_GPU','commands':records,'wall_s':time.monotonic()-t,'GPU_calls':0,'limits':'Source/ordinary-op syntax only; Root actual CUDA graph placement, native seven-row token/reference check and brief matched HTTP measurements remain required.'}
    a.out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':value['status'],'evidence':str(a.out)}))
if __name__=='__main__':main()
