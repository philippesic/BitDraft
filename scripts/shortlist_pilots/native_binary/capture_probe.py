#!/usr/bin/env python3
"""Generate a standalone current-native input capture helper from pinned probe source.

Only the callback selector/stage changes. Existing genuine feature injection,
seven-row geometry, KV policy, resource checks and model load remain intact.
"""
import argparse
import hashlib
import json
from pathlib import Path

def transform(source):
    pairs=[('(name!="l_out-4"&&name!="result_norm")','(name!="binary_input-4")'),
           ('seen["l_out-4"]==1&&seen["result_norm"]==1&&tensors.size()==2','seen["binary_input-4"]==1&&tensors.size()==1'),
           ('name=="l_out-4"?"decoder.layer4.post_residual.pre_output_norm":"decoder.final.post_output_norm"','"decoder.layer4.ffn_up.before_input_scale_and_A8"')]
    for old,new in pairs:
        if source.count(old)!=1:raise ValueError("capture source missing/nonunique anchor: "+old)
        source=source.replace(old,new,1)
    return source

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-sha256",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    raw=a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.source_sha256:raise ValueError("probe source digest differs")
    changed=transform(raw.decode())
    with a.out.open("x") as stream:stream.write(changed)
    print(json.dumps({"source_sha256":a.source_sha256,"output_sha256":hashlib.sha256(changed.encode()).hexdigest(),"GPU_calls":0}))

if __name__=="__main__":main()
