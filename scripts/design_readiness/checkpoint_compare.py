"""Exact comparison of declared checkpoint fields, including dtype and signed zero.

Only load trusted project checkpoints. Cost/timing metadata is compared only if
explicitly requested; this tool does not decide which scientific fields matter.
"""
import argparse
import json
from pathlib import Path
import torch


def exact(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype!=right.dtype or left.shape!=right.shape:
            return False
        a=left.detach().contiguous().reshape(-1).view(torch.uint8)
        b=right.detach().contiguous().reshape(-1).view(torch.uint8)
        return all(torch.equal(a[i:i+1048576].cpu(),b[i:i+1048576].cpu()) for i in range(0,a.numel(),1048576))
    if type(left) is not type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(exact(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
    return left==right


def compare(left, right, fields):
    return {k:k in left and k in right and exact(left[k],right[k]) for k in fields}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--fields',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    left=torch.load(a.left,map_location='cpu',weights_only=False);right=torch.load(a.right,map_location='cpu',weights_only=False)
    checks=compare(left,right,a.fields);result=dict(status='DECLARED_FIELDS_BYTE_EXACT' if all(checks.values()) else 'DECLARED_FIELDS_DIFFER',left=str(a.left.resolve()),right=str(a.right.resolve()),checks=checks)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));raise SystemExit(0 if all(checks.values()) else 1)

if __name__=='__main__':main()
