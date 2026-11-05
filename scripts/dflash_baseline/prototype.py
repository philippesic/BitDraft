"""Versioned CPU prototype contracts. Native admission is deliberately fail-closed."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

PRESETS = {'vanilla': (8,8,False), 'essential-a8': (8,8,True),
           'gate-up-a4': (4,8,True), 'all-a4': (4,4,True),
           'a2-stress': (2,2,True), 'protected-example': (4,8,True)}

def digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def resolve(request):
    allowed={'version','preset','source','parent','data','budget_seconds','evaluation','overrides','features','quantizer'}
    if set(request)-allowed: raise ValueError('unknown config fields')
    if request.get('version') != 1: raise ValueError('unsupported config version')
    for name in ('source','parent','data'):
        pin=request.get(name)
        if not isinstance(pin,dict) or set(pin)!={'path','sha256'} or len(pin['sha256'])!=64:
            raise ValueError(f'{name} requires path and SHA256 pin')
    budget=request.get('budget_seconds')
    if type(budget) not in (int,float) or not math.isfinite(budget) or budget<=0: raise ValueError('explicit positive update-time budget required')
    gate,down,affine=PRESETS[request['preset']]
    features={'midpoint':affine,'gain':affine,'scale':True,'sign':True,'norm':True}
    for k,v in request.get('features',{}).items():
        if k not in features or type(v) is not bool: raise ValueError('invalid feature switch')
        features[k]=v
    quantizer=request.get('quantizer',{'mode':'dynamic','clip':1.0})
    if set(quantizer)!={'mode','clip'} or quantizer['mode'] not in ('dynamic','fixed','learned') or not math.isfinite(quantizer['clip']) or quantizer['clip']<=0: raise ValueError('invalid quantizer')
    coverage={f'blk.{i}.ffn_{p}':{'weight':'W1','bits':down if p=='down' else gate} for i in range(6) for p in ('gate','up','down')}
    if request['preset']=='protected-example': coverage['blk.0.ffn_down']={'weight':'BF16','bits':None}
    for name,value in request.get('overrides',{}).items():
        if name not in coverage or set(value)!={'weight','bits'}: raise ValueError('invalid projection override')
        if not ((value['weight']=='W1' and type(value['bits']) is int and value['bits'] in (2,4,8)) or (value['weight']=='BF16' and value['bits'] is None)): raise ValueError('unsupported weight/activation combination')
        coverage[name]=copy.deepcopy(value)
    result=copy.deepcopy(request)
    result.update(coverage=coverage,features=features,quantizer=quantizer,arithmetic='fixed_w1ax_finite_reciprocal_v2',schedule='constant',operand_storage='int8-reference')
    return result

class Projection(nn.Module):
    def __init__(self,weight,bits=8,midpoint=True,gain=None,quantizer=None):
        super().__init__()
        if bits not in (2,4,8): raise ValueError('unsupported activation width')
        self.bits=bits
        self.quantizer=quantizer or {'mode':'dynamic','clip':1.0}
        self.latent_sign=nn.Parameter(weight.detach().float().clone())
        self.scale=nn.Parameter(weight.abs().mean(-1))
        self.midpoint=nn.Parameter(torch.zeros(weight.shape[0]),requires_grad=midpoint)
        self.log_gain=gain if gain is not None else nn.Parameter(torch.zeros(weight.shape[1]),requires_grad=False)
        self.log_clip=nn.Parameter(torch.tensor(math.log(self.quantizer['clip'])),requires_grad=self.quantizer['mode']=='learned')
    def stream(self,x):
        x=x.float()*self.log_gain.exp()
        qmax=2**(self.bits-1)-1
        if self.quantizer['mode']=='dynamic':
            magnitude=x.detach().contiguous().view(torch.int32)&2147483647
            bound=magnitude.amax(-1,keepdim=True).view(torch.float32)
            inv=torch.where(bound>0,qmax/bound,0)
            normalized=x*inv
            overflow=(bound>0)&~torch.isfinite(inv)
            if bool(overflow.any()):
                fallback=(x.double()/bound.double().clamp_min(torch.finfo(torch.float64).tiny)*qmax).float()
                normalized=torch.where(overflow,fallback,normalized)
            hard=normalized.round().clamp(-qmax,qmax)
            return hard.detach(),bound/qmax
        bound=self.log_clip.exp() if self.quantizer['mode']=='learned' else x.new_tensor(self.quantizer['clip'])
        clipped=torch.maximum(torch.minimum(x,bound),-bound)
        z=clipped*(qmax/bound)
        hard=z.round().clamp(-qmax,qmax)
        return z+(hard-z).detach(),bound/qmax
    @torch.no_grad()
    def project_scales_(self):
        self.scale.clamp_(min=0)
    def forward(self,x):
        codes,step=self.stream(x)
        from w1a1_eagle.recurrent_binary import hard_sign_ste
        signs=hard_sign_ste(self.latent_sign)
        scale=self.scale.clamp_min(0)
        native=(F.linear(codes,signs)*scale)*step + (codes.sum(-1,keepdim=True)*step)*self.midpoint
        values=codes*step
        if self.quantizer['mode']=='dynamic':
            raw=x.float()*self.log_gain.exp()
            values=values.detach()+(raw-raw.detach())
        surrogate=F.linear(values,signs)*scale+values.sum(-1,keepdim=True)*self.midpoint
        return native.detach()+(surrogate-surrogate.detach())



def native_admit(config):
    raise ValueError('prototype v1 has no native loader: affine/per-projection metadata unsupported; legacy baseline export only')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['resolve','native-screen'])
    parser.add_argument('config',type=Path)
    args=parser.parse_args()
    config=resolve(json.loads(args.config.read_text()))
    if args.mode=='native-screen': native_admit(config)
    print(json.dumps({'config':config,'sha256':digest(config),'qualification':'CPU arithmetic only'},indent=2))
if __name__=='__main__': main()
