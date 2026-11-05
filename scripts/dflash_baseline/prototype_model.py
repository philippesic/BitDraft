"""Explicit conversion of a loaded baseline model; baseline source stays unchanged."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from .prototype import Projection, digest


def authenticate(config):
    for name in ('source', 'parent', 'data'):
        pin=config[name]
        h=hashlib.sha256()
        with Path(pin['path']).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        if h.hexdigest()!=pin['sha256']:
            raise ValueError(f'{name} SHA256 mismatch')


def install(model, config, source_weights):
    """Install after loading parent parameters. Protected weights require source tensors.

    source_weights maps canonical projection names to original BF16 tensors.
    Never reconstruct a protected projection from binary latents.
    """
    actual=set(model.binary_linears())
    if actual!=set(config['coverage']): raise ValueError('projection coverage mismatch')
    features=config['features']
    for p in model.parameters(): p.requires_grad_(False)
    for i,layer in enumerate(model.layers):
        shared=None
        for name in ('ffn_gate','ffn_up','ffn_down'):
            key=f'blk.{i}.{name}'; spec=config['coverage'][key]; old=getattr(layer,name)
            if spec['weight']=='BF16':
                weight=source_weights[key]
                if weight.dtype!=torch.bfloat16: raise ValueError('protected source must be BF16')
                from .model import NativeBF16Linear
                linear=NativeBF16Linear(weight.float())
                setattr(layer,name,linear)
                continue
            if name!='ffn_down' and shared is None:
                shared=nn.Parameter(torch.zeros(old.latent_sign.shape[1],device=old.latent_sign.device),requires_grad=features['gain'])
            gain=shared if name!='ffn_down' else nn.Parameter(torch.zeros(old.latent_sign.shape[1],device=old.latent_sign.device),requires_grad=features['gain'])
            new=Projection(old.latent_sign.detach(),spec['bits'],features['midpoint'],gain,config['quantizer']).to(old.latent_sign.device)
            with torch.no_grad():
                new.scale.copy_(old.scale if isinstance(old,Projection) else old.effective_scales())
                if isinstance(old,Projection):
                    if features['midpoint']: new.midpoint.copy_(old.midpoint)
                    if features['gain']: new.log_gain.copy_(old.log_gain)
                    if config['quantizer']==old.quantizer: new.log_clip.copy_(old.log_clip)
            new.latent_sign.requires_grad_(features['sign']);new.scale.requires_grad_(features['scale'])
            setattr(layer,name,new)
        for name in ('attn_norm','attn_q_norm','attn_k_norm','ffn_norm'):
            getattr(layer,name).requires_grad_(features['norm'])
    for name in ('output_norm','output_norm_enc'): getattr(model,name).requires_grad_(features['norm'])
    model.prototype_config=config
    return model


def optimizer(model, rates=None):
    rates=rates or dict(sign=1e-3,scale=1e-5,midpoint=1e-5,gain=1e-5,clip=1e-5,norm=1e-5)
    groups={k:[] for k in rates}
    suffixes={'latent_sign':'sign','scale':'scale','midpoint':'midpoint','log_gain':'gain','log_clip':'clip'}
    for name,p in model.named_parameters():
        if p.requires_grad: groups[suffixes.get(name.rsplit('.',1)[-1],'norm')].append(p)
    return torch.optim.AdamW([dict(params=v,lr=rates[k],name=k) for k,v in groups.items() if v],weight_decay=0,foreach=False)


def save(path, model, opt, progress):
    import random
    import numpy as np
    required={'updates','update_seconds','cursor','order'}
    if set(progress)!=required: raise ValueError('incomplete progress')
    state=dict(version=1,config=model.prototype_config,config_hash=digest(model.prototype_config),model=model.state_dict(),optimizer=opt.state_dict(),progress=progress,
               rng=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None))
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def resume(path,model,opt):
    import random
    import numpy as np
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['version']!=1 or state['config_hash']!=digest(state['config']) or state['config_hash']!=digest(model.prototype_config):
        raise ValueError('exact resume config mismatch; explicit adaptation required')
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng']['torch']);random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    if state['rng']['cuda'] is not None: torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['progress']


def adapt(model, config, source_weights):
    """Explicit model adaptation resets optimizer and exposure accounting.

    Caller must load the authenticated parent before adapting. Strict resume
    remains a different operation and never passes through this function.
    """
    authenticate(config)
    previous=getattr(model,'prototype_config',None)
    install(model,config,source_weights)
    model.prototype_lineage=dict(parent=config['parent'],
        parent_config_hash=digest(previous) if previous else None,
        parent_kind='prototype' if previous else 'baseline',optimizer_policy='reset')
    return model,optimizer(model)
