"""Unfused vocabulary references. Torch operations also run on CPU in unit tests.

Only the CUDA-only runner measures GPU costs. Selected-row arithmetic is F32 on
original decoded weights, not the native quantized activation/reduction path.
"""
from collections import Counter
import numpy as np
import torch


def fit_frequencies(records, fit_groups, calibration_groups, vocab):
    fit_groups=set(fit_groups);calibration_groups=set(calibration_groups)
    if fit_groups&calibration_groups or vocab<1:raise ValueError('FIT/CAL group overlap or invalid vocabulary')
    counts=np.zeros(vocab,dtype=np.int64);seen=set();tokens=0
    for r in records:
        group=r['group_id']
        if group in calibration_groups or group not in fit_groups:raise ValueError('frequency record is outside FIT groups')
        if r['source_split']!='TRAIN' or group in seen:raise ValueError('original TRAIN only; one continuation per group')
        ids=np.asarray(r['target_tokens'])
        if ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=vocab):raise ValueError('target token IDs')
        counts+=np.bincount(ids,minlength=vocab);tokens+=len(ids);seen.add(group)
    if seen!=fit_groups:raise ValueError('every FIT group needs one authentic TRAIN continuation')
    return counts,{'groups':len(seen),'tokens':tokens,'observed_token_types':int(np.count_nonzero(counts))}


def static_ids(counts,k):
    counts=np.asarray(counts)
    if counts.ndim!=1 or not 1<=k<=len(counts) or np.any(counts<0):raise ValueError('static count/K geometry')
    # Unobserved positions fill the fixed K deterministically, without CAL data.
    return np.lexsort((np.arange(len(counts)),-counts))[:k].astype(np.int64)


def prefix(predictions, reference):
    return next((i for i,(a,b) in enumerate(zip(predictions,reference)) if a!=b),min(len(predictions),len(reference)))


def choose(ids,scores):
    """Exact support argmax with lowest absolute token ID breaking ties."""
    return torch.where(scores==scores.max(),ids,torch.iinfo(torch.int64).max).min()


def _phase(phase,name,fn):
    return fn() if phase is None else phase(name,fn)


def markov_bias(m1,m2,previous,phase=None):
    latent=_phase(phase,'markov_lookup',lambda:m1.index_select(0,previous.reshape(1)).reshape(-1))
    return _phase(phase,'full_markov_projection',lambda:m2@latent)


def full_chain(hidden,head,m1,m2,anchor,depth=5,phase=None,trace=False):
    base=_phase(phase,'full_base_projection',lambda:hidden@head.T)
    previous=torch.tensor(anchor,dtype=torch.int64,device=hidden.device);all_ids=torch.arange(len(head),device=hidden.device)
    predictions=[];parents=[];biases=[]
    for d in range(depth):
        parents.append(previous);bias=markov_bias(m1,m2,previous,phase)
        previous=_phase(phase,'argmax_selection',lambda:choose(all_ids,base[d]+bias));predictions.append(previous)
        if trace:biases.append(bias)
    result={'predictions':torch.stack(predictions)}
    if trace:result.update(parents=torch.stack(parents),biases=biases,base=base,supports=[None]*depth)
    return result


def dynamic_chain(hidden,head,m1,m2,basis,coefficients,anchor,k,depth=5,phase=None,trace=False):
    if not 1<=k<=len(head):raise ValueError('K outside full vocabulary')
    approximate=_phase(phase,'router_projection',lambda:((hidden.to(basis.dtype)@basis)@coefficients.T).float())
    previous=torch.tensor(anchor,dtype=torch.int64,device=hidden.device)
    predictions=[];parents=[];biases=[];supports=[]
    for d in range(depth):
        parents.append(previous)
        # Re-evaluate the full original Markov distribution for the ACTUAL new
        # predecessor, including after the first difference from the full chain.
        bias=markov_bias(m1,m2,previous,phase)
        ids=_phase(phase,'combine_and_topk',lambda:torch.topk(approximate[d]+bias,k,sorted=False).indices)
        gathered=_phase(phase,'gather_original_rows',lambda:head.index_select(0,ids))
        # Deliberately explicit copied-row F32 packing, charged in whole timing.
        # No claim to implement native Q8_1 activation packing/integer MMVQ.
        packed=_phase(phase,'pack_copied_rows',lambda:gathered.clone(memory_format=torch.contiguous_format))
        exact=_phase(phase,'selected_row_projection',lambda:packed@hidden[d]+bias.index_select(0,ids))
        previous=_phase(phase,'argmax_selection',lambda:choose(ids,exact));predictions.append(previous)
        if trace:biases.append(bias);supports.append(ids)
    result={'predictions':torch.stack(predictions)}
    if trace:result.update(parents=torch.stack(parents),biases=biases,supports=supports,approximate_base=approximate)
    return result


def pack_static(head,ids,phase=None):
    gathered=_phase(phase,'gather_original_rows',lambda:head.index_select(0,ids))
    return _phase(phase,'pack_copied_rows',lambda:gathered.clone(memory_format=torch.contiguous_format))


def static_chain(hidden,packed,ids,m1,m2,anchor,depth=5,phase=None,trace=False):
    # The fixed head rows are persistently packed once. Report that measured
    # preparation separately, both amortized and fully charged to one call.
    base=_phase(phase,'static_base_projection',lambda:hidden@packed.T)
    previous=torch.tensor(anchor,dtype=torch.int64,device=hidden.device)
    predictions=[];parents=[];biases=[]
    for d in range(depth):
        parents.append(previous);bias=markov_bias(m1,m2,previous,phase)
        scores=_phase(phase,'restrict_full_markov',lambda:base[d]+bias.index_select(0,ids))
        previous=_phase(phase,'argmax_selection',lambda:choose(ids,scores));predictions.append(previous)
        if trace:biases.append(bias)
    result={'predictions':torch.stack(predictions)}
    if trace:result.update(parents=torch.stack(parents),biases=biases,supports=[ids]*depth)
    return result


def trace_quality(result,decoded_base,native_chain,labels,anchor):
    predictions=result['predictions'].cpu().tolist();parents=result['parents'].cpu().tolist();rows=[]
    for d,(parent,support,bias) in enumerate(zip(parents,result['supports'],result['biases'])):
        full_winner=int(torch.argmax(decoded_base[d]+bias).item())
        native_parent=anchor if d==0 else native_chain[d-1]
        ids=None if support is None else support.cpu().tolist()
        rows.append({'depth':d,'actual_predecessor':parent,'native_predecessor':native_parent,
            'decoded_conditional_full_winner':full_winner,'decoded_winner_in_support':True if ids is None else full_winner in ids,
            'native_winner_in_support_same_predecessor':(True if ids is None else native_chain[d] in ids) if parent==native_parent else None})
    return {'predictions':predictions,'native_chain':native_chain,'target_labels':labels,'prefix_vs_native':prefix(predictions,native_chain),
        'prefix_vs_target':prefix(predictions,labels),'native_prefix_vs_target':prefix(native_chain,labels),
        'raw_native_chain_disagreements':sum(x!=y for x,y in zip(predictions,native_chain)),
        'conditional_rows':rows}
