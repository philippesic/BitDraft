#!/usr/bin/env python3
"""CUDA-only current-data vocabulary pilots; no native/server speed claim.

Dynamic: weight-space rank64/128 router, FULL original Markov + topK, exact
selected decoded-row scoring. Static: original TRAIN-target frequency rows,
persistently packed once, FULL Markov retained. Zero training updates.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
import torch
import vocabulary_core as core
import vocabulary_data as data


def require_cuda(device):
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('CUDA required; no CPU fallback or synthetic CUDA timings')
    if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 5080':raise ValueError('qualified RTX5080 device required')


class PhaseEvents:
    """Intrusive component diagnostic, measured separately from clean samples."""
    def __init__(self):self.events=[]
    def __call__(self,name,fn):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();result=fn();end.record();self.events.append((name,start,end));return result
    def summary(self):
        result={}
        for name,start,end in self.events:
            item=result.setdefault(name,{'calls':0,'cuda_event_ms':0.})
            item['calls']+=1;item['cuda_event_ms']+=start.elapsed_time(end)
        return result


def measured(fn,device,profile=False,readback=True):
    require_cuda(device)
    phase=PhaseEvents() if profile else None
    torch.cuda.synchronize(device);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    before=time.perf_counter();start.record();value=fn(phase)
    tokens=None
    if readback:
        action=lambda:value['predictions'].detach().cpu().tolist()
        tokens=action() if phase is None else phase('token_readback',action)
    end.record();torch.cuda.synchronize(device)
    record={'cuda_event_ms':start.elapsed_time(end),'wall_ms':(time.perf_counter()-before)*1000}
    if phase is not None:record['phases']=phase.summary()
    return value,tokens,record


def timing_samples(fn,device,repeats,warmup,expected):
    for _ in range(warmup):measured(fn,device)
    clean=[];stable=True
    for _ in range(repeats):
        _,tokens,record=measured(fn,device);stable&=tokens==expected;clean.append(record)
    _,tokens,profile=measured(fn,device,True);stable&=tokens==expected
    return {'clean_samples':clean,'median_cuda_event_ms':statistics.median(x['cuda_event_ms'] for x in clean),
        'median_wall_ms':statistics.median(x['wall_ms'] for x in clean),'predictions_stable_across_runs':stable,
        'component_profile_separate_intrusive_call':profile,
        'scope':'Full seven-row head path + five full Markov/selection stages + actual host token readback; no body/target/server/native quantized arithmetic'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ranks',default='64,128');p.add_argument('--ks',default='256,1024,4096');p.add_argument('--include-k16384',action='store_true')
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmup',type=int,default=2);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--router-dtype',choices=['float16','float32'],default='float16');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();device=torch.device(a.device);require_cuda(device)
    ranks=sorted(set(map(int,a.ranks.split(','))));ks=sorted(set(map(int,a.ks.split(','))))
    if a.include_k16384:ks=sorted(set(ks+[16384]))
    if any(x not in (64,128) for x in ranks) or any(x not in (256,1024,4096,16384) for x in ks):p.error('fixed pilot ranks64/128 and K256/1024/4096/16384 only')
    if not ranks or not ks or a.repeats<1 or a.warmup<0:p.error('nonempty ranks/K, positive repeats and nonnegative warmup')
    if a.out.exists():p.error('fresh output required')
    a.out.mkdir(parents=True);started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    if hasattr(torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    packet,cases,fit,cal=data.load_packet(a.packet,a.arm)
    original_records,record_refs=data.original_train_records(packet,fit,cal)
    counts,frequency=core.fit_frequencies(original_records,fit,cal,151936)
    cpu_load_start=time.perf_counter();weights,storage=data.load_weights(packet['model']['path'])
    weights={name:value.to(device) for name,value in weights.items()};torch.cuda.synchronize(device)
    load_s=time.perf_counter()-cpu_load_start
    head,m1,m2=weights['head'],weights['m1'],weights['m2']
    report={'schema':'current_vocabulary_reference_pilot_v1','status':'STARTED','arm':a.arm,'packet':data.pin(a.packet),'model':packet['model'],
        'target':packet['target'],'target_kv':'F16','training_updates':0,'body_rows':7,'submitted_depth':5,
        'source_files':{f.name:data.pin(f) for f in Path(__file__).parent.glob('*.py')},'head_storage':storage,
        'frequency':{**frequency,'fit_groups':sorted(fit),'excluded_calibration_groups':sorted(cal),'original_TRAIN_receipts':record_refs},
        'hardware':{'name':torch.cuda.get_device_name(device),'torch':torch.__version__,'device':str(device)},
        'model_decode_and_transfer_wall_s':load_s,'preparation':{},'cases':[],
        'fixed_configuration':{'ranks':ranks,'ks':ks,'seed':a.seed,'router_dtype':a.router_dtype,'repeats':a.repeats,'warmup':a.warmup},
        'historical_distinction':'Prior rank128/K256 TRAIN-PCA shortlist and static frontier are existing evidence. This cycle uses current matched W1/Q4 captures, global head-weight router basis, multiple fixed K and actual CUDA-inclusive unfused costs; no novelty or rediscovered throughput claim.',
        'limitations':['Untrained weight-space router adaptation of SpecVocab; not its jointly trained router or fused indexed kernel.',
            'Exact selected-row scores mean F32 on decoded original GGUF values; native Q8_1 packing/integer reduction rounding is not reproduced.',
            'Explicit gathered-row F32 clone/repack is charged; no native quantized repacking/activation kernel is implemented.',
            'CUDA event/wall timings compare portable operations against a decoded F32 full-head reference, not native Q4_K/Q6_K speed or full requests.',
            'Only three reused group-separated CAL cases, one/domain; no request panels used or production quality bound.',
            'Torch topK support cutoff ties are not guaranteed stable. Selected-score ties use lowest absolute token ID; actual prediction repeatability is reported.']}
    dtype=torch.float16 if a.router_dtype=='float16' else torch.float32
    with torch.no_grad():
        # The shared nested basis sees only the actual original full head weights.
        # It does not read TRAIN/CAL hidden states or token labels.
        def make_basis(_):
            _,_,right=torch.svd_lowrank(head,q=max(ranks)+16,niter=1)
            return right[:,:max(ranks)].contiguous()
        basis,_,svd_timing=measured(make_basis,device,readback=False)
        routers={}
        for rank in ranks:
            def make_router(_,rank=rank):
                down=basis[:,:rank].contiguous();up=head@down
                return down.to(dtype).contiguous(),up.to(dtype).contiguous()
            router,_,record=measured(make_router,device,readback=False);routers[rank]=router
            report['preparation']['router_rank'+str(rank)]={'timing':record,'factor_bytes':sum(x.numel()*x.element_size() for x in router),'data_used':'global original head weights only'}
        report['preparation']['global_weight_SVD']=svd_timing
        static={}
        for k in ks:
            selection_start=time.perf_counter();ids_np=core.static_ids(counts,k);selection_s=time.perf_counter()-selection_start
            ids=torch.from_numpy(ids_np).to(device)
            packed,_,record=measured(lambda phase:core.pack_static(head,ids,phase),device,profile=True,readback=False)
            static[k]=(ids,packed)
            report['preparation']['static_K'+str(k)]={'cpu_frequency_selection_s':selection_s,'row_pack':record,
                'observed_frequency_tokens_in_subset':int(np.count_nonzero(counts[ids_np])),
                'zero_frequency_fill_positions':int(np.sum(counts[ids_np]==0)),
                'persistently_packed_F32_bytes':packed.numel()*packed.element_size(),
                'copy_bytes_gather':k*2560*4,'copy_bytes_repack':k*2560*4,
                'amortization':'Packed once for a fixed vocabulary; report run time and run+entire-pack cost separately.'}
        for c in [x for x in cases if x['split']=='calibration']:
            h=data.hidden(c).to(device);native=c['native_argmax'][:5];labels=c['labels'][:5]
            base_trace=core.full_chain(h,head,m1,m2,c['anchor_token'],trace=True)
            decoded=base_trace['predictions'].cpu().tolist();decoded_base=base_trace['base']
            entry={'case_index':c['case_index'],'group_id':c['group_id'],'domain':c['domain'],
                'native_predictions':native,'native_prefix_vs_target':core.prefix(native,labels),
                'decoded_full_reference':core.trace_quality(base_trace,decoded_base,native,labels,c['anchor_token']),
                'decoded_full_native_chain_prefix':core.prefix(decoded,native),
                'decoded_full_timing':timing_samples(lambda phase:core.full_chain(h,head,m1,m2,c['anchor_token'],phase=phase),device,a.repeats,a.warmup,decoded),
                'candidates':[]}
            for rank in ranks:
                down,up=routers[rank]
                for k in ks:
                    fn=lambda phase,trace=False,down=down,up=up,k=k:core.dynamic_chain(h,head,m1,m2,down,up,c['anchor_token'],k,phase=phase,trace=trace)
                    trace=fn(None,True);quality=core.trace_quality(trace,decoded_base,native,labels,c['anchor_token'])
                    entry['candidates'].append({'method':'dynamic','rank':rank,'k':k,'quality':quality,
                        'timing':timing_samples(fn,device,a.repeats,a.warmup,quality['predictions']),
                        'per_call_F32_gather_copy_bytes':5*k*2560*4,'per_call_F32_repack_copy_bytes':5*k*2560*4,
                        'full_Markov_projection_rows':5,'seven_row_router_projection':True})
            for k in ks:
                ids,packed=static[k]
                fn=lambda phase,trace=False,ids=ids,packed=packed:core.static_chain(h,packed,ids,m1,m2,c['anchor_token'],phase=phase,trace=trace)
                trace=fn(None,True);quality=core.trace_quality(trace,decoded_base,native,labels,c['anchor_token']);timing=timing_samples(fn,device,a.repeats,a.warmup,quality['predictions'])
                pack=report['preparation']['static_K'+str(k)]['row_pack']
                entry['candidates'].append({'method':'static_frequency','rank':None,'k':k,'quality':quality,'timing':timing,
                    'fully_unamortized_run_plus_pack_cuda_ms':timing['median_cuda_event_ms']+pack['cuda_event_ms'],
                    'fully_unamortized_run_plus_pack_wall_ms':timing['median_wall_ms']+pack['wall_ms'],
                    'full_Markov_projection_rows':5,'per_call_row_gather_and_repack_bytes':0,'persistent_static_rows_bytes':packed.numel()*packed.element_size()})
            report['cases'].append(entry)
            (a.out/'partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    summaries={}
    for case in report['cases']:
        for candidate in case['candidates']:
            key=f"{candidate['method']}/r{candidate['rank']}/K{candidate['k']}";q=candidate['quality'];s=summaries.setdefault(key,{'cases':0,'native_prefix_sum':0,'candidate_prefix_sum':0,'prefix_vs_native_sum':0,'native_winner_hits_same_parent':0,'positions_same_parent':0,'cuda_ms':[],'wall_ms':[]})
            s['cases']+=1;s['native_prefix_sum']+=q['native_prefix_vs_target'];s['candidate_prefix_sum']+=q['prefix_vs_target'];s['prefix_vs_native_sum']+=q['prefix_vs_native']
            for row in q['conditional_rows']:
                if row['native_winner_in_support_same_predecessor'] is not None:s['positions_same_parent']+=1;s['native_winner_hits_same_parent']+=int(row['native_winner_in_support_same_predecessor'])
            s['cuda_ms'].append(candidate['timing']['median_cuda_event_ms']);s['wall_ms'].append(candidate['timing']['median_wall_ms'])
    for s in summaries.values():s['median_case_cuda_ms']=statistics.median(s.pop('cuda_ms'));s['median_case_wall_ms']=statistics.median(s.pop('wall_ms'))
    report.update(status='CUDA_COMPONENT_COMPLETE_NO_NATIVE_THROUGHPUT_CLAIM',summary=summaries,
        decoded_full_native_disagreements=sum(c['decoded_full_reference']['raw_native_chain_disagreements'] for c in report['cases']),
        wall_s=time.perf_counter()-started,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device))
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.out/'partial.json').unlink()
    print(json.dumps({'status':report['status'],'report':str(a.out/'report.json'),'summary':summaries,'decoded_full_native_disagreements':report['decoded_full_native_disagreements'],'training_updates':0}))
if __name__=='__main__':main()
