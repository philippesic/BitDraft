"""CPU replay of actual serving EAGLE inputs with explicit rollback/cache ancestry.

This consumes supplied decoder FC/pre-norm inputs verbatim. It never substitutes
conditional teacher features. All outputs are diagnostic on a reused request.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import torch

P = Path('/home/philip/binary-eagle-decoding')
W = Path('/home/philip/.codex/worktrees/w1a8-modern-replay-20261103')
R = P/'runs/w1a8-modern-actual-actor-replay-20261103-01'
sys.path[:0] = [str(W/'src'), str(P/'third_party/llama.cpp/gguf-py')]
spec = importlib.util.spec_from_file_location('modern_actor', R/'modern_actor.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from w1a1_eagle.native_step import NativeStepCache
from w1a1_eagle.recurrent_qat import shared_round_hard_signs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--trace',type=Path,default=P/'runs/w1a8-modern-serving-trace-20261103-03/prompt-02/trace')
    parser.add_argument('--embedding-root',type=Path,default=P/'runs/w1a8-modern-actual-embedding-20261103-01')
    parser.add_argument('--out',type=Path,default=R/'owned-report.json')
    args=parser.parse_args()
    require(args.out.resolve().is_relative_to(R.resolve()) and not args.out.exists(),'output must be new and inside assigned run')
    started=time.perf_counter()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    pins={}

    def pin(path):
        path=Path(path).resolve()
        # All target/HF weight files are excluded; only the packed draft constructor opens a GGUF.
        require(path.suffix not in ('.gguf','.safetensors','.npz'),'weight access outside packed actor constructor forbidden')
        if str(path) not in pins:
            pins[str(path)]=module.file_pin(path)
        return pins[str(path)]

    def checked(ref):
        actual=pin(ref['path'])
        require(actual['sha256']==ref['sha256'],'input pin changed: '+str(ref['path']))
        if 'bytes' in ref:
            require(actual['bytes']==ref['bytes'],'input size changed')
        return Path(ref['path'])

    def read(path):
        pin(path);return json.loads(Path(path).read_text())

    def f32(path,shape):
        pin(path)
        require(Path(path).stat().st_size==int(np.prod(shape))*4,'F32 shape mismatch: '+str(path))
        value=np.fromfile(path,dtype='<f4').reshape(shape)
        require(np.isfinite(value).all(),'nonfinite saved F32')
        return value

    def traced(spec):
        path=(args.trace/spec['file']).resolve()
        require(path.parent==args.trace.resolve(),'trace F32 path escaped source directory')
        return f32(path,tuple(spec['shape']))

    def diff(a,b):
        require(a.shape==b.shape and a.size>0,'comparison shape mismatch')
        aa,bb=a.astype(np.float64),b.astype(np.float64)
        delta=aa-bb
        return {'max_abs':float(np.abs(delta).max()),'mean_abs':float(np.abs(delta).mean()),
                'rms':float(np.sqrt(np.mean(delta*delta))),
                'relative_l2_native_denominator':float(np.linalg.norm(delta)/max(np.linalg.norm(bb),1e-30)),
                'byteexact':a.tobytes()==b.tobytes()}

    def digest(value):
        return hashlib.sha256(np.ascontiguousarray(value).astype('<f4',copy=False).tobytes()).hexdigest()

    pin(__file__);pin(R/'modern_actor.py')
    for name in ('native_step.py','recurrent_qat.py','recurrent_binary.py'):
        pin(W/'src/w1a1_eagle'/name)
    manifest=read(args.trace/'manifest.json')
    require(manifest['schema']=='w1a8_actual_native_actor_trace_v2','actual trace must use corrected explicit candidate-ID schema')
    require(not (args.trace/'failure.json').exists(),'trace explicitly aborted')
    logpath=args.trace/'calls.jsonl';pin(logpath)
    events=[json.loads(x) for x in logpath.read_text().splitlines()]
    require([x['event'] for x in events]==list(range(1,len(events)+1)),'trace event chronology missing')
    befores={x['call']:x for x in events if x['phase']=='before'}
    afters={x['call']:x for x in events if x['phase']=='after'}
    require(set(befores)==set(afters) and len(events)==2*len(befores),'incomplete/duplicate trace calls')
    require(all(x['result']==0 for x in afters.values()),'native call failed')
    native_map=np.loadtxt(manifest['draft_map'],dtype=np.int64,ndmin=1);pin(manifest['draft_map'])
    require(native_map.shape==(50000,) and len(np.unique(native_map))==50000,'native map invalid')
    inverse=np.full(248320,-1,dtype=np.int64);inverse[native_map]=np.arange(50000)
    export=read(P/'runs/w1a8-modern-packed-init-20261103-01/export.json')
    require(hashlib.sha256(native_map.astype('<i8').tobytes()).hexdigest()==export['preserved_tensors']['d2t']['raw_sha256'],
            'actual trace map differs from frozen native initializer')
    receipt=read(P/'runs/w1a8-modern-admission-20261103-01/prompt-02-admitted.json')
    old=read(checked(receipt['evidence']['capture']))
    oldshape=tuple(old['feature_shape'])
    oldarrays={tap:f32(checked(receipt['evidence']['arrays']['layer-%d.f32'%tap]),oldshape) for tap in (0,2,16,29)}
    oldraw=np.concatenate([oldarrays[t] for t in (2,16,29)],axis=1)
    oldtokens=old['input_tokens']
    embedding_freeze=read(args.embedding_root/'freeze.json')
    embedding_result=read(args.embedding_root/'result.json')
    require(embedding_result['status']=='NATIVE_EMBEDDING_ROWS_CAPTURED','native embedding capture incomplete')
    require(embedding_freeze['target']==receipt['evidence']['target'],'native embeddings use wrong target ancestry')
    for key in ('source','tokens','build'):
        checked(embedding_freeze[key])
    emeta=read(checked(embedding_result['capture']))
    ematrix=f32(checked(embedding_result['layer0']),tuple(emeta['feature_shape']))
    require(emeta['taps']==[0] and emeta['dtype']=='little_endian_float32' and
            emeta['model']==embedding_freeze['target']['path'],'embedding native capture contract differs')
    native_embeddings={token:torch.from_numpy(ematrix[i].copy()) for i,token in enumerate(emeta['input_tokens'])}
    require(len(native_embeddings)==len(emeta['input_tokens']),'duplicate embedding input IDs')
    overlap=[]
    for i,token in enumerate(oldtokens):
        if token in native_embeddings:
            require(np.array_equal(native_embeddings[token].numpy(),oldarrays[0][i]),'new native embedding differs from old admitted layer0')
            overlap.append(token)
    used=set();warmup=[]
    for event in befores.values():
        if event['architecture']=='eagle3' and event['process_type_name']=='decode':
            if 'embeddings' not in event['batch']:
                warmup.append(event['call']);continue
            used.update(x['token'] for x in event['batch']['rows'])
    require(used<=set(native_embeddings),'native embedding row unavailable for actual trace token')
    lookup=module.ExternalNativeF32Embedding(lookup=native_embeddings.__getitem__,vocab_size=248320,hidden_size=2560,
        target_identity=receipt['target_id'],target_sha256=embedding_freeze['target']['sha256'],
        capture_sha256=embedding_result['capture']['sha256'])
    stage=time.perf_counter()
    actor=module.build_modern_actor(P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',
        P/'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json',embedding_lookup=lookup,scale_only=True)
    construct_seconds=time.perf_counter()-stage
    adapter=actor.adapter
    cache=adapter.new_cache()
    target_history=[];target_info={};encoder_cursor={};source_by_hash={};owned_source_by_native_hash={}
    target_batches=[];encoder_results=[];decoder_results=[];rollback_records=[]
    selected_states=[];selected_logits=[];fc_states=[]
    stage=time.perf_counter()
    with torch.inference_mode(),shared_round_hard_signs(actor.linears):
        for event in events:
            if event['phase']!='before':
                continue
            call=event['call'];batch=event['batch'];rows=batch['rows'];after=afters[call]
            if event['architecture'].startswith('qwen35'):
                first=rows[0]['positions'][0]
                require(first<=len(target_history),'target history has an unobserved position hole')
                target_history=target_history[:first]
                matches=[]
                for row in rows:
                    pos=row['positions'][0]
                    require(row['seq_ids']==[0] and pos==len(target_history),'target single-sequence contiguous history required')
                    target_history.append(row['token'])
                    matches.append(pos<len(oldtokens) and target_history==oldtokens[:pos+1])
                target_info[call]={'rows':rows,'same_old_prefix':matches}
                target_batches.append({'call':call,'positions':[r['positions'][0] for r in rows],
                                       'tokens':[r['token'] for r in rows]})
                continue
            if event['architecture']!='eagle3':
                continue
            if event['process_type_name']=='encode':
                require(batch['n_embd']==7680 and all(r['has_embd'] for r in rows),'unexpected native ENC input geometry')
                raw=traced(batch['embeddings']);native=traced(after['encoded_fc'])
                require(raw.shape==(len(rows),7680) and native.shape==(len(rows),2560),'ENC rows misaligned')
                computed=actor.drafter.fc(torch.from_numpy(raw)).numpy()
                upstream=event['active_target_call'] or event['last_completed_target_call']
                require(upstream in target_info,'ENC lacks authenticated upstream target batch')
                offset=encoder_cursor.get(upstream,0)
                target_rows=target_info[upstream]['rows'][offset:offset+len(rows)]
                require(len(target_rows)==len(rows),'ENC chunk exceeds upstream target rows')
                entries=[]
                for i,row in enumerate(rows):
                    require(row['saved_embedding_row']==i,'ENC embedding row index differs')
                    targetpos=target_rows[i]['positions'][0]
                    item={'encoder_row':i,'placeholder_position':row['positions'][0],
                          'actual_target_position':targetpos,'actual_target_token':target_rows[i]['token'],
                          'fc_cpu_minus_native':diff(computed[i],native[i]),
                          'same_old_conditional_prefix':target_info[upstream]['same_old_prefix'][offset+i]}
                    if item['same_old_conditional_prefix']:
                        item['raw_feature_actual_minus_old_conditional']=diff(raw[i],oldraw[targetpos])
                        item['per_tap_actual_minus_old_conditional']={str(t):diff(raw[i,j*2560:(j+1)*2560],oldarrays[t][targetpos])
                                                                       for j,t in enumerate((2,16,29))}
                    owned_source_by_native_hash[digest(native[i])]=computed[i].copy()
                    source_by_hash.setdefault(digest(native[i]),[]).append({'source':'encoder_fc','call':call,
                        'row':i,'target_call':upstream,'target_position':targetpos})
                    entries.append(item);fc_states.append(computed[i].copy())
                encoder_cursor[upstream]=offset+len(rows)
                encoder_results.append({'call':call,'upstream_target_call':upstream,'rows':entries,
                                        'whole_fc_cpu_minus_native':diff(computed,native)})
                continue
            if call in warmup:
                continue
            require(event['process_type_name']=='decode' and batch['n_embd']==2560,'unexpected native decoder geometry')
            require(all(row['has_embd'] and not row['has_state'] and row['seq_ids']==[0] for row in rows),
                    'actual decoder must be supplied FC state and single sequence')
            supplied=traced(batch['embeddings'])
            require(supplied.shape==(len(rows),2560),'actual decoder FC rows misaligned')
            first=rows[0]['positions'][0]
            length=cache.key.shape[1]
            require(first<=length,'decoder cache has an unobserved position hole')
            if first<length:
                rollback_records.append({'call':call,'cache_before':length,'first_position':first,'discarded_rows':length-first})
                cache=NativeStepCache(cache.key[:,:first,:].contiguous(),cache.value[:,:first,:].contiguous())
            selected={x['batch_row']:x for x in after.get('selected',[])}
            require(set(selected)=={i for i,r in enumerate(rows) if r['output']},'selected native output row metadata differs')
            native_states=traced(after['prenorm']) if selected else None
            for i,row in enumerate(rows):
                pos=row['positions'][0]
                require(row['saved_embedding_row']==i and pos==cache.key.shape[1],'decoder row/cache position mismatch')
                require(digest(supplied[i]) in owned_source_by_native_hash,'actor-owned source state unavailable')
                feature=torch.from_numpy(owned_source_by_native_hash[digest(supplied[i])].copy())
                output=(adapter.decode_step(row['token'],feature,pos,cache) if row['output'] else
                        adapter.decode_context(row['token'],feature,pos,cache))
                cache=output.cache
                item={'call':call,'batch_row':i,'decoder_position':pos,'input_token':row['token'],
                      'output_selected':row['output'],'input_fc_or_prenorm_sources':source_by_hash.get(digest(supplied[i]),[]),
                      'cache_positions_after':cache.key.shape[1]}
                if row['output']:
                    nrow=selected[i];saved=nrow['saved_row'];state=output.pre_norm.numpy();logits=output.logits.numpy()
                    native_state=native_states[saved]
                    item['prenorm_cpu_minus_native']=diff(state,native_state)
                    item['cpu_proposed']=int(native_map[int(logits.argmax())])
                    mode=nrow['logits_mode'];item['native_logits_mode']=mode
                    item['native_proposed']=nrow['argmax_target_token']
                    item['proposal_matches']=item['cpu_proposed']==item['native_proposed']
                    if mode=='backend_explicit_candidate_pairs':
                        ids=np.array(nrow['candidate_ids'],dtype=np.int64)
                        nlogits=traced(nrow['candidate_logits']).reshape(-1)
                        require(len(ids)==nrow['candidate_count']==len(nlogits) and len(np.unique(ids))==len(ids),
                                'backend candidate pairs misaligned')
                        classes=inverse[ids]
                        require((classes>=0).all(),'native sampled candidate lacks draft support')
                        require(int(ids[int(nlogits.argmax())])==item['native_proposed'],'native candidate argmax metadata wrong')
                        item['candidate_ids']=ids.tolist()
                        item['cpu_minus_native_candidate_logits']=diff(logits[classes],nlogits)
                        item['cpu_ranked_candidate_top1']=int(ids[int(logits[classes].argmax())])
                        item['native_backend_sampled_token']=nrow['backend_sampled_token']
                    elif mode=='dense_target_vocab_mapped_subset':
                        nlogits=traced(nrow['mapped_logits']).reshape(50000)
                        require(int(native_map[int(nlogits.argmax())])==item['native_proposed'],'native dense argmax metadata wrong')
                        item['cpu_minus_native_dense_logits']=diff(logits,nlogits)
                    else:
                        raise ValueError('unknown actual native logits mode')
                    item['cpu_saved_row']=len(selected_states)
                    selected_states.append(state.copy());selected_logits.append(logits.copy())
                    owned_source_by_native_hash[digest(native_state)]=state.copy()
                    source_by_hash.setdefault(digest(native_state),[]).append({'source':'native_decoder_prenorm','call':call,'row':saved})
                decoder_results.append(item)
    replay_seconds=time.perf_counter()-stage
    selected=[x for x in decoder_results if x['output_selected']]
    for filename,values in (('owned-cpu-selected-prenorm.f32',selected_states),('owned-cpu-selected-logits.f32',selected_logits),
                            ('owned-cpu-encoder-fc.f32',fc_states)):
        path=R/filename
        with path.open('xb') as stream:
            stream.write(np.stack(values).astype('<f4').tobytes())
        pin(path)
    sameprefix=[{'encoder_call':e['call'],**r} for e in encoder_results for r in e['rows'] if r['same_old_conditional_prefix']]
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(peak<6*1024**3,'actual actor replay exceeded6GiB RSS')
    result={'status':'ACTOR_OWNED_STATE_CPU_REPLAY_COMPLETE','trace_manifest':manifest,
            'trace_root':str(args.trace.resolve()),'embedding_capture':embedding_result,
            'actual_decode_calls':len({x['call'] for x in decoder_results}),'actual_decode_rows':len(decoder_results),
            'selected_rows':len(selected),'proposal_matches':sum(x['proposal_matches'] for x in selected),
            'proposal_mismatches':[x for x in selected if not x['proposal_matches']],
            'warmup_calls_skipped_without_supplied_embedding':warmup,'cache_rollbacks':rollback_records,
            'target_batches':target_batches,'encoder_results':encoder_results,'decoder_results':decoder_results,
            'conditional_same_prefix_feature_comparisons':sameprefix,'initializer':actor.initializer,
            'config':actor.config,'actor_contract':actor.contract,'input_and_output_pins':list(pins.values()),
            'old_and_fresh_native_embedding_overlap_rows_checked':len(overlap),'used_embedding_ids':len(used),
            'construct_seconds':construct_seconds,'replay_seconds':replay_seconds,
            'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':peak,'gpu_seconds':0,'optimizer_steps':0,
            'target_model_opened_or_hashed':False,'hf_weights_opened':False,
            'arithmetic':'Unchanged CPU RowBinaryLinear A8; native raw FC inputs recomputed through CPU FC and CPU pre-norm carried through native source links; inference only',
            'scope':'Reused P2 actor-owned-state diagnostic on native token/cache trajectory; prefix qualification must stop at first gold divergence; no gradient/update, generalization or throughput proof',
            'native_integer_arithmetic_parity':False}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','actual_decode_calls','actual_decode_rows','selected_rows','proposal_matches',
          'construct_seconds','replay_seconds','wall_seconds','peak_rss_bytes')},indent=2),flush=True)


if __name__=='__main__':
    main()
