#!/usr/bin/env python3
"""Seven-row native vocabulary conditional trace/reference gate (CPU only).

The original quantized FULL Markov projection is read from native trace, not
replaced by decoded-F32 arithmetic. We independently check its actual preceding
ID/lookup, combined-score support, F16 selected-weight dot products, scattered
logits and all seven resulting tokens. This is not full native Q8_1 emulation.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from export import g,load_rows
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vocabulary'))
from vocabulary_data import pin

def tensor(gg,name,dtype):
    t=gg.tensors[name]
    with gg.path.open('rb') as f:f.seek(gg.data_start+t.offset);raw=f.read(t.nbytes)
    return np.frombuffer(raw,dtype=dtype).reshape(tuple(reversed(t.shape))).copy()

def trace(row,name):
    r=row['vocabulary_trace'][name];a=np.fromfile(r['path'],dtype='<i4' if r['dtype']=='int32' else '<f4')
    if a.nbytes!=r['bytes'] or a.size!=np.prod(r['ne']):raise ValueError('trace size: '+name)
    return a.reshape(tuple(reversed(r['ne']))).reshape(-1,r['ne'][0])

def prefix(a,b):return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))

def chosen(ids,scores):
    return int(np.min(ids[scores==np.max(scores)]))

def projected(weights,h,activation):
    # Distinguish ordinary GGML vector F32 activation from cuBLAS F16 conversion;
    # both retain actual F16 weights and accumulate independent CPU F32.
    x=h.astype(np.float16).astype(np.float32) if activation=='f16' else h
    return weights.astype(np.float32)@x

def compare(source,exported,rows_path,control_path=None):
    original,head=load_rows(source);model=g.GGUF(exported)
    if model.get('dflash.vocab_source_sha256')!=g.sha(source):raise ValueError('export/source binding')
    mode=model.get('dflash.vocab_mode');dynamic=mode=='dynamic-r64-k1024';k=1024 if dynamic else 4096
    if not dynamic and mode!='static-k4096':raise ValueError('fixed vocabulary variant required')
    static_ids=None if dynamic else tensor(model,'vocab_static_ids','<i4').reshape(-1).astype(np.int64)
    packed_static=None if dynamic else tensor(model,'vocab_static_head.weight','<f2').astype(np.float32)
    down=tensor(model,'vocab_router_down.weight','<f2').astype(np.float32) if dynamic else None
    up=tensor(model,'vocab_router_up.weight','<f2').astype(np.float32) if dynamic else None
    # Original Markov lookup is pure get_rows dequantization; independently verify
    # the native lookup came from its ACTUAL predecessor after any divergence.
    mt=original.tensors['markov_w1.weight'];Rows=__import__('quant_rows').StoredRows;m1=Rows(original,mt.name,mt.shape,mt.kind)
    rows=[json.loads(x) for x in Path(rows_path).read_text().splitlines() if x.strip()]
    if not 1<=len(rows)<=12:raise ValueError('1..12 authentic native cases required')
    controls={} if control_path is None else {r['case_index']:r for r in [json.loads(x) for x in Path(control_path).read_text().splitlines() if x.strip()]}
    result=[]
    for row in rows:
        h=np.fromfile(row['hidden_tensors']['result_norm']['path'],dtype='<f4').reshape(7,2560)
        native=np.fromfile(row['raw_path'],dtype='<f4').reshape(7,151936)
        if np.isnan(native).any() or np.isposinf(native).any() or not np.isfinite(h).all():raise ValueError('NaN/+inf native or invalid hidden')
        base=trace(row,'vocab_base_projection');base=base.reshape(7,151936 if dynamic else k)
        predictions=np.argmax(native,axis=1).tolist();checks=[];ref_predictions={'f32':[],'f16':[]}
        if dynamic:
            router_f32=(h@down.T)@up.T;router_f16=(h.astype(np.float16).astype(np.float32)@down.T).astype(np.float16).astype(np.float32)@up.T
            router_differences={key:float(np.max(np.abs(base-value))) for key,value in [('f32_activation',router_f32),('f16_activation_intermediate',router_f16)]}
        else:router_differences={}
        for depth in range(7):
            bias=trace(row,'dspark_markov_projection-'+str(depth)).reshape(-1);latent=trace(row,'dspark_markov_lookup-'+str(depth)).reshape(-1)
            if len(bias)!=151936 or not np.isfinite(bias).all():raise ValueError('FULL finite native Markov projection required at every row')
            ids=trace(row,'vocab_combined_topk-'+str(depth)).reshape(-1).astype(np.int64) if dynamic else static_ids
            if len(ids)!=k or len(set(ids.tolist()))!=k or np.min(ids)<0 or np.max(ids)>=151936:raise ValueError('unique full-vocabulary selected IDs')
            support=np.isfinite(native[depth]);mask=np.zeros(151936,bool);mask[ids]=True
            if not np.array_equal(support,mask):raise ValueError('finite support must equal IDs and all omitted logits be -inf')
            if dynamic:
                combined=base[depth]+bias;cutoff=float(np.min(combined[ids]));excluded=combined.copy();excluded[ids]=-np.inf
                support_ok=float(np.max(excluded))<=cutoff+1e-6
                packed=np.concatenate([head(ids[i:i+1024]) for i in range(0,k,1024)]).astype(np.float16).astype(np.float32)
                native_projection=trace(row,'vocab_selected_projection-'+str(depth)).reshape(-1)
            else:
                support_ok=True;packed=packed_static;native_projection=base[depth]
            expected_prev=int(row['anchor_token']) if 'anchor_token' in row else None
            if depth:expected_prev=predictions[depth-1]
            elif controls and 'anchor_token' in controls.get(row['case_index'],{}):expected_prev=int(controls[row['case_index']]['anchor_token'])
            # Probe output predates anchor_token; require the caller-provided
            # original cases join in main to populate it, never infer a parent.
            if expected_prev is None:raise ValueError('original case anchor_token join required')
            lookup_ok=np.allclose(latent,m1(np.array([expected_prev],np.int64)).reshape(-1),rtol=1e-6,atol=1e-6)
            if depth and int(trace(row,'dspark_markov_argmax-'+str(depth-1)).reshape(-1)[0])!=expected_prev:raise ValueError('Markov suffix did not use prior selected winner')
            projected_errors={};ref_tokens={};portable_tokens={};score_tolerance=0.05+0.001*float(np.max(np.abs(native_projection)))
            for activation in ('f32','f16'):
                scores=projected(packed,h[depth],activation)+bias[ids]
                ref_tokens[activation]=chosen(ids,scores)
                if dynamic:
                    approximate=router_f32 if activation=='f32' else router_f16
                    portable_ids=np.argpartition(approximate[depth]+bias,-k)[-k:]
                    portable_rows=head(portable_ids).astype(np.float16).astype(np.float32)
                    portable_scores=projected(portable_rows,h[depth],activation)+bias[portable_ids]
                    portable_tokens[activation]=chosen(portable_ids,portable_scores)
                else:portable_tokens[activation]=ref_tokens[activation]
                ref_predictions[activation].append(portable_tokens[activation])
                projected_errors[activation]=float(np.max(np.abs((scores-bias[ids])-native_projection)))
            selected_scatter_ok=np.allclose(native[depth,ids],native_projection+bias[ids],rtol=1e-6,atol=1e-6)
            checks.append({'depth':depth,'actual_predecessor':expected_prev,'finite_support':int(support.sum()),'full_combined_topk_valid':bool(support_ok),'original_predecessor_lookup_valid':bool(lookup_ok),'selected_bias_scatter_valid':bool(selected_scatter_ok),'native_token':predictions[depth],'independent_native_support_F16_weight_tokens':ref_tokens,'independent_router_and_selected_F16_weight_tokens':portable_tokens,'selected_projection_max_abs_error':projected_errors,'selected_projection_tolerance':score_tolerance})
        modes=[name for name,tokens in ref_predictions.items() if tokens==predictions and all(r['independent_native_support_F16_weight_tokens'][name]==r['native_token'] and r['selected_projection_max_abs_error'][name]<=r['selected_projection_tolerance'] for r in checks)]
        body=None
        if controls:
            control=controls[row['case_index']];body={name:pin(control['hidden_tensors'][name]['path'])['sha256']==pin(row['hidden_tensors'][name]['path'])['sha256'] for name in ('l_out-4','result_norm')}
        valid=bool(modes) and all(x['full_combined_topk_valid'] and x['original_predecessor_lookup_valid'] and x['selected_bias_scatter_valid'] for x in checks) and (body is None or all(body.values()))
        result.append({'case_index':row['case_index'],'pass':valid,'native_tokens7':predictions,'independent_reference_tokens7':ref_predictions,'matching_activation_modes':modes,'reference_prefix7':{name:prefix(tokens,predictions) for name,tokens in ref_predictions.items()},'native_prefix5_vs_target':prefix(predictions[:5],[x['label'] for x in row['rows'][:5]]),'body_bitwise_control':body,'router_F16_weight_reference_max_abs_error':router_differences,'rows':checks})
    return {'schema':'native_vocabulary_conditional_trace_v1','status':'SEVEN_ROW_CONDITIONAL_REFERENCE_PASS' if all(x['pass'] for x in result) else 'REFERENCE_FAILED','source':pin(source),'exported_model':pin(exported),'native_rows':pin(rows_path),'mode':mode,'body_rows':7,'native_Markov_projections_per_case':7,'cases':result,'scope':'Actual native seven-row conditional chain, full original native Markov traces + independent original-row F16-weight scoring. Candidate-prefix same-precision reference, independent F16-weight router/topK/indexed-rerank token check; not independent Q8_1 Markov emulation, router cutoff bitwise parity or throughput. Diagnostic callback overhead excluded from subsequent HTTP timing.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','model','rows','cases','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--control-rows',type=Path);a=p.parse_args()
    if a.out.exists():p.error('fresh evidence required')
    # Add exact anchor tokens to a temporary diagnostic rows view; preserve raw
    # rows and case receipts and bind both originals in final evidence.
    import tempfile
    source_cases=json.loads(a.cases.read_text());cases={c['case_index']:c for c in source_cases['cases']}
    raw=[json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    for row in raw:
        case=cases[row['case_index']]
        if row['anchor']!=case['anchor'] or [x['label'] for x in row['rows']]!=case['labels']:raise ValueError('native cases join')
        row['anchor_token']=case['anchor_token']
    with tempfile.TemporaryDirectory(prefix='vocab-native-reference-') as td:
        joined=Path(td)/'rows.jsonl';joined.write_text(''.join(json.dumps(r)+'\n' for r in raw));result=compare(a.source,a.model,joined,a.control_rows)
    result['native_rows']=pin(a.rows);result['original_cases']=pin(a.cases)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'out':str(a.out),'cases':len(result['cases'])}))
    if result['status']!='SEVEN_ROW_CONDITIONAL_REFERENCE_PASS':raise SystemExit(2)
if __name__=='__main__':main()
