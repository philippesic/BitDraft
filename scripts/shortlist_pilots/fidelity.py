"""Run the fixed four-cell serving-feature substitution on authentic native models."""
import argparse, hashlib, json, os, pathlib, subprocess, time

def sha(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    c=json.loads(pathlib.Path(a.config).read_text());out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    r=pathlib.Path('/home/philip/binary-eagle-decoding/runs/w1a8-algorithms-20261020')
    fit=json.loads((r/'loop2-mode-fitprobe01/fit12.json').read_text());case=fit['cases'][0];assert case['chain_id']=='answer:0' and case['anchor']==161 and case['anchor_token']==279
    for x in [c['serving_features'],c['serving_diagnostic'],case['features']]:assert sha(x['path'])==x['sha256']
    rows=json.loads(pathlib.Path(c['serving_diagnostic']['path']).read_text())['rows'];assert len(rows)==161 and [x['position'] for x in rows]==list(range(161))
    x=np.memmap(c['serving_features']['path'],dtype='<f4',mode='r').reshape(-1,12800)
    indices=[v['feature_row'] for v in rows];assert indices[154:]==[154,155,160,161,166,172,173]
    gathered=np.asarray(x[indices],dtype='<f4');assert np.isfinite(gathered).all();f=out/'serving161.f32';gathered.tofile(f)
    plans=[]
    for i,kind in enumerate(['historical','serving']):
        v=dict(case);v['case_index']=i
        if kind=='serving':v['features']={'path':str(f),'sha256':sha(f)}
        plans.append(v)
    cases=out/'cases.json';cases.write_text(json.dumps({'schema':'group_transfer12_v1','cases':plans},indent=2))
    results={};started=time.monotonic()
    for arm,model in c['models'].items():
        dest=out/arm;cmd=[c['helper']['path'],model['path'],str(cases),str(dest),'on'];t=time.monotonic()
        with (out/(arm+'.log')).open('wb') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=True)
        rr=[json.loads(l) for l in (dest/'rows.jsonl').read_text().splitlines()];assert len(rr)==2
        summaries=[]
        for row in rr:
            raw=np.fromfile(row['raw_path'],dtype='<f4').reshape(7,151936);assert np.isfinite(raw).all()
            first=next((i for i,v in enumerate(row['rows'][:5]) if v['argmax']!=case['labels'][i]),5)
            summaries.append({'first5_depth':first,'first7_depth':row['first_rejection'],'argmax':[v['argmax'] for v in row['rows']],'common_prefix_margins':[v['margin'] for v in row['rows'][:min(first+1,5)]],'row_record':row})
        results[arm]={'cells':dict(zip(['historical','serving'],summaries)),'wall_seconds':time.monotonic()-t}
    cells={k:v['cells'] for k,v in results.items()}
    interaction=(cells['fullbasis']['serving']['first5_depth']-cells['parent']['serving']['first5_depth'])-(cells['fullbasis']['historical']['first5_depth']-cells['parent']['historical']['first5_depth'])
    result={'status':'FOUR_CELL_NATIVE_COMPLETE','results':results,'first5_interaction':interaction,'total_wall_seconds':time.monotonic()-started,'scope':'one TRAIN anchor; no training, new target capture, request throughput or production-KV equivalence claim'}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'}))

if __name__=='__main__':main()
