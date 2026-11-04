"""Audit every native-panel payload and release, then summarize paired DEV contrasts."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve(strict=True)
    with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=d)
def checked(ref):
    assert pin(ref['path'])==ref,ref['path']
    return read(ref['path'])
def signature(r):return {k:r[k] for k in ('tokens','content','stop_type','stopping_word','truncated')}


def audit(root, control, paid):
    v=read(root/'result.json');assert v['status']=='DEVELOPMENT_NATIVE_REQUESTS_PASS'
    models=read(root/'models.json');inputs=read(root/'inputs.json');arms={a['id']:a for a in models['arms']}
    items={i['id']:i for i in inputs['rows']+inputs['warmups']}
    gold={k:signature(checked(i['reference'])) for k,i in items.items()}
    payload={k:checked(i['request']) for k,i in items.items()}
    seen=set();measured=[];counts=collections.Counter();spec=0
    for r in v['records']:
        key=(r['arm'],r['repetition'],r['id']);assert key not in seen;seen.add(key)
        req=checked(r['request']);res=checked(r['response']);assert req==payload[r['id']]
        assert req['cache_prompt'] is False and req['temperature']==0 and req['seed']==1234
        assert signature(res)==gold[r['id']]
        assert len(res['tokens'])==r['completion_tokens']==res['tokens_predicted']
        assert res['timings']['cache_n']==0
        assert math.isfinite(r['full_http_wall_seconds']) and r['full_http_wall_seconds']>0
        metrics=checked(r['native_metrics']);delta={k:metrics['after'].get(k,0)-metrics['before'].get(k,0) for k in set(metrics['after'])|set(metrics['before'])}
        assert delta==r['native_metrics_delta'] and all(n>=0 for n in delta.values())
        assert delta['spec_decode_num_draft_tokens_total']==r['draft_proposed']
        assert delta['spec_decode_num_accepted_tokens_total']==r['draft_accepted']
        assert delta['spec_decode_num_drafts_total']==r['verification_rounds']
        pos=sum(n for k,n in delta.items() if 'accepted_tokens_per_pos_total' in k)
        assert pos==r['draft_accepted'],(r['arm'],pos,r['draft_accepted'])
        if arms[r['arm']]['kind']!='target':assert r['verification_rounds']>0;spec+=1
        else:assert r['verification_rounds']==0
        counts[r['arm']]+=1
        if not r['warmup']:measured.append(r)
    assert len(measured)==v['measured_requests']==len(arms)*5*len(inputs['rows'])
    assert all(n==5*len(items) for n in counts.values())
    images={};groups=set();cells=list(root.glob('rep-*'));assert len(cells)==5*len(arms)
    for c in cells:
        release=read(c/'release.json');assert release['owned_process_reaped'] and release['owned_process_group_absent'] and release['returncode']==0
        groups.add(release['identity']['pgid']);records=[r for r in v['records'] if Path(r['response']['path']).parent==c];assert len(records)==len(items)
        arm=arms[records[0]['arm']];cmd=read(c/'command.json');argv=cmd['argv'];env=cmd['environment']
        assert argv[0]==arm['runtime']['server']['path'] and argv[argv.index('-m')+1]==models['target']['path']
        for option,value in [('-c','4096'),('-b','512'),('-ub','512'),('-t','4'),('-fa','on'),('--cache-type-k','f16'),('--cache-type-v','f16'),('-lv','3')]:assert argv[argv.index(option)+1]==value
        assert env['GGML_CUDA_DECODE_CONSISTENT']=='32' and env['CUDA_VISIBLE_DEVICES']=='0'
        if arm['kind']!='target':assert argv[argv.index('-md')+1]==arm['model']['path'] and argv[argv.index('--spec-draft-n-max')+1]=='2'
        if arm['kind']=='w1a8':assert env['GGML_W1AX_ACT_BITS']=='8'
        else:assert 'GGML_W1AX_ACT_BITS' not in env
        for tag in ['warm','final']:
            for ref in read(c/f'loaded-images-{tag}.json')['images']:
                assert ref['path'] not in images or images[ref['path']]==ref;images[ref['path']]=ref
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:group=int((p/'stat').read_text().rsplit(')',1)[1].split()[2])
            except (OSError,ValueError,IndexError):continue
            assert group not in groups,group
    for ref in images.values():assert pin(ref['path'])==ref
    for a in arms.values():
        if a['model']:assert pin(a['model']['path'])==a['model']
    assert pin(models['target']['path'])==models['target']
    for entry in read(root/'freeze.json')['source_snapshots']:
        if 'snapshot' in entry:assert pin(entry['snapshot']['path'])==entry['snapshot']
    prompts=sorted(i['id'] for i in inputs['rows']);tables={};summary=[]
    for name in arms:
        rows=[r for r in measured if r['arm']==name];wall=np.full((len(prompts),5),np.nan);tokens=np.full_like(wall,np.nan)
        for r in rows:wall[prompts.index(r['id']),r['repetition']]=r['full_http_wall_seconds'];tokens[prompts.index(r['id']),r['repetition']]=r['completion_tokens']
        assert np.isfinite(wall).all();tables[name]=(wall,tokens)
        rounds=sum(r['verification_rounds'] for r in rows);accepted=sum(r['draft_accepted'] for r in rows);proposed=sum(r['draft_proposed'] for r in rows)
        generation=sum(r['timings']['predicted_ms']/1000 for r in rows)
        summary.append(dict(arm=name,tokens=int(tokens.sum()),wall_seconds=float(wall.sum()),TPS=float(tokens.sum()/wall.sum()),server_generation_seconds=generation,verification_rounds=rounds,draft_accepted=accepted,draft_proposed=proposed,acceptance=accepted/proposed if proposed else None,emitted_tokens_per_spec_round=float(tokens.sum()/rounds) if rounds else None,accepted_tokens_per_spec_round=accepted/rounds if rounds else None,server_generation_ms_per_spec_round=1000*generation/rounds if rounds else None,per_repetition_TPS=(tokens.sum(0)/wall.sum(0)).tolist()))
    rng=np.random.default_rng(421104);pw=rng.multinomial(len(prompts),[1/len(prompts)]*len(prompts),size=10000);rw=rng.multinomial(5,[.2]*5,size=10000);contrasts=[]
    for a in arms:
        if arms[a]['kind']!='w1a8':continue
        for b in dict.fromkeys([control,paid]):
            if a==b:continue
            ta,na=tables[a];tb,nb=tables[b];assert np.array_equal(na,nb)
            samples=np.einsum('bi,ij,bj->b',pw,tb,rw)/np.einsum('bi,ij,bj->b',pw,ta,rw);ci=np.quantile(samples,[.025,.975]).tolist()
            contrasts.append(dict(candidate=a,control=b,ratio=float(tb.sum()/ta.sum()),bootstrap95=ci,ranking='faster' if ci[0]>1 else 'slower' if ci[1]<1 else 'unresolved'))
    return dict(status='RAW_REQUEST_MODEL_RUNTIME_METRICS_AND_RELEASE_AUDIT_PASS',panel=pin(root/'result.json'),summary=summary,contrasts=contrasts,measured_requests=len(measured),all_responses=len(seen),speculative_exact=spec,released_groups=len(groups),rehashed_loaded_images=len(images),scope='Consumed DEV exploratory comparisons; unadjusted paired prompt/repetition intervals. No new generalization, causal component decomposition, universal early predictor or architectural choice.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--control',required=True);p.add_argument('--paid',default='paid5760');p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.panel,a.control,a.paid)
    with a.out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':r['status'],'summary':r['summary'],'contrasts':r['contrasts']}))
if __name__=='__main__':main()
