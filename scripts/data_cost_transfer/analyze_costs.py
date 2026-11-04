"""Disjoint host-cost accounting and explicitly conditional architecture sensitivity."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,sys,json,math
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import read,pin,atomic,require

def analyze(root):
 receipt=read(root/'result.json');require(receipt['status']=='PAIRED_NATIVE_COST_AND_OVERHEAD_PASS','paired proof missing')
 require(pin(receipt['plain_audit']['path'])==receipt['plain_audit'] and pin(receipt['traced_result']['path'])==receipt['traced_result'],'panel receipt changed')
 plain=read(root/'plain/audit.json');traced=read(root/'traced/result.json');require(traced['exact_decode_counters_match_uninstrumented'],'counter equivalence missing')
 measured={}
 for r in traced['records']:
  if not r['warmup']:measured[(r['arm'],r['repetition'],r['id'])]=r
 accum=defaultdict(lambda:dict(prefill=Counter(),generation=Counter(),http_ms=0.,unattributed_http_ms=0.,server_generation_ms=0.,tokens=0,rounds=0,cases=0))
 seen=set();raw_inputs=[]
 for path in sorted((root/'traced').glob('rep-*/cost-trace.summary.json')):
  summary=read(path);require(pin(summary['trace']['path'])==summary['trace'],'trace changed');raw_inputs += [pin(path),summary['trace']]
  spans=[json.loads(line) for line in Path(summary['trace']['path']).read_text().splitlines()]
  for r in summary['rows']:
   if r['warmup']:continue
   key=(r['arm'],r['repetition'],r['id']);require(key in measured and key not in seen,'incomplete/duplicate traced request');seen.add(key);raw=measured[key];a=accum[r['arm']]
   bounds_path=Path(raw['response']['path']).with_suffix('.bounds.json');bounds=read(bounds_path);raw_inputs.append(pin(bounds_path))
   roots=sorted([s for s in spans if s['depth']==0 and bounds['start_ns']<=s['start_ns'] and s['end_ns']<=bounds['end_ns']],key=lambda s:s['start_ns'])
   require(roots and all(x['end_ns']<=y['start_ns'] for x,y in zip(roots,roots[1:])),'overlapping root spans')
   actual=Counter()
   for span in roots:actual[span['name']]+=(span['end_ns']-span['start_ns'])/1e6
   claimed=Counter(r['root_ms']['prefill']);claimed.update(r['root_ms']['generation']);require(set(actual)==set(claimed) and all(math.isclose(actual[k],claimed[k],rel_tol=1e-12,abs_tol=1e-9) for k in actual),'raw root costs differ')
   for phase in ('prefill','generation'):a[phase].update(r['root_ms'][phase])
   a['http_ms']+=r['http_ms'];a['unattributed_http_ms']+=r['unattributed_http_ms'];a['server_generation_ms']+=raw['timings']['predicted_ms'];a['tokens']+=r['completion_tokens'];a['rounds']+=r['verification_rounds'];a['cases']+=1
 require(seen==set(measured),'missing traced requests')
 output={}
 for arm,a in accum.items():
  draft=sum(a['generation'][k] for k in ('draft_generate','draft_catchup','draft_accept'))
  target=sum(a['generation'][k] for k in ('target_process','target_sync','target_sample','verifier_sample'))
  pairs=[r for r in receipt['pairs'] if r['arm']==arm and not r['warmup']]
  ids=sorted({r['id'] for r in pairs});pw=np.full((len(ids),5),np.nan);tw=pw.copy()
  for r in pairs:pw[ids.index(r['id']),r['repetition']]=r['plain_ms'];tw[ids.index(r['id']),r['repetition']]=r['traced_ms']
  require(np.isfinite(pw).all() and np.isfinite(tw).all(),'incomplete overhead pairs')
  rng=np.random.default_rng(611042);weights=rng.multinomial(len(ids),[1/len(ids)]*len(ids),size=10000);reps=rng.multinomial(5,[.2]*5,size=len(weights));ratios=np.einsum('bi,ij,bj->b',weights,tw,reps)/np.einsum('bi,ij,bj->b',weights,pw,reps)
  output[arm]=dict(root_prefill_ms=dict(a['prefill']),root_generation_ms=dict(a['generation']),full_request_ms=a['http_ms'],unattributed_full_request_ms=a['unattributed_http_ms'],server_generation_ms=a['server_generation_ms'],tokens=a['tokens'],rounds=a['rounds'],generation_draft_ms=draft,generation_target_and_sampling_ms=target,root_ms_per_round={k:v/a['rounds'] for k,v in a['generation'].items()} if a['rounds'] else None,full_draft_ms_per_round=draft/a['rounds'] if a['rounds'] else None,tokens_per_round=a['tokens']/a['rounds'] if a['rounds'] else None,overhead_percent=float((tw.sum()/pw.sum()-1)*100),overhead_bootstrap95_percent=((np.quantile(ratios,[.025,.975])-1)*100).tolist(),scope='Root host spans are disjoint. Nested spans omitted from totals. Async GPU work may be charged at a later synchronization; these are not isolated GPU critical paths.')
 w=output['paid5760'];d=output['dflash_Q4KM'];require(w['tokens']==d['tokens'],'matched outputs required')
 zero_gen=w['server_generation_ms']-w['generation_draft_ms'];zero_request=w['full_request_ms']-w['generation_draft_ms']-sum(w['root_prefill_ms'].get(k,0) for k in ('draft_generate','draft_catchup','draft_accept'))
 require(zero_gen>0 and zero_request>0,'invalid fixed-cost subtraction')
 sensitivity=dict(current_emitted_tokens_per_round=w['tokens_per_round'],dflash_emitted_tokens_per_round=d['tokens_per_round'],needed_emitted_tokens_per_round_at_current_cost=(w['server_generation_ms']/w['rounds'])/(d['server_generation_ms']/d['tokens']),zero_draft_generation_speed_ratio_to_dflash=d['server_generation_ms']/zero_gen,zero_draft_full_request_speed_ratio_to_dflash=d['full_request_ms']/zero_request,assumption='Counterfactual sensitivity within the SAME traced panel: remove all measured draft spans while holding verification rounds, target costs, sampling, and outputs fixed. Real changes can alter these costs. Not a hard speed ceiling or pruning rule.')
 return dict(status='COMPLETE_NATIVE_HOST_COSTS_AND_OVERHEAD_AUDITED',source_panel=pin(root/'result.json'),raw_inputs=raw_inputs,uninstrumented=plain['summary'],costs=output,sensitivity=sensitivity,scope='Fixed current5760 row-W1A8 reference and frozen DFlash; consumed DEV12. Five interleaved repetitions; all outputs and draft/round counters identical with/without instrumentation.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();v=analyze(a.panel);atomic(a.out,v);print(v['sensitivity'])
