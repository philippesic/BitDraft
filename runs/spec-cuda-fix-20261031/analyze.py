from pathlib import Path
import json,re
P=Path(__file__).resolve().parent
results=[]
for target in sorted((P/'measurements').iterdir()):
 files=list(target.glob('*/patched/requests.jsonl'))
 assert len(files)==2,(target,files)
 arms={f.parent.parent.name:[json.loads(l) for l in f.read_text().splitlines()] for f in files}
 base=arms.pop('no-spec');name,rows=next(iter(arms.items()))
 assert len(base)==len(rows)==32
 b={(r['prompt'],r['rep']):r for r in base};c={(r['prompt'],r['rep']):r for r in rows}
 assert len(b)==len(c)==32 and b.keys()==c.keys()
 assert all(r['result']['tokens_cached']==0 for r in base+rows)
 bad=[k for k in b if b[k]['result']['tokens']!=c[k]['result']['tokens']]
 repeats=lambda arm:sum(arm[i,0]['result']['tokens']==arm[i,1]['result']['tokens'] for i in range(16))
 def stats(rs):
  n=sum(x['result']['tokens_predicted'] for x in rs);t=sum(x['elapsed'] for x in rs)
  return {'tokens':n,'elapsed':t,'request_tps':n/t}
 bs,cs=stats(base),stats(rows)
 results.append({'target':target.name,'draft':name,'matches':32-len(bad),'comparisons':32,'mismatches':bad,'baseline_repeat_matches':repeats(b),'candidate_repeat_matches':repeats(c),'baseline':bs,'speculative':cs,'speed_ratio':cs['request_tps']/bs['request_tps'],'math_baseline':b[2,0]['result']['content'],'math_speculative':c[2,0]['result']['content']})
regs=[]
for f in sorted(P.glob('regression-*/*.log')):
 matches=re.findall(r'spec parity: checks=(\d+) greedy_flips=(\d+) max_abs=([^\s]+)',f.read_text())
 assert len(matches)==2,(f,matches)
 regs.append({'model_log':str(f.relative_to(P)),'checks':sum(int(m[0]) for m in matches),'flips':sum(int(m[1]) for m in matches),'max_abs':max(float(m[2]) for m in matches)})
assert len(regs)==4 and sum(r['checks'] for r in regs)==2040
out={'full_requests':results,'native_regressions':regs,'total_matches':sum(r['matches'] for r in results),'total_comparisons':sum(r['comparisons'] for r in results)}
(P/'final-results.json').write_text(json.dumps(out,indent=2))
print(json.dumps({**out,'full_requests':[{k:v for k,v in r.items() if not k.startswith('math_')} for r in results]},indent=2))
assert out['total_matches']==128,out
assert all(r['max_abs']==0 and r['flips']==0 for r in regs)
