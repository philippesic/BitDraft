"""Attribute diagnostic CUDA kernels by authenticated request bounds and launch phase."""
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_right
import argparse,json,sqlite3,statistics,sys
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import read,pin,atomic,require

def analyze(root,rule):
 result=read(root/'result.json');require(result['status']=='NSIGHT_NATIVE_OUTPUT_AND_TRACE_PASS','native trace not qualified')
 for k in ('sqlite','trace','release'):require(pin(result[k]['path'])==result[k],'trace artifact changed')
 hosts=[json.loads(x) for x in (root/'host-spans.jsonl').read_text().splitlines()];roots=sorted((x for x in hosts if x['depth']==0),key=lambda x:x['start_ns']);starts=[x['start_ns'] for x in roots];require(all(a['end_ns']<=b['start_ns'] for a,b in zip(roots,roots[1:])),'overlapping host roots')
 c=sqlite3.connect('file:'+result['sqlite']['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 nv=list(c.execute("select start,end from NVTX_EVENTS where text='generation_begin' order by start"));hg=[x for x in roots if x['name']=='generation_begin'];require(len(nv)==len(hg)==len(result['records'])==5,'generation marker mismatch')
 offsets=[h['start_ns']-n['start'] for h,n in zip(hg,nv)]+[h['end_ns']-n['end'] for h,n in zip(hg,nv)];offset=int(statistics.median(offsets));error=max(abs(x-offset) for x in offsets)+2000;require(error<10000,'clock alignment too uncertain')
 requests=[]
 for r,h in zip(result['records'],hg):
  require(r['start_ns']<h['start_ns']<r['end_ns'],'request/generation marker mismatch');require(pin(r['response']['path'])==r['response'],'response changed');requests.append(dict(r,generation_start_ns=h['start_ns']))
 rt={}
 for r in c.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
  require(r['correlationId'] not in rt,'ambiguous runtime correlation');rt[r['correlationId']]=dict(r)
 counters=defaultdict(lambda:dict(kernels=0,gpu_ms=0.,head_kernels=0,head_gpu_ms=0.,graph_kernels=0));top=defaultdict(lambda:dict(kernels=0,gpu_ms=0.));rows=[];boundary=0;ambiguous=0
 for k in c.execute('select k.*,s.value as kernel_name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start'):
  start,end=k['start']+offset,k['end']+offset
  matches=[r for r in requests if not r['warmup'] and r['start_ns']+error<=start and end<=r['end_ns']-error]
  if not matches:
   if any(not r['warmup'] and start<r['end_ns'] and end>r['start_ns'] for r in requests):boundary+=1
   continue
  require(len(matches)==1,'overlapping request bounds');request=matches[0];runtime=rt.get(k['correlationId']);phase='unattributed';launch=None
  if runtime:
   launch=runtime['start']+offset;i=bisect_right(starts,launch)-1
   if i>=0:
    span=roots[i]
    if span['start_ns']+error<=launch and runtime['end']+offset<=span['end_ns']-error:phase=('prefill/' if launch<request['generation_start_ns'] else 'generation/')+span['name']
    elif span['start_ns']<=launch<=span['end_ns']:ambiguous+=1
  head=rule['kernel_name_substring'] in k['kernel_name'] and k['gridX']==rule['head_grid_x'];ms=(k['end']-k['start'])/1e6;a=counters[phase];a['kernels']+=1;a['graph_kernels']+=k['graphNodeId'] is not None;a['gpu_ms']+=ms;a['head_kernels']+=head;a['head_gpu_ms']+=ms if head else 0;top[k['kernel_name']]['kernels']+=1;top[k['kernel_name']]['gpu_ms']+=ms
  if head:rows.append(dict(request=request['id'],phase=phase,start_ns=start,end_ns=end,launch_ns=launch,grid=[k['gridX'],k['gridY'],k['gridZ']],correlation=k['correlationId'],gpu_ms=ms))
 c.close();total=sum(v['gpu_ms'] for v in counters.values());head=sum(v['head_gpu_ms'] for v in counters.values());draft=sum(v['gpu_ms'] for k,v in counters.items() if k.endswith(('/draft_generate','/draft_catchup','/draft_accept')))
 return dict(status='BOUNDED_CUDA_ATTRIBUTION_AUDITED',source=pin(root/'result.json'),host_spans=pin(root/'host-spans.jsonl'),clock_offset_ns=offset,clock_margin_ns=error,request_boundary_kernels_excluded=boundary,launch_phase_boundary_ambiguous=ambiguous,measured_requests=3,phases=dict(counters),total_kernel_ms=total,head_kernel_ms=head,head_fraction_all_kernels=head/total,head_fraction_draft_kernels=head/draft if draft else None,head_events=rows,top_kernels=sorted([dict(name=k,**v) for k,v in top.items()],key=lambda x:-x['gpu_ms'])[:15],scope='Three preselected consumed DEV cases, one diagnostic trace after two warmups. GPU kernel durations can overlap and are not removable critical-path time. Host/NVTX clocks aligned using five generation markers; CUDA runtime correlation assigns enqueue phase. Boundary-ambiguous launches remain unattributed. Only the frozen W1A8 grid rule identifies a head; zero matching DFlash kernels does not mean no head cost.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rule=read(a.campaign/'head-attribution-rule.json');require(pin(rule['source']['path'])==rule['source'] and pin(rule['model']['path'])==rule['model'],'head rule source/model changed');arms={x:analyze(a.campaign/f'kernel-{x}-v4',rule) for x in ('paid5760','dflash_Q4KM')};atomic(a.out,dict(status='FIXED_REFERENCE_KERNEL_ATTRIBUTION',rule=pin(a.campaign/'head-attribution-rule.json'),arms=arms));print({k:{x:v[x] for x in ('total_kernel_ms','head_kernel_ms','head_fraction_draft_kernels','request_boundary_kernels_excluded','launch_phase_boundary_ambiguous')} for k,v in arms.items()})
