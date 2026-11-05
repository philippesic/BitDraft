"""Attribute observed request-bounded CUDA work; no inferred DRAM or occupancy."""
from pathlib import Path
import json,sqlite3,time,collections,subprocess
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/execution-campaign-20261105-01';rows=json.load(open(R/'profile-panel/result.json'))['rows'];offset=time.time_ns()-time.monotonic_ns();out=[]
for arm in ['w1','q4']:
 r=R/f'profile-panel/00-{arm}';c=sqlite3.connect(r/'profile.sqlite');epoch=c.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0];row=next(x for x in rows if x['arm']==arm and not x['warmup']);lo,hi=[x+offset-epoch for x in row['bounds_ns']]
 kernels=c.execute('select s.value,k.start,k.end,k.gridX,k.gridY,k.blockX,k.blockY,k.graphId,k.graphNodeId,k.registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=? order by k.start',(lo,hi)).fetchall()
 graphs=collections.defaultdict(list)
 for k in kernels:graphs[k[7]].append(k)
 draft_ids=[g for g,ks in graphs.items() if g and any('w1ax_integer_dot' in k[0] for k in ks)] if arm=='w1' else []
 # Q4 draft graph: six layers have Q4_K gate/up with gridX4608? inspect all graph inventories explicitly.
 inventory=[]
 for g,ks in graphs.items():
  inventory.append(dict(graph_id=g,kernels=len(ks),gpu_us=sum(k[2]-k[1] for k in ks)/1000,head_shapes=sorted(set(tuple(k[3:7]) for k in ks if 'mul_mat_vec_q<(ggml_type)14' in k[0] and k[3]>100000)),w1_count=sum('w1ax_integer_dot' in k[0] for k in ks),kernel_names=sorted(set(k[0] for k in ks))))
 def group(ks):
  d={}
  for name,a,b,gx,gy,bx,by,gid,nid,regs in ks:
   v=d.setdefault(name,dict(count=0,gpu_us=0,shapes=[],registers=[]));v['count']+=1;v['gpu_us']+=(b-a)/1000
   if [gx,gy,bx,by] not in v['shapes']:v['shapes'].append([gx,gy,bx,by])
   if regs not in v['registers']:v['registers'].append(regs)
  return sorted(d.items(),key=lambda x:-x[1]['gpu_us'])
 selected=[k for k in kernels if k[7] in draft_ids]
 out.append(dict(arm=arm,request=row,epoch_ns=epoch,monotonic_to_unix_ns=offset,all_groups=group(kernels),graphs=inventory,draft_graph_ids=draft_ids,draft_groups=group(selected),apis=c.execute('select s.value,count(*),sum(a.end-a.start)/1000. from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId where a.start>=? and a.end<=? group by s.value',(lo,hi)).fetchall(),scope='instrumented 32-token request; graph/kernel active durations; host API spans may overlap GPU; no DRAM counters; template shapes and complete graph inventory retained'))
(R/'trace-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for v in out:
 print(v['arm'],'native',v['request']['native'],'draft ids',v['draft_graph_ids'])
 print([(name[:70],round(g['gpu_us'],2),g['count']) for name,g in v['draft_groups']])
 print('graphs',[(g['graph_id'],g['kernels'],g['w1_count'],g['head_shapes']) for g in v['graphs']])
