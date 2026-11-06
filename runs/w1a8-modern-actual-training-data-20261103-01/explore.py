from pathlib import Path
import json,hashlib,numpy as np
P=Path('/home/philip/binary-eagle-decoding');C=P/'runs/w1a8-modern-actual-parent-capture-20261103-01/capture01'
s=json.loads((C/'summary.json').read_text());overall=[]
def h(x):return hashlib.sha256(x.tobytes()).hexdigest()
for g in s['groups']:
 t=C/g['id']/'trace';a=[json.loads(x) for x in (t/'calls.jsonl').read_text().splitlines()];bf={x['call']:x for x in a if x['phase']=='before'};af={x['call']:x for x in a if x['phase']=='after'};pro=json.loads((C/g['id']/'protection.json').read_text());rounds=iter(pro['rounds']);cur=next(rounds,None); fc={};pn={};cache=[];enc_cursor={};issues=[];prefixpn=0;seeds=0;aliases=0
 def arr(spec):return np.fromfile(t/spec['file'],dtype='<f4').reshape(spec['shape'])
 for e in a:
  if e['architecture']!='eagle3':continue
  call=e['call']
  if e['phase']=='after':
   if 'prenorm' in e:
    v=arr(e['prenorm'])
    for r in e['selected']:pn[h(v[r['saved_row']])]=(call,r['batch_row'])
   continue
  if e['process_type_name']=='encode':
   raws=arr(e['batch']['embeddings']);vals=arr(af[call]['encoded_fc']);tc=e['active_target_call'] or e['last_completed_target_call'];off=enc_cursor.get(tc,0);tar=bf[tc]['batch']['rows'][off:off+len(vals)];enc_cursor[tc]=off+len(vals)
   for i,v in enumerate(vals):fc.setdefault(h(v),[]).append((call,i,tar[i]['positions'][0],h(raws[i])))
  elif 'embeddings' in e['batch']:
   rs=e['batch']['rows'];vals=arr(e['batch']['embeddings']);start=rs[0]['positions'][0]
   if start>len(cache):issues.append(('hole',call,start,len(cache)));break
   cache=cache[:start]
   for i,r in enumerate(rs):
    pos=r['positions'][0];f=[x for x in fc.get(h(vals[i]),[]) if x[2]==pos];types={x[3] for x in f}
    if len(types)>1:issues.append(('FC-ambiguous',call,pos,len(types)))
    if f:src=('fc',f[-1])
    elif h(vals[i]) in pn:src=('pn',pn[h(vals[i])])
    else:issues.append(('missing',call,pos));src=('none',None)
    if r['output'] and cur is not None and pos==cur['anchor_position'] and r['token']==cur['boundary_token']:
     seeds+=1;prefixpn+=sum(x[0]!='fc' for x in cache)
     if src[0]!='fc':issues.append(('seednonfc',call,pos,src))
     cur=next(rounds,None)
    cache.append(src)
 overall.append({'id':g['id'],'rounds':len(pro['rounds']),'seeds':seeds,'prefix_nonfc':prefixpn,'issues':issues})
 print(json.dumps(overall[-1]),flush=True)
(P/'runs/w1a8-modern-actual-training-data-20261103-01/explore.json').write_text(json.dumps(overall,indent=2)+'\n')
