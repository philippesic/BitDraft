"""Join admitted chunk indices without copying arrays or constructing a large union."""
from pathlib import Path
import argparse,sys,time,resource
from collections import Counter,defaultdict
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/contender_pipeline'),str(Path(__file__).parent)]
from runner import pin,read,atomic,require
from training_bank import TrainingBank

def register(roots,out):
 require(not out.exists(),'registry output exists');out.mkdir(parents=True)
 registry=dict(schema='training_manifest_stream_v1',contract=None,shards=[])
 merged=dict(schema='native_f32_shards_v1',shard_bytes=64*1024**2,shards=[],entries={})
 groups=set();windows=set();source_groups=[]
 head_map=P/'runs/w1a8-modern-native-replay-20261103-01/draft-map.ids';supported=set(map(int,head_map.read_text().split()));stats=defaultdict(Counter)
 for root in roots:
  admission=read(root/'admission.json');require(admission['status']=='TRAIN_CHUNK_NATIVE_MATERIALIZATION_PASS','chunk not admitted')
  for key in ('manifest','shards'):require(pin(admission[key]['path'])==admission[key],'chunk dependency changed')
  m=read(admission['manifest']['path']);s=read(admission['shards']['path']);contract=dict(target=m['target'],parent=m['parent'])
  if registry['contract'] is None:registry['contract']=contract
  require(contract==registry['contract'] and not m['missing_embedding_ids'],'chunk contract differs')
  for g in m['groups']:
   require(g['role']=='fit' and g['group_id'] not in groups,'duplicate or heldout group');groups.add(g['group_id']);source_groups.append(g['group_id'])
  for w in m['windows']:
   d=stats[w['domain']];d['windows']+=1;d['labels']+=len(w['labels']);d['supported_labels']+=sum(t in supported for t in w['labels']);d['reachable_labels']+=sum(w['native_supervision_reachable_mask']);d['trainable_labels']+=sum(t in supported and ok for t,ok in zip(w['labels'],w['native_supervision_reachable_mask']));d['native_prefix']+=w['native_prefix']
  ids=[w['id'] for w in m['windows']];require(len(set(ids))==len(ids) and not set(ids)&windows,'duplicate windows');windows.update(ids)
  registry['shards'].append(dict(manifest=admission['manifest'],admission=pin(root/'admission.json'),contract=contract,windows=ids))
  offset=len(merged['shards']);merged['shards']+=s['shards']
  for key,entry in s['entries'].items():
   if key in merged['entries']:require(merged['entries'][key]['source']==entry['source'],'conflicting packed source')
   else:merged['entries'][key]={**entry,'shard':entry['shard']+offset}
 atomic(out/'registry.json',registry);atomic(out/'shards.json',merged)
 begin=time.monotonic();bank=TrainingBank(out/'registry.json',out/'shards.json')
 checked=[]
 # Force each chunk through the two-entry LRU and revisit after eviction.
 for entry in registry['shards']*2:
  for wid in (entry['windows'][0],entry['windows'][-1]):
   _,w=bank.stream.window(bank.keys[wid]);bank.materialize(w,'cpu');checked.append(wid)
 exposure=dict(new_windows=1920,unique_groups=len({bank.meta[k]['group_id'] for k in bank.order}),domains=dict(Counter(bank.meta[k]['domain'] for k in bank.order)),trainable_labels=sum(sum(t in supported and ok for t,ok in zip(bank.meta[k]['labels'],bank.meta[k]['native_supervision_reachable_mask'])) for k in bank.order))
 atomic(out/'qualification.json',dict(head_map=pin(head_map),domain_window_coverage={k:dict(v) for k,v in stats.items()},planned_new_exposure=exposure,status='BROADER_TRAIN_STREAM_QUALIFIED',source=pin(__file__),registry=pin(out/'registry.json'),shards=pin(out/'shards.json'),groups=len(groups),windows=len(windows),source_groups=source_groups,materialized_views=len(checked),sample_order_sha256=bank.contract['new_order_sha256'],report=bank.report(),wall_seconds=time.monotonic()-begin,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Every window validated at chunk admission; all chunks revisited after bounded metadata eviction. TRAIN only.'))
 print('BROADER_TRAIN_STREAM_QUALIFIED',len(groups),len(windows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--chunks',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();register(a.chunks,a.out)
