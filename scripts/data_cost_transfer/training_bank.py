"""Bounded new TRAIN metadata/array stream and fixed 1:1 old/new CE exposure."""
from pathlib import Path
import hashlib,sys
import numpy as np
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/contender_pipeline'),str(P/'scripts/w1a8_generalization')]
from runner import pin,read,require
from manifest_stream import ManifestStream
from sharded_rows import ShardRows,ExposureSampler
from modern_actual_data import ActualDataset

class TrainingBank:
 def __init__(self,registry,shards):
  self.reader=ShardRows(shards,max_maps=2)
  self.stream=ManifestStream(registry,ActualDataset,max_manifests=2,row_reader=self.reader)
  self.keys={};self.meta={};self.embeddings={}
  for i,entry in enumerate(self.stream.registry['shards']):
   data=ActualDataset(entry['manifest']['path']);selected=set(entry['windows'])
   admission=read(entry['admission']['path']);require(admission['status']=='TRAIN_CHUNK_NATIVE_MATERIALIZATION_PASS' and admission['manifest']==entry['manifest'],'chunk not admitted')
   for w in data.windows:
    if w['id'] not in selected:continue
    require(w['role']=='fit' and w['id'] not in self.keys,'duplicate or heldout window')
    self.keys[w['id']]=f"{i}:{w['id']}"
    self.meta[w['id']]={k:w[k] for k in ('id','labels','native_supervision_reachable_mask','domain','group_id','anchor')}
   for token in data.manifest['required_embedding_ids']:
    row=data.manifest['embedding_rows'][str(token)];spec=data.manifest['embedding_files'][row['file']]
    if token in self.embeddings:require(self.embeddings[token][1]['sha256']==row['sha256'],'embedding disagreement')
    else:self.embeddings[token]=(spec,row)
  sampler=ExposureSampler(sorted(self.keys),seed=61104)
  self.order=[sampler.next() for _ in range(1920)]
  self.contract=dict(registry=pin(registry),shards=pin(shards),source=pin(__file__),new_windows=len(self.keys),new_order_sha256=hashlib.sha256('\n'.join(self.order).encode()).hexdigest(),ce_mix='one original CE window (first of original pair) plus one new shuffled-window CE; fixed4 denominator; unchanged replay',sampler_seed=61104,maximum_updates=1920)
 def add_embeddings(self,embeddings,device):
  import torch
  for token,(spec,row) in self.embeddings.items():
   value=np.ascontiguousarray(self.reader.matrix(spec)[row['row']],dtype='<f4')
   require(hashlib.sha256(value.tobytes()).hexdigest()==row['sha256'] and np.isfinite(value).all(),'native embedding corrupted')
   if token in embeddings:require(torch.equal(embeddings[token].cpu(),torch.from_numpy(value.copy())),'old/new embedding differs')
   else:embeddings[token]=torch.from_numpy(value.copy()).to(device)
 def windows(self,old,order,cursor,original,metadata=False):
  require(5760<=cursor<7680,'bounded cursor required')
  key=self.order[cursor-5760]
  new=self.meta[key] if metadata else self.stream.window(self.keys[key])[1]
  return [original(old,order,cursor)[0],new]
 def materialize(self,window,device):return self.stream.materialize(self.keys[window['id']],device)
 def report(self):return dict(stream=self.stream.report(),arrays=self.reader.report(),contract=self.contract)
