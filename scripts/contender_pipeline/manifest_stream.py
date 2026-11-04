"""Bounded metadata streaming over already-admitted dataset manifests.

A registry selects existing window IDs; it cannot relabel held-out windows as FIT.
The factory must retain its native row/ancestry checks. Admission of a new corpus
and source-alias exclusions precede registry creation, not inferred by this reader.
"""
from collections import OrderedDict
from pathlib import Path
from runner import pin,read,require


class ManifestStream:
    def __init__(self, registry, factory, max_manifests=1, max_manifest_bytes=64*1024**2, max_index_bytes=32*1024**2, row_reader=None):
        require(type(max_manifests) is int and max_manifests>0,'invalid manifest cache bound')
        require(Path(registry).stat().st_size<=max_index_bytes,'registry exceeds metadata bound')
        self.registry_pin=pin(registry);self.registry=read(registry);self.factory=factory
        require(self.registry['schema']=='training_manifest_stream_v1','invalid registry')
        self.max_manifests=max_manifests;self.max_manifest_bytes=max_manifest_bytes
        self.row_reader=row_reader;self.cache=OrderedDict();self.lookup={};self.loads=0;self.peak_cached_bytes=0
        for i,entry in enumerate(self.registry['shards']):
            require(entry['manifest']['bytes']<=max_manifest_bytes,'manifest exceeds bound')
            require(entry['admission'] and entry['contract']==self.registry['contract'],'shard contract/admission differs')
            for ref in [entry['manifest'],entry['admission']]:require(pin(ref['path'])==ref,'registry dependency changed')
            require(entry['windows'] and len(set(entry['windows']))==len(entry['windows']),'duplicate/empty shard windows')
            for local_id in entry['windows']:
                key=f'{i}:{local_id}';self.lookup[key]=(i,local_id)
        require(self.lookup,'empty training stream')
    @property
    def ids(self):return list(self.lookup)
    def _load(self,index):
        entry=self.registry['shards'][index]
        if index not in self.cache:
            # Revalidate on reopening, including after LRU eviction.
            require(pin(entry['manifest']['path'])==entry['manifest'],'manifest changed')
            while len(self.cache)>=self.max_manifests:self.cache.popitem(last=False)
            data=self.factory(entry['manifest']['path'])
            if self.row_reader is not None:
                require(hasattr(data,'_rows'),'dataset has no compatible row-reader hook')
                data._rows=self.row_reader
            windows={w['id']:w for w in data.windows}
            require(all(key in windows and windows[key]['role']=='fit' for key in entry['windows']),'held-out or missing window in TRAIN registry')
            require(data.manifest['target']==self.registry['contract']['target'],'dataset target differs')
            self.cache[index]=(data,windows);self.loads+=1
            self.peak_cached_bytes=max(self.peak_cached_bytes,sum(self.registry['shards'][i]['manifest']['bytes'] for i in self.cache))
        value=self.cache.pop(index);self.cache[index]=value;return value
    def window(self,key):
        require(key in self.lookup,'unknown sample ID')
        index,local=self.lookup[key];data,windows=self._load(index)
        return data,windows[local]
    def materialize(self,key,device='cpu'):
        data,window=self.window(key);return data.materialize(window,device)
    def embedding_lookup(self,key,token,device='cpu'):
        data,_=self.window(key);return data.embedding_lookup(token,device)
    def report(self):
        return dict(registry=self.registry_pin,windows=len(self.lookup),shards=len(self.registry['shards']),loads=self.loads,cached_manifests=len(self.cache),peak_cached_manifest_payload_bytes=self.peak_cached_bytes,max_manifest_bytes=self.max_manifest_bytes,max_manifests=self.max_manifests,scope='Manifest payload bounds, not total Python RSS; callers must not retain arbitrary datasets/views.')
