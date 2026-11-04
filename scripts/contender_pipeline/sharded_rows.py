"""Pack authenticated native F32 files into bounded shards; preserve source identities.

Optional dataset reader with the existing matrix(spec) interface. Caller retains
row-hash, finite-value and ancestry checks. Does not alter training partitions or
replace capture/union admission. Maps at most max_maps shards internally; callers
must not retain unbounded returned views. OS page cache is outside this limit.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from runner import atomic, pin, read, require


def descriptor(spec, resolved_path=None):
    shape=spec['shape']
    require(type(shape) is list and len(shape)==2 and all(type(n) is int and n>0 for n in shape),'invalid shape')
    require(type(spec['bytes']) is int and spec['bytes']==shape[0]*shape[1]*4,'invalid F32 bytes')
    require(len(spec['sha256'])==64 and all(c in '0123456789abcdef' for c in spec['sha256']),'invalid digest')
    return dict(path=resolved_path or str(Path(spec['path']).resolve()),bytes=spec['bytes'],sha256=spec['sha256'],shape=shape)


def pack(specs, output, shard_bytes=64*1024**2):
    require(type(shard_bytes) is int and shard_bytes>0,'invalid shard limit')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    entries={};shards=[];stream=None;size=0;path=None
    def finish():
        if stream is not None:
            stream.flush();os.fsync(stream.fileno());stream.close();shards.append(pin(path))
    try:
        for spec in specs:
            spec=descriptor(spec);key=spec['path']
            if key in entries:
                require(entries[key]['source']==spec,'conflicting source');continue
            require(spec['bytes']<=shard_bytes,'source exceeds shard limit; choose an explicit larger shard bound')
            if stream is None or size+spec['bytes']>shard_bytes:
                finish();path=output/f'shard-{len(shards):05d}.f32';stream=path.open('xb');size=0
            start=size;digest=hashlib.sha256()
            with Path(key).open('rb') as source:
                while chunk:=source.read(1024**2):
                    digest.update(chunk);stream.write(chunk);size+=len(chunk)
                    require(size-start<=spec['bytes'],'source grew during packing')
            require(size-start==spec['bytes'] and digest.hexdigest()==spec['sha256'],'source bytes changed')
            entries[key]=dict(source=spec,shard=len(shards),offset=start)
        finish();stream=None
        require(entries,'no sources')
        manifest=dict(schema='native_f32_shards_v1',shard_bytes=shard_bytes,shards=shards,entries=entries)
        atomic(output/'manifest.json',manifest)
        return manifest
    finally:
        if stream is not None and not stream.closed:stream.close()
        # Failed partial shards remain evidence; no manifest means never admitted.


class ShardRows:
    def __init__(self, manifest, max_maps=2):
        require(type(max_maps) is int and max_maps>0,'invalid mapping limit')
        self.manifest_pin=pin(manifest);self.manifest=read(manifest)
        require(self.manifest['schema']=='native_f32_shards_v1','invalid shard schema')
        self.aliases={};self.max_maps=max_maps;self.maps=OrderedDict();self.checked={};self.verified_sources=set();self.hits=0;self.misses=0
        for entry in self.manifest['entries'].values():
            spec=descriptor(entry['source']);shard=self.manifest['shards'][entry['shard']]
            require(type(entry['offset']) is int and entry['offset']>=0 and entry['offset']%4==0 and entry['offset']+spec['bytes']<=shard['bytes'],'invalid shard range')
    def matrix(self, spec):
        raw=str(spec['path'])
        if raw not in self.aliases:self.aliases[raw]=str(Path(raw).resolve())
        spec=descriptor(spec,self.aliases[raw]);entry=self.manifest['entries'].get(spec['path'])
        require(entry is not None and entry['source']==spec,'source absent or differs')
        index=entry['shard'];ref=self.manifest['shards'][index]
        stat=Path(ref['path']).stat();stamp=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if index not in self.checked:
            require(pin(ref['path'])==ref,'shard changed');self.checked[index]=stamp
        require(self.checked[index]==stamp,'shard identity changed after admission')
        if index not in self.maps:
            self.misses+=1
            self.maps[index]=np.memmap(ref['path'],dtype=np.uint8,mode='r')
        else:self.hits+=1
        mapping=self.maps.pop(index);self.maps[index]=mapping
        while len(self.maps)>self.max_maps:self.maps.popitem(last=False)
        start=entry['offset'];result=mapping[start:start+spec['bytes']].view('<f4').reshape(spec['shape'])
        # Whole shard authenticated before map; unchanged caller row checks remain mandatory.
        require(not result.flags.writeable,'mapping is writable')
        if spec['path'] not in self.verified_sources:
            require(hashlib.sha256(result).hexdigest()==spec['sha256'],'packed source bytes differ')
            self.verified_sources.add(spec['path'])
        return result
    def report(self):
        return dict(manifest=self.manifest_pin,max_maps=self.max_maps,mapped_shards=len(self.maps),mapped_bytes=sum(v.nbytes for v in self.maps.values()),authenticated_shards=len(self.checked),hits=self.hits,misses=self.misses,scope='Internal mappings only; retained caller views and OS file cache are not bounded by max_maps.')


class ExposureSampler:
    """Deterministic shuffled epochs and explicit resume cursor for future adapters."""
    def __init__(self, ids, seed, state=None):
        require(ids and len(ids)==len(set(ids)),'duplicate/empty sample IDs')
        require(type(seed) is int and seed>=0,'invalid seed')
        self.ids=list(ids);self.seed=seed;self.cursor=0;self.epoch=None;self.order=None
        self.digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if state is not None:
            require(state['ids_sha256']==self.digest and state['seed']==seed,'sampler contract changed')
            require(type(state['cursor']) is int and state['cursor']>=0,'invalid cursor');self.cursor=state['cursor']
    def next(self):
        epoch,offset=divmod(self.cursor,len(self.ids))
        if epoch!=self.epoch:
            self.order=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,epoch]))).permutation(len(self.ids));self.epoch=epoch
        value=self.ids[int(self.order[offset])];self.cursor+=1;return value
    def state(self):return dict(ids_sha256=self.digest,seed=self.seed,cursor=self.cursor)
