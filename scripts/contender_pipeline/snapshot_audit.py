"""Bounded immutable file snapshots for repeated rows within one union audit.

The original Audit.row still checks geometry, byte accounting, row hash and
finiteness. Original Audit.file authenticates each snapshot before copying; the
copy is rehashed. Other Audit.file calls retain original filesystem checks.
"""
from collections import OrderedDict
import hashlib
import io
from pathlib import Path


def snapshot_audit(base, max_bytes=64*1024**2):
    class SnapshotAudit(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.snapshots=OrderedDict();self.snapshot_bytes=0;self.snapshot_hits=0;self.snapshot_misses=0
            self._active_snapshot=None
        def file(self,ref):
            active=self._active_snapshot
            if active is not None and ref==active[0]:
                class ImmutableFile:
                    def open(_,mode):
                        if mode!='rb':raise ValueError('read-only snapshot')
                        return io.BytesIO(active[1])
                return ImmutableFile()
            return super().file(ref)
        def row(self,spec,row,width,expected):
            self.guard()
            key=(str(spec['path']),spec['sha256'],spec['bytes'])
            value=self.snapshots.pop(key,None)
            if value is None and spec['bytes']<=max_bytes:
                path=super().file(spec)
                value=Path(path).read_bytes()
                if len(value)!=spec['bytes'] or hashlib.sha256(value).hexdigest()!=spec['sha256']:
                    raise ValueError('source changed while snapshotting')
                while self.snapshots and self.snapshot_bytes+len(value)>max_bytes:
                    _,old=self.snapshots.popitem(last=False);self.snapshot_bytes-=len(old)
                self.snapshot_bytes+=len(value);self.snapshot_misses+=1
            elif value is not None:self.snapshot_hits+=1
            if value is not None:self.snapshots[key]=value
            self._active_snapshot=(spec,value) if value is not None else None
            try:return super().row(spec,row,width,expected)
            finally:self._active_snapshot=None
    return SnapshotAudit
