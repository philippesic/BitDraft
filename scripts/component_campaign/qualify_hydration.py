"""Real paid-archive byte recovery plus preservation of an existing partial."""
import hashlib,json,shutil
import zstandard as z
from .common import *
from .archive import paths,hydrate,release
from scripts.dflash_baseline import native_archive as old

def main():
 out=R/'qualification/hydration-safety';out.mkdir(exist_ok=True)
 if (out/'verified.json').exists():return
 free_before={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')}
 # Decode the real protected B0 archive into a NEW owned destination. Neither
 # the canonical model pathname nor the original archive/receipt/basis changes.
 _,source=old.paths(B/'model-final.gguf');record=json.load(open(source));old.authenticate(record)
 destination=out/'B0-legacy-copy.gguf';clone=dict(record,original=dict(record['original'],path=str(destination)))
 _,receipt=old.paths(destination)
 if receipt.exists() and json.load(open(receipt))!=clone:raise ValueError('existing validation receipt differs')
 save(receipt,clone);created=hydrate(destination);actual=pin(destination)
 if any(actual[k]!=record['original'][k] for k in ('bytes','sha256')):raise ValueError('actual B0 legacy recovery differs')
 release(destination,created);old.authenticate(record)
 p=out/'partial-sentinel.gguf';archive,receipt=paths(p);payload=b'new owned hydration fixture\n';archive.write_bytes(z.ZstdCompressor().compress(payload))
 save(receipt,dict(schema='dflash.authenticated_tensor_slices.v1',archive=pin(archive),dependencies=[],original=dict(path=str(p),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()),segments=[dict(kind='payload',offset=0,coded_bytes=archive.stat().st_size,bytes=len(payload))]))
 partial=p.with_suffix('.gguf.hydrating');sentinel=b'preserve an already present partial\n';partial.write_bytes(sentinel);sentinel_pin=pin(partial)
 try:hydrate(p)
 except FileExistsError:pass
 else:raise ValueError('existing partial was unexpectedly overwritten')
 if pin(partial)!=sentinel_pin:raise ValueError('existing partial was removed/changed')
 partial.unlink();created=hydrate(p)
 if p.read_bytes()!=payload:raise ValueError('exclusive verified publication differs')
 release(p,created)
 for item in (archive,receipt):item.unlink()
 save(out/'verified.json',dict(status='REAL_B0_BYTES_AND_EXISTING_PARTIAL_PRESERVATION_PASS',B0_original=record['original'],decoded_owned_copy=actual,protected_source_receipt=pin(source),existing_partial=sentinel_pin,source=pin(__file__),helper=pin(ROOT/'scripts/component_campaign/archive.py'),owner=identity(),free_before=free_before,free_after={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')},scope='Authentic legacy B0 stream decoded and hashed exactly into NEW scratch; original paid archive/dictionary/receipt untouched. Existing partial survived exclusive-open failure. Only authenticated owned fixture scratch removed. Concurrent training can change free-space readings.'))
 print('NATIVE_HYDRATION_OWNERSHIP_AND_REAL_BYTES_PASS',flush=True)

if __name__=='__main__':main()
