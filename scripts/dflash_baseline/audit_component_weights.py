"""Independently audit deployed component codes against authenticated frozen sources."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.model import GGUFReader,dequantize
from scripts.dflash_baseline.components import tensor_component,encoded,FORMATS
from scripts.dflash_baseline.pipeline import sha
from scripts.dflash_baseline.native_archive import hydrate,release

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def audit(path):
 created=hydrate(path)
 try:return audit_raw(path)
 finally:release(path,created)
def audit_raw(path):
 path=Path(path);meta=json.load(open(path.with_suffix('.json')));config=meta['config'];source=config['source'];target=config['evaluation']['target']
 if sha(source['path'])!=source['sha256'] or sha(target['path'])!=target['sha256']:raise ValueError('source model changed')
 source_reader=GGUFReader(source['path']);actual={t.name:t for t in GGUFReader(path).tensors};rows=[]
 for t in source_reader.tensors:
  kind=tensor_component(t.name)
  if kind not in config.get('components',{}):continue
  format=config['components'][kind];weight=dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32);expected=encoded(weight,format);observed=actual[t.name]
  if observed.tensor_type!=FORMATS[format] or digest(observed.data)!=digest(expected):raise ValueError('deployed component code mismatch '+t.name)
  rows.append(dict(tensor=t.name,format=format,raw_codes_sha256=digest(expected),source_tensor_type=t.tensor_type.name))
 if 'head' in config.get('components',{}):
  reader=GGUFReader(target['path']);lookup={t.name:t for t in reader.tensors};t=lookup.get('output.weight',lookup['token_embd.weight']);format=config['components']['head'];weight=dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32);expected=encoded(weight,format);observed=actual['output.weight']
  if observed.tensor_type!=FORMATS[format] or digest(observed.data)!=digest(expected):raise ValueError('private head code mismatch')
  if 'token_embd.weight' in actual:raise ValueError('unexpected private embedding')
  rows.append(dict(tensor='output.weight',format=format,raw_codes_sha256=digest(expected),source_tensor=t.name,source_tensor_type=t.tensor_type.name,target_unmodified=True))
 return dict(status='DEPLOYED_COMPONENT_CODES_EXACT',model=str(path),sha256=sha(path),components=rows,training='Uses the same frozen source block-code quantization; no new trainable protected masters')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('models',type=Path,nargs='+');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();result=[audit(p) for p in a.models];a.out.write_text(json.dumps(result,indent=2)+'\n');print('Component exports audited:',len(result))
