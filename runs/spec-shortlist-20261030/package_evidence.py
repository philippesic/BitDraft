from pathlib import Path
import json,gzip,hashlib,os,time
P=Path(__file__).resolve().parent
stage=P/'sealed-evidence';stage.mkdir(exist_ok=False);files={}
patterns=['*.py','*.json','*.log','measurements/**/*.json','measurements/**/*.jsonl','measurements/**/server.log','build/CMakeCache.txt']
selected=set()
for pattern in patterns:
 for f in P.glob(pattern):
  if f.is_file() and 'sealed-evidence' not in f.parts:selected.add(f)
for f in sorted(selected):
 if f.stat().st_size>12*1024**2:raise ValueError('unexpected evidence size '+str(f))
 b=f.read_bytes();files[str(f.relative_to(P))]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
payload=json.dumps({'created_at':time.time(),'remote_root':str(P),'files':files},separators=(',',':')).encode();b=gzip.compress(payload,compresslevel=6)
f=stage/'evidence.json.gz';f.write_bytes(b);assert len(b)<8*1024**2
inventory={f.name:hashlib.sha256(b).hexdigest()};(stage/'source-hashes.json').write_text(json.dumps(inventory));print(json.dumps({'inventory':inventory,'compressed_bytes':len(b),'files':len(files)}))
