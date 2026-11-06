from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;manifest=json.loads((P/'environment-model-manifest.json').read_text());known={x['path']:x['sha256'] for x in manifest['models'] if 'sha256' in x};rows=[]
for x in json.loads((P/'expected-inputs.json').read_text()):
 f=P/'models'/x.get('dest',x['filename']);source=f
 if f.name=='config.json' and (f.parent/'config.source.json').exists():source=f.parent/'config.source.json'
 expected=x['expected'];actual=known.get(str(source))
 if actual is None:
  with source.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
 row={'path':str(f),'source_checked':str(source),'repo':x['repo'],'revision':x['revision'],'sha256':actual,'bytes':source.stat().st_size,'expected_bytes':expected.get('size')}
 if expected.get('lfs'):row.update(expected_sha256=expected['lfs']['sha256'],match=actual==expected['lfs']['sha256'] and source.stat().st_size==expected['size'])
 else:row['match']=hashlib.sha1(('blob '+str(source.stat().st_size)+'\0').encode()+source.read_bytes()).hexdigest()==expected['blobId']
 if source!=f:row['normalized_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
 rows.append(row)
(P/'input-validation.json').write_text(json.dumps(rows,indent=2));assert all(r['match'] for r in rows);print('All',len(rows),'pinned input blobs match')
