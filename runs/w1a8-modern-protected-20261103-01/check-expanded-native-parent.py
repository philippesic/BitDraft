from pathlib import Path
import json,hashlib,subprocess,os,time,signal
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01');M=C/'views-final01/manifest.json';m=json.loads(M.read_text());assert not m['missing_embedding_ids'] and m['native_protections']=={'fit':407};assert len(m['groups'])==24 and all(g['role']=='fit' for g in m['groups']);assert (C/'embedding01/result.json').exists()
# Other root supervisor has completed its final GPU call and now audits CPU-only union metadata.
embed=json.loads((C/'embedding01/identity.json').read_text());proc=Path('/proc')/str(embed['pid']);assert not proc.exists() or int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])!=embed['start_ticks']
start=time.monotonic();out=R/'expanded-data-parent-native01';out.mkdir();template=json.loads((R/'actual-native-cases01/cases.json').read_text());windows=sorted([w for w in m['windows'] if w['native_prefix']>0],key=lambda w:(w['domain'],w['group'],w['round']));assert sum(w['native_prefix'] for w in windows)==407
fields=['id','role','domain','group','group_id','round','anchor','seed_token','labels','max_depth','native_prefix','native_verified_prefix','native_proposals','native_success_mask','native_supervision_reachable_mask','prefix_tokens','prefix_raw_rows','seed_raw_row'];packets=[]
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
for i,startrow in enumerate(range(0,len(windows),96)):
 selected=windows[startrow:startrow+96];filemap={};files=[];rows={}
 for w in selected:
  for rid in w['prefix_raw_rows']+[w['seed_raw_row']]:
   if rid in rows:continue
   row=m['raw_rows'][rid];fid=row['file']
   if fid not in filemap:filemap[fid]=len(files);files.append(m['raw_files'][fid])
   rows[rid]=[rid,filemap[fid],row['row'],row['raw_sha256'],row['target_position']]
 packet={k:template[k] for k in ['schema','diagnostic_only','build','helper','runtime_libraries','cpu_library','cuda_library','target','draft_map','raw_row_columns']};packet.update(manifest=pin(M),models={'parent':m['parent'],'candidate':m['parent']},files=files,raw_rows=list(rows.values()),cases=[{**{k:w[k] for k in fields},'classification':'all_new_actual_parent_emitted_successes'} for w in selected]);f=out/f'cases-{i:02d}.json';blob=json.dumps(packet,separators=(',',':')).encode();assert len(blob)<=2*1024**2;f.write_bytes(blob);packets.append(pin(f))
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GGML_CUDA_DECODE_CONSISTENT':'32','GGML_W1AX_ACT_BITS':'8','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','LD_LIBRARY_PATH':str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin')};env.pop('LD_PRELOAD',None);results=[]
for i,ref in enumerate(packets):
 batch=out/f'batch-{i:02d}';batch.mkdir();argv=[str(R/'actual-native-build01/modern_actual_native_replay'),'--cases',ref['path'],'--model','parent','--out',str(batch/'parent'),'--allow-cuda']
 with (batch/'parent.log').open('x') as log:
  child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc')/str(child.pid);(batch/'identity.json').write_text(json.dumps({'pid':child.pid,'pgid':os.getpgid(child.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n');assert child.wait()==0,(i,child.returncode)
 d=json.loads((batch/'parent/replay.json').read_text());assert d['native_reference_tokens']==d['calibrated_reference_tokens'] and d['native_successes_outside_parent_calibration']==0 and d['lost_calibrated_reference_tokens']==0;results.append({'batch':i,'native_tokens':d['native_reference_tokens'],'windows':len(d['windows']),'replay':pin(batch/'parent/replay.json'),'wall_seconds':d['wall_seconds']});print(json.dumps(results[-1]),flush=True)
assert sum(d['native_tokens'] for d in results)==407
(out/'result.json').write_text(json.dumps({'status':'ALL407_NEW_NATIVE_PARENT_SUCCESSES_REPRODUCED','model':m['parent'],'manifest':pin(M),'packets':packets,'results':results,'native_successes':407,'windows':len(windows),'wall_seconds':time.monotonic()-start,'scope':'sameparent allnewnativepositions, source reconstruction check; not trainedgeneralization/servingperformance'},indent=2)+'\n')
