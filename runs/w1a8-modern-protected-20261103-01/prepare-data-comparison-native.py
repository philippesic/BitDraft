from pathlib import Path
import json,hashlib,copy
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');M=C/'w1a8-modern-fit-expansion-20261103-01/union01/manifest.json';m=json.loads(M.read_text());template=json.loads((R/'actual-native-cases01/cases.json').read_text());assert m['native_protections']=={'development':193,'fit':763}
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
windows=sorted([w for w in m['windows'] if w['native_prefix']>0],key=lambda w:(w['role'],w['domain'],w['group'],w['round']));assert len(windows)==883 and sum(w['native_prefix'] for w in windows)==956
fields=['id','role','domain','group','group_id','round','anchor','seed_token','labels','max_depth','native_prefix','native_verified_prefix','native_proposals','native_success_mask','native_supervision_reachable_mask','prefix_tokens','prefix_raw_rows','seed_raw_row'];choices=[]
for name,modelroot in [('control',C/'w1a8-expanded-ce-control3840-01'),('expanded',C/'w1a8-expanded-ce-resident-new24-3840-01')]:
 d=json.loads((modelroot/'result.json').read_text());assert pin(d['model']['path'])==d['model'];out=R/f'data-{name}-all-native-cases01';out.mkdir();packets=[]
 for index,start in enumerate(range(0,len(windows),96)):
  subset=windows[start:start+96];filemap={};files=[];rows={}
  for w in subset:
   for rid in w['prefix_raw_rows']+[w['seed_raw_row']]:
    if rid in rows:continue
    row=m['raw_rows'][rid];fid=row['file']
    if fid not in filemap:filemap[fid]=len(files);files.append(m['raw_files'][fid])
    rows[rid]=[rid,filemap[fid],row['row'],row['raw_sha256'],row['target_position']]
  packet={k:template[k] for k in ['schema','diagnostic_only','build','helper','runtime_libraries','cpu_library','cuda_library','target','draft_map','raw_row_columns']};packet.update(manifest=pin(M),models={'parent':m['parent'],'candidate':d['model']},files=files,raw_rows=list(rows.values()),cases=[{**{k:w[k] for k in fields},'classification':'all_original_and_new_emitted_native_successes'} for w in subset]);f=out/f'cases-{index:02d}.json';blob=json.dumps(packet,separators=(',',':')).encode();assert len(blob)<=2*1024**2;f.write_bytes(blob);packets.append({'packet':pin(f),'windows':len(subset),'native_successes':sum(w['native_prefix'] for w in subset)})
 record={'schema':'all956_native_success_cases_v1','scope':'old549+new407, all60groups; no model-outcome selection','model':d['model'],'training_result':pin(modelroot/'result.json'),'manifest':pin(M),'packets':packets,'windows':len(windows),'native_tokens':956};(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');choices.append({'arm':name,'packet_manifest':pin(out/'manifest.json'),'output':str(C/'w1a8-modern-data-native-comparison-20261103-01'/name)})
(R/'data-native-comparison-plan.json').write_text(json.dumps({'scope':'matchedold/fullnewCE all956nativepreservation; notservingthroughput/freshclaim','arms':choices,'source':pin(__file__),'old_control_exact':pin(R/'expanded-ce-training-control-parity.json'),'resident_control_exact':pin(R/'resident-ce-GPU-control-parity.json')},indent=2)+'\n');print([(x['arm'],x['output']) for x in choices])
