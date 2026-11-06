from pathlib import Path
import json,hashlib,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';manifest=P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json';m=json.loads(manifest.read_text());template=json.loads((R/'actual-native-cases01/cases.json').read_text());choice=json.loads((R/'body-quarter-export.json').read_text());output=R/'quarter-all-native-cases01';assert not output.exists();output.mkdir()
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
windows=sorted([w for w in m['windows'] if w['native_prefix']>0],key=lambda w:(w['role'],w['group'],w['round']));assert sum(w['native_prefix'] for w in windows)==549
assert {role:sum(w['native_prefix'] for w in windows if w['role']==role) for role in ['fit','development']}=={'fit':356,'development':193}
fields=['id','role','domain','group','group_id','round','anchor','seed_token','labels','max_depth','native_prefix','native_verified_prefix','native_proposals','native_success_mask','native_supervision_reachable_mask','prefix_tokens','prefix_raw_rows','seed_raw_row']
packets=[];seen=[]
for index,start in enumerate(range(0,len(windows),96)):
 subset=windows[start:start+96];filemap={};files=[];rawrows={};cases=[]
 for w in subset:
  assert w['native_success_mask']==[d<w['native_prefix'] for d in range(w['max_depth'])]
  case={k:w[k] for k in fields};case['classification']='complete_original_native_success_set';cases.append(case);seen.append(w['id'])
  for rid in w['prefix_raw_rows']+[w['seed_raw_row']]:
   if rid in rawrows:continue
   row=m['raw_rows'][rid];fid=row['file']
   if fid not in filemap:filemap[fid]=len(files);files.append(m['raw_files'][fid])
   rawrows[rid]=[rid,filemap[fid],row['row'],row['raw_sha256'],row['target_position']]
 packet={k:template[k] for k in ['schema','diagnostic_only','manifest','build','helper','runtime_libraries','cpu_library','cuda_library','target','draft_map','raw_row_columns']}
 packet.update(models={'parent':m['parent'],'candidate':choice['output']},files=files,raw_rows=list(rawrows.values()),cases=cases,model_choice=pin(R/'body-quarter-export.json'),selection={'scope':'Every original native-success window, no model-outcome selection','batch':index,'total_batches':(len(windows)+95)//96,'global_windows':len(windows),'global_original_native_success_tokens':549,'global_fit_successes':356,'global_development_successes':193,'source_manifest':pin(manifest),'derivation_source':pin(__file__),'no_fresh_examples':True})
 blob=json.dumps(packet,separators=(',',':'),allow_nan=False).encode();assert len(blob)<=2*1024**2
 path=output/f'cases-{index:02d}.json';path.write_bytes(blob);packets.append({'packet':pin(path),'windows':len(subset),'native_successes':sum(w['native_prefix'] for w in subset)})
assert len(seen)==len(set(seen))==len(windows)
summary={'schema':'complete_native_success_partition_v1','packets':packets,'all_original_native_success_tokens':549,'all_original_success_windows':len(windows),'all_groups':len({w['group'] for w in windows}),'candidate':choice['output'],'model_choice':pin(R/'body-quarter-export.json'),'original_manifest':pin(manifest),'all_window_ids':seen,'array_bytes_copied':0};(output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'windows':len(windows),'successes':549,'groups':summary['all_groups'],'packets':len(packets),'bytes':sum(p['packet']['bytes'] for p in packets)}))
