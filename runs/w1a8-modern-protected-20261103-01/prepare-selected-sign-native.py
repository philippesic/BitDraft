from pathlib import Path
import json,hashlib,copy
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';Q=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-selected-sign-qualification-20261103-01');F=R/'selected-sign-native01';F.mkdir(exist_ok=False)
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
result=json.loads((Q/'result.json').read_text());assert result['status']=='SELECTED_O_GPU_BACKWARD_RESUME_EXPORT_PLUMBING_PASS';frozen=json.loads((Q/'freeze.json').read_text());manifest=Path(frozen['manifest']['path']);m=json.loads(manifest.read_text());selected=frozen['windows'];assert len(selected)==2 and all(w['role']=='fit' for w in selected);assert len({w['group_id'] for w in selected})==2
byid={w['id']:w for w in m['windows']}
for w in selected:assert w==byid[w['id']]
template=json.loads((R/'actual-native-cases01/cases.json').read_text());fields=['id','role','domain','group','group_id','round','anchor','seed_token','labels','max_depth','native_prefix','native_verified_prefix','native_proposals','native_success_mask','native_supervision_reachable_mask','prefix_tokens','prefix_raw_rows','seed_raw_row'];cases=[];files=[];filemap={};rawrows={}
for w in selected:
 cases.append({**{k:w[k] for k in fields},'classification':'selected_sign_two_FIT_plumbing_only'})
 for rid in w['prefix_raw_rows']+[w['seed_raw_row']]:
  if rid in rawrows:continue
  row=m['raw_rows'][rid];fid=row['file']
  if fid not in filemap:filemap[fid]=len(files);files.append(m['raw_files'][fid])
  rawrows[rid]=[rid,filemap[fid],row['row'],row['raw_sha256'],row['target_position']]
base={k:template[k] for k in ['schema','diagnostic_only','manifest','build','helper','runtime_libraries','cpu_library','cuda_library','target','draft_map','raw_row_columns']};packets=[]
for name,key in [('learned','learned_export'),('FORCED-DIAGNOSTIC','forced_diagnostic_export')]:
 model=result[key];assert pin(model['path'])==model;packet=copy.deepcopy(base);packet.update(models={'parent':m['parent'],'candidate':model},files=files,raw_rows=list(rawrows.values()),cases=cases,model_choice=pin(Q/'result.json'),selection={'scope':'two unchanged FIT windows used for plumbing only; not generalization or all-success preservation','qualification':pin(Q/'freeze.json'),'model_kind':name,'source':pin(__file__)});path=F/(name+'-cases.json');write(path,packet);packets.append({'kind':name,'packet':pin(path),'GPU_readout':pin(Q/('learned-fit.json' if name=='learned' else 'forced-fit.json'))})
write(F/'manifest.json',{'schema':'selected_sign_native_plumbing_v1','packets':packets,'source_qualification':pin(Q/'result.json'),'native_full_HTTP_also_required':True})
models=json.loads((R/'fresh-quarter-confirmation01/models.json').read_text());models['arms']=[a for a in models['arms'] if a['id']!='w1a8_quarter'];source=next(a for a in models['arms'] if a['id']=='w1a8_initial');extra=[]
for name,key in [('learned','learned_export'),('FORCED_DIAGNOSTIC','forced_diagnostic_export')]:
 a=copy.deepcopy(source);a['id']='w1a8_sign_'+name;a['model']=result[key];a['runtime']['log_verbosity']=4;a['runtime'].pop('loader_qualification',None);extra.append(a)
models['arms']=models['arms'][:-1]+extra+models['arms'][-1:];models['source_pins'] += [pin(Q/'result.json'),pin(Q/'freeze.json')];models['data_usage']={'scope':'plumbing qualification on reused12dev+2FITwarmups; forced one-bit arm is ONLY a packing diagnostic','not_capacity_or_fresh':True};write(R/'selected-sign-models-verbose.json',models)
src=(R/'qualify-quarter.py').read_text().replace("R/'quarter-models-verbose.json'","R/'selected-sign-models-verbose.json'").replace("R/'quarter-http-qualification01'","R/'selected-sign-http-qualification01'").replace("a['id']=='w1a8_quarter'","a['id'] in ['w1a8_sign_learned','w1a8_sign_FORCED_DIAGNOSTIC']").replace('QUARTER_NATIVE_HTTP_QUALIFIED_NOT_REPEATED_TIMING','SELECTED_SIGN_TWO_UPDATE_AND_FORCED_BIT_NATIVE_HTTP_QUALIFIED_NOT_CAPACITY');(R/'qualify-selected-sign-http.py').write_text(src);print({'packets':len(packets),'FITwindows':len(selected),'HTTP_models':[a['id'] for a in extra]})
