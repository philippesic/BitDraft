from pathlib import Path
import hashlib,json,os,sys,subprocess,time,signal,shutil
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01');R=P/'runs/w1a8-modern-protected-20261103-01'
sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
import modern_actual_data as actual
import modern_actual_union as union
import modern_expand_capture as collect
start=time.monotonic();stopping=False;stages=[]
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
def auth(ref):
 assert pin(ref['path'])==ref,ref['path'];return Path(ref['path'])
def guard(reserve=0):
 assert not stopping
 c=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080'];assert not c.get('pause_requested') and not c.get('stop_all_requested')
 assert shutil.disk_usage(P).free>=2*1024**3+64*1024**2
 assert shutil.disk_usage(C).free>=2*1024**3+64*1024**2+reserve
state=json.loads((P/'runs/local-w1a8-modern-fit-expansion-20261103-01/state.json').read_text());assert state['status']=='finished' and state['exit_code']==0
baseline_proof=json.loads((C/'frozen-target-check01/result.json').read_text());assert baseline_proof['status']=='NEW24FIT_LABELS_EXACT_FROZEN_TARGET_RUNTIME'
capture=C/'capture01';summary=json.loads((capture/'summary.json').read_text());freeze=json.loads((capture/'freeze.json').read_text());roster=json.loads(auth(freeze['roster']).read_text());collect.validate_roster(roster)
assert summary['status']=='ACTUAL_PARENT_COLLECTION_COMPLETE_WITH_ACCOUNTED_FAILURES' and len(summary['groups'])==24
assert {x['id'] for x in summary['groups']}=={x['id'] for x in roster['rows']};assert all(x['status']=='ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS' and x['role']=='fit' for x in summary['groups'])
for ref in freeze['source_pins']+freeze['input_pins']:auth(ref)
released=0;arrays={};targetrefs=0
for group in summary['groups']:
 cell=capture/group['id'];response=json.loads((cell/'response.json').read_text());refs=json.loads((cell/'target-reference.json').read_text());expected=json.loads(auth(refs['response']).read_text());request=json.loads(auth(refs['request']).read_text());assert json.loads((cell/'request.json').read_text())==request;assert request=={'prompt':Path(next(r['rendered_text_path'] for r in roster['rows'] if r['id']==group['id'])).read_text(),**collect.protocol()};assert collect.response_signature(response)==collect.response_signature(expected);assert response['timings']['cache_n']==expected['timings']['cache_n']==0;targetrefs+=1
 proof=json.loads(auth(group['protection']).read_text());assert proof['emitted_draft_successes']==group['emitted_draft_successes']
 for ref in proof['native_output_pair_audit']['trace_files']:arrays[ref['path']]=ref
for f in [capture/'target-only/release.json']+[capture/g['id']/'release.json' for g in summary['groups']]:
 d=json.loads(f.read_text());assert d['owned_process_reaped'] and d['owned_process_group_absent'];pid=d['identity']['pid'];proc=Path('/proc')/str(pid);assert not proc.exists() or int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])!=d['identity']['start_ticks'];released+=1
old_embed=P/'runs/w1a8-modern-actual-embedding-20261103-01';guard(160*1024**2);t=time.monotonic();pre=actual.prepare(capture,C/'views-pre01',embedding_results=[old_embed]);stages.append({'stage':'prepare_preliminary','wall_seconds':time.monotonic()-t,'windows':len(pre['windows']),'missing_embeddings':len(pre['missing_embedding_ids'])})
missing=pre['missing_embedding_ids'];embeds=[old_embed]
if missing:
 assert len(missing)==len(set(missing)) and len(missing)<=3000
 emb=C/'embedding01';emb.mkdir();tokenfile=emb/'tokens.ids';tokenfile.write_text('\n'.join(map(str,missing))+'\n');b=json.loads((P/'runs/w1a8-modern-capture-build-20261103-09/build.json').read_text());target=freeze['target'];auth(target)
 for ref in [b['source'],b['builder'],b['executable']]+b['runtime_libraries']+[b['cuda_overlay']]:auth(ref)
 clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env={**clean,**b['environment'],'CUDA_VISIBLE_DEVICES':'0','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda')}
 argv=[b['executable']['path'],'--model',target['path'],'--tokens',str(tokenfile),'--out',str(emb/'capture'),*b['required_backend_argv'],'--taps','0','--logit-positions',str(len(missing)-1),'--prefill-chunk','512','--decode-chunk','1','--threads','4','--output-cap','1','--max-output-mib','32']
 (emb/'freeze.json').write_text(json.dumps({'source':pin(__file__),'tokens':pin(tokenfile),'target':target,'build':pin(P/'runs/w1a8-modern-capture-build-20261103-09/build.json'),'argv':argv,'scope':'native layer0 token lookup only; no conditional hidden teacher features'},indent=2)+'\n');guard(160*1024**2);t=time.monotonic()
 with (emb/'stdout.log').open('x') as log:
  process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);proc=Path('/proc')/str(process.pid);(emb/'identity.json').write_text(json.dumps({'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n');assert process.wait(timeout=120)==0
 m=json.loads((emb/'capture/capture.json').read_text());assert m['feature_shape']==[len(missing),2560] and m['taps']==[0] and m['input_tokens']==missing
 (emb/'result.json').write_text(json.dumps({'status':'NATIVE_EMBEDDING_ROWS_CAPTURED','wall_seconds':time.monotonic()-t,'capture':pin(emb/'capture/capture.json'),'layer0':pin(emb/'capture/layer-0.f32'),'unique_ids':len(missing)},indent=2)+'\n');stages.append({'stage':'native_embedding','wall_seconds':time.monotonic()-t,'unique_ids':len(missing)});embeds.append(emb)
guard(140*1024**2);t=time.monotonic();new=actual.prepare(capture,C/'views-final01',embedding_results=embeds);assert not new['missing_embedding_ids'];stages.append({'stage':'prepare_final','wall_seconds':time.monotonic()-t,'counts':new['counts'],'native_protections':new['native_protections']})
guard(100*1024**2);t=time.monotonic();old=P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json';destination=C/'union01';union.prepare_union([old,C/'views-final01/manifest.json'],destination);joined=json.loads((destination/'manifest.json').read_text());stages.append({'stage':'source_index_union','wall_seconds':time.monotonic()-t,'counts':joined['counts']})
proof={'status':'ROOT_NEW24FIT_CAPTURE_AUDIT_AND_DATA_UNION_COMPLETE_NOT_TRAINED','root_source':pin(__file__),'frozen_target_crosscheck':pin(C/'frozen-target-check01/result.json'),'capture_summary':pin(capture/'summary.json'),'roster':pin(C/'inputs01/roster.json'),'views':pin(C/'views-final01/manifest.json'),'union':pin(destination/'manifest.json'),'target_reference_pairs_exact':targetrefs,'server_groups_released':released,'native_trace_files_preserved':len(arrays),'stages':stages,'wall_seconds':time.monotonic()-start,'scope':'newFIT only; oldDEV/old549 kept, no repeatedthroughput/newqualityclaim'};(R/'fit-expansion-data-root-result.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2),flush=True)
