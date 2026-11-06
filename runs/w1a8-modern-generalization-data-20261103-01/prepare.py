#!/usr/bin/env python3
"""Freeze native-tokenized TRAIN/dev inputs only; no features, labels or generation."""
import collections, hashlib, json, os, pathlib, re, subprocess, unicodedata
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).resolve().parent
SEED='w1a8-modern4b-generalization-data-v1'
SYSTEM='You are a helpful assistant. Answer directly and accurately.'
PREFIX='<|im_start|>system\n'+SYSTEM+'<|im_end|>\n'
SUFFIX='<|im_start|>assistant\n<think>\n\n</think>\n\n'
BUILD=ROOT/'runs/w1a8-modern-capture-build-20261103-08'
MODEL=ROOT/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'
SPLIT=pathlib.Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031/split.json')
BINDINGS=[ROOT/'runs/w1a8-algorithms-20261020/loop2-answer-expand01/bindings.json',ROOT/'runs/w1a8-algorithms-20261020/loop2-mode-capture05/bindings.json']
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split())
def identity(msgs,normalized=False):
 return sha(json.dumps([(m['role'],norm(m['content']) if normalized else m['content']) for m in msgs],ensure_ascii=False,separators=(',',':')).encode())
def render(msgs):
 assert len(msgs)==1 and msgs[0]['role']=='user', 'unexpected source chat; must preserve rather than flatten'
 return PREFIX+'<|im_start|>user\n'+msgs[0]['content']+'<|im_end|>\n'+SUFFIX
pins=[]
def read(p):
 b=p.read_bytes();pins.append({'path':str(p),'bytes':len(b),'sha256':sha(b)});return b
# Scope is explicit modern development request/input artifacts only; never evaluation panels.
used_files=[]
for d in sorted((ROOT/'runs').glob('w1a8-modern*')):
 if d==OUT or 'build-' in d.name or 'packed-runtime-' in d.name:continue
 used_files.extend(d.rglob('*request*.json'))
 used_files.extend(d.rglob('prompt*.txt'))
used=[]
for p in sorted(set(used_files)):
 assert not any(w in str(p.relative_to(ROOT)).lower() for w in ['sealed','holdout','final-quality','final-performance'])
 b=read(p)
 if p.suffix=='.txt':texts=[b.decode()]
 else:
  obj=json.loads(b);objs=obj if isinstance(obj,list) else [obj];texts=[]
  for item in objs:
   if not isinstance(item,dict):continue
   if isinstance(item.get('prompt'),str):texts.append(item['prompt'])
   if isinstance(item.get('messages'),list):used.append({'path':str(p),'messages':item['messages']})
 for text in texts:
  matches=re.findall(r'<\|im_start\|>(user|system)\n(.*?)<\|im_end\|>',text,re.S)
  msgs=[{'role':r,'content':c} for r,c in matches if r=='user']
  # Raw development prompt text is allowed, but is not misrepresented as templated.
  if not msgs:msgs=[{'role':'user','content':text}]
  used.append({'path':str(p),'messages':msgs,'rendered_sha256':sha(text.encode()),'same_wrapper':text.startswith(PREFIX) and text.endswith(SUFFIX)})
used_exact={identity(u['messages']) for u in used};used_normal={identity(u['messages'],True) for u in used}
raw={}
for p in BINDINGS:
 for i,r in enumerate(json.loads(read(p))['rows']):
  assert r['group_id'] not in raw and r['source_ancestry']['source_split']=='TRAIN'
  raw[r['group_id']]={**r,'binding_path':str(p),'binding_row':i}
split=json.loads(read(SPLIT));cases=split['cases'];assert len({c['group_id'] for c in cases})==len(cases)
# Exclude every ancestor group whose exact or whitespace-normalized user content was used.
overlap=[];excluded_groups=set()
for gid,r in raw.items():
 e=identity(r['messages']) in used_exact;n=identity(r['messages'],True) in used_normal
 if e or n:excluded_groups.add(gid);overlap.append({'group_id':gid,'prompt_id':r['prompt_id'],'exact':e,'normalized':n})
prior=json.loads(read(ROOT/'runs/w1a8-modern-native-replay-20261103-01/generalization-data-inventory.json'))
prior_groups={c['group_id'] for c in prior['candidates']}
build=json.loads(read(BUILD/'build.json'))
assert build['source']['sha256']==sha(read(BUILD/'source/modern_capture.cpp'))
read(BUILD/'modern_capture')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(BUILD/'runtime-overlay')+':'+str(ROOT/'runs/spec-shortlist-20261030/build/bin'),GGML_CUDA_DECODE_CONSISTENT='32')
roster=[];rejections=[];seen_normal=set();attempts=[]
for role,n in [('fit',8),('development',4)]:
 for domain in ['code','reasoning','prose']:
  pool=[c for c in cases if c['role']==role and c['domain']==domain]
  pool.sort(key=lambda c:sha((SEED+'\0'+c['group_id']).encode()))
  picked=0
  for c in pool:
   if picked>=n:break
   gid=c['group_id'];r=raw[gid];normalized=identity(r['messages'],True)
   reason='already_used_modern_input' if gid in excluded_groups else 'duplicate_normalized_message_across_roster' if normalized in seen_normal else None
   if reason:rejections.append({'group_id':gid,'reason':reason});continue
   caseid=f'{role}-{domain}-{picked:02d}';attempt=OUT/'tokenization'/gid.split(':')[-1][:16];attempt.parent.mkdir(exist_ok=True)
   text=render(r['messages']);textpath=OUT/(caseid+'-prompt.txt');textpath.write_text(text)
   argv=[str(BUILD/'modern_capture'),'--tokenize','--model',str(MODEL),'--text-file',str(textpath),'--out',str(attempt)]
   command={'argv':argv,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','GGML_CUDA_DECODE_CONSISTENT']}}
   dump(OUT/(caseid+'-tokenize-command.json'),command)
   if not (attempt/'tokenization.json').exists():
    proc=subprocess.run(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
    (attempt.parent/(attempt.name+'.stdout')).write_text(proc.stdout);(attempt.parent/(attempt.name+'.stderr')).write_text(proc.stderr)
    if proc.returncode:raise RuntimeError(f'tokenization failed {gid}: '+proc.stderr[-1000:])
   t=json.loads((attempt/'tokenization.json').read_text());assert t['vocab_only'] and not t['weights_loaded'] and not t['backend_initialized']
   assert t['model']==str(MODEL) and t['text_file']==str(textpath) and t['text_bytes']==len(text.encode())
   assert len((attempt/'tokens.ids').read_text().split())==t['token_count']
   count=t['token_count'];attempts.append({'group_id':gid,'token_count':count,'tokenization':str(attempt/'tokenization.json')})
   if count+80>384:rejections.append({'group_id':gid,'reason':'native_prompt_plus80_exceeds384','prompt_tokens':count});continue
   seen_normal.add(normalized);picked+=1
   roster.append({'id':caseid,'role':role,'domain':domain,'group_id':gid,'historical_split_role':c['role'],'historical_binding_role':r['role'],'historical_source_split':'TRAIN','binding_path':r['binding_path'],'binding_row':r['binding_row'],'prompt_id':r['prompt_id'],'source_ancestry':r['source_ancestry'],'original_receipt':r['original_receipt'],'messages':r['messages'],'message_identity_sha256':identity(r['messages']),'normalized_message_identity_sha256':normalized,'rendered_prompt':str(textpath),'rendered_prompt_sha256':sha(text.encode()),'native_tokenization':str(attempt/'tokenization.json'),'native_token_ids':str(attempt/'tokens.ids'),'native_token_ids_sha256':sha((attempt/'tokens.ids').read_bytes()),'native_tokenization_sha256':sha((attempt/'tokenization.json').read_bytes()),'native_prompt_tokens':count,'historical_prompt_tokens':c['prompt_length'],'total_rows_with80':count+80})
  assert picked==n, f'not enough cap-compatible candidates for{role}/{domain}'
# Pin source inputs at the end; concurrent new inputs require a fresh audit, not silent inclusion.
for pin in pins:assert sha(pathlib.Path(pin['path']).read_bytes())==pin['sha256'],pin['path']
selected={r['group_id'] for r in roster};assert len(selected)==36 and not(selected & excluded_groups)
audit={'schema':'w1a8_modern_input_overlap_audit_v1','scope':'all explicit *request*.json and prompt*.txt in existing runs/w1a8-modern* development dirs excluding build/source trees; no final/sealed panels','input_files':sorted({u['path'] for u in used}),'input_records':len(used),'unique_used_user_message_identities':len(used_exact),'wrapper_mismatch_paths':sorted({u['path'] for u in used if not u.get('same_wrapper',True)}),'raw_groups_checked':len(raw),'historical_split_groups_checked':len(cases),'matching_raw_groups':overlap,'rejections':rejections,'retained_from_original36':len(selected&prior_groups),'removed_from_original36':sorted(prior_groups-selected),'added_to_original36':sorted(selected-prior_groups),'historical_role_cross_tab':{f'{a}/{b}':n for (a,b),n in collections.Counter((c['role'],raw[c['group_id']]['role']) for c in cases).items()},'historical_exposure_warning':'Development groups were previously historical F16 training fit groups; they are not pristine original-corpus holdouts. Exact/normalized overlap audit does not establish semantic/topic disjointness.','source_pins':pins}
dump(OUT/'overlap-audit.json',audit)
baseline=json.loads((ROOT/'experiments/w1a8-standing-20261102/baseline.json').read_text())
packet={'schema':'w1a8_modern_phase1_generalization_roster_v1','status':'FROZEN_INPUTS_AND_BOUNDED_GEOMETRY_NO_LABELS','seed':SEED,'target_pin':baseline['rows'][0]['target_pin'],'tokenizer_build':str(BUILD/'build.json'),'rendering':{'interface':'raw-string /completion; template rendered explicitly before native tokenize','system':SYSTEM,'prefix':PREFIX,'assistant_suffix':SUFFIX,'add_special':True,'parse_special':True,'chat_template_applied_by_tokenizer':False,'thinking':'forced empty think block, matching retained modern request files'},'geometry':{'groups':36,'fit_groups':24,'development_groups':12,'max_total_target_rows':384,'max_continuation_tokens':80,'retained_teacher_rows_cap_per_group':17,'greedy_transient_full_logit_rows_cap_per_group':81,'logit_retention_note':'Native greedy80 requires all81prompt-terminal/continuation logit rows during generation;17is the final training packet retention cap, not permission to omit required greedy rows.','features_taps':[0,2,16,29],'feature_width':2560,'full_target_vocabulary':248320,'native_generation_or_labels_created':False,'do_not_truncate_source_messages':True},'counts':{r:dict(collections.Counter(c['domain'] for c in roster if c['role']==r)) for r in ['fit','development']},'native_prompt_tokens_min':min(c['native_prompt_tokens'] for c in roster),'native_prompt_tokens_max':max(c['native_prompt_tokens'] for c in roster),'actual_max_target_rows_with80':max(c['total_rows_with80'] for c in roster),'roster':roster,'untouched_definition':'No exact or whitespace-normalized match with discovered modern development input artifacts, and group-disjoint from this modern fit arm. Historical F16 TRAIN exposure remains.','phase':'1:generalization; not feature screening or predictor calibration','pending_gates':['native numeric admission','unchanged-parent native success protection masks','training recipe/scale-only approval as scoped by root','native token correctness/quality/full-request timing with unchanged controls'],'audit':str(OUT/'overlap-audit.json')}
dump(OUT/'roster.json',packet)
read(OUT/'prepare.py')
dump(OUT/'freeze.json',{'schema':'w1a8_modern_data_preparation_freeze_v1','labels_generated':False,'features_captured':False,'weights_loaded':False,'gpu_backend_initialized':False,'pins':pins+[{'path':str(OUT/f),'bytes':(OUT/f).stat().st_size,'sha256':sha((OUT/f).read_bytes())} for f in ['roster.json','overlap-audit.json']]})
print(json.dumps({'groups':len(roster),'counts':packet['counts'],'prompt_tokens':[packet['native_prompt_tokens_min'],packet['native_prompt_tokens_max']],'max_rows_with80':packet['actual_max_target_rows_with80'],'used_inputs':len(used_exact),'overlap_groups':len(overlap),'rejections':rejections,'retained_original':len(selected&prior_groups)}))
