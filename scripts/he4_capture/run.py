"""2080 Ti only, capture-only campaign with durable receipts and terminal webhook."""
import argparse,collections,datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback,urllib.request
import numpy as np
P=pathlib.Path('/home/philip/he4-capture-2080ti')
TAPS=[2,6,10,14,18,22,26,30]
PIN='3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630'
MODEL=P/'Qwen3.5-4B-Q4_K_M.gguf';BIN=P/'capture'
DEADLINE=datetime.datetime.fromisoformat('2026-11-09T20:00:00+00:00').timestamp()
LIB=P/'build/bin';CUDA=LIB/'libggml-cuda.so.0.28.0'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0',GGML_CUDA_DECODE_CONSISTENT='32',LD_LIBRARY_PATH=str(LIB)+':/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/lib:/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/targets/x86_64-linux/lib',CUDA_CACHE_PATH=str(P/'cuda-cache'))

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def command(argv,log,timeout=600):
 with log.open('w') as f:subprocess.run(list(map(str,argv)),env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def broad(i):
 # Authored input prompts only. No labels or evaluation answers are synthesized.
 templates=[
 ('math',f'A shop packs {7+i%37} items per box and ships {3+i%53} boxes. It receives {11+i%71} returned items. Explain how to calculate the net number delivered.'),
 ('dialogue',f'Customer: I ordered {2+i%9} desk lamps, but only {1+i%2} arrived. Agent: I can help. Continue this support dialogue with a question and a practical resolution. Order reference TRAIN-{i:06d}.'),
 ('instructions',f'Write a checklist for organizing a community workshop for {12+i%87} people with {2+i%7} volunteers and a budget of {100+i*3} dollars. Include accessibility and cleanup.'),
 ('structured_output',f'Return JSON describing three fictional library books. Use keys title, author, year, available. Start the publication years at {1950+i%70}; set available to a boolean.'),
 ('translation',f'Translate this notice into Spanish and French: The library will close at {1+i%11} pm on Saturday for maintenance. Please return borrowed equipment by Friday.'),
 ('editing',f'Rewrite this email politely and concisely: hey team, we need {3+i%12} more chairs for room {100+i%200} before the workshop starts. tell me if you can bring some.'),
 ('planning',f'Plan a {2+i%5}-day walking holiday for a family with a child aged {5+i%10}. Assume a fictional coastal town, include rainy-day alternatives and time to rest.'),
 ('classification',f'Classify each message as request, complaint, or appreciation, and explain briefly: 1. Please send {2+i%13} copies. 2. My parcel TRAIN-{i} arrived damaged. 3. Thank you for the helpful explanation.'),
 ]
 domain,text=templates[i%len(templates)]
 return dict(id=f'authored-broad-{i:07d}',group_id=f'authored-broad-{i:07d}',domain=domain,pool='authored_broader_domains',rendered_text='\n'.join(['<|im_start|>system','You are a helpful assistant. Answer directly and accurately.<|im_end|>','<|im_start|>user',text+'<|im_end|>','<|im_start|>assistant','<think>','','</think>','','']))

def verify(out,row):
 c=json.loads((out/'capture.json').read_text());n,w=c['feature_shape'];k,v=c['logits_shape'];a=c['greedy_ids'];pos=c['logit_positions'];prompt=c['prompt_tokens']
 assert c['status']=='native_capture_complete' and c['taps']==TAPS and w==2560
 assert c['feature_semantics']=='zero_based_pre_block_input_all_tokens'
 assert c['settings']['consistent']==32 and c['settings']['kv']=='F16'
 assert '2080 Ti' in c['gpu_description']
 assert 1<=k<=128 and k==len(a)==len(pos) and pos==list(range(prompt-1,prompt-1+k))
 assert len(c['input_tokens'])==n==prompt+k-1
 assert a[:-1]==c['input_tokens'][prompt:]
 assert list(map(int,(out.parent/'tokenization/tokens.ids').read_text().split()))==c['input_tokens'][:prompt]
 assert not any(t in c['eog_ids'] for t in a[:-1])
 logits=np.memmap(out/'logits.f32',dtype='<f4',mode='r',shape=(k,v))
 assert np.isfinite(logits).all();argmax=logits.argmax(axis=1)
 assert np.array_equal(argmax,np.array(a))
 scratch=pin(out/'logits.f32');del logits
 files=[]
 for tap in TAPS:
  f=out/f'layer-{tap}.f32';assert f.stat().st_size==n*w*4
  x=np.memmap(f,dtype='<f4',mode='r',shape=(n,w));assert np.isfinite(x).all();del x;files.append(pin(f))
 np.save(out/'labels.npy',np.asarray(a,dtype='<i4'));np.save(out/'label_positions.npy',np.asarray(pos,dtype='<i4'));np.save(out/'inputs.npy',np.asarray(c['input_tokens'],dtype='<i4'))
 receipt=dict(status='VERIFIED_NATIVE_HARD_LABEL_CAPTURE',id=row['id'],group_id=row['group_id'],domain=row['domain'],pool=row['pool'],captured_at=now(),metadata=pin(out/'capture.json'),files=files+[pin(out/f) for f in ('labels.npy','label_positions.npy','inputs.npy')]+[pin(out.parent/f) for f in ('source.json','prompt.txt','tokenization/tokens.ids','tokenization/tokenization.json','launch.json')],full_logits=scratch,greedy_vs_full_logits_argmax=True,verified_rows=k,no_post_eog_labels=True,eog_observed=a[-1] in c['eog_ids'],sequences=1,windows=max(0,k-14),unique_label_positions=k,unique_label_positions_in_windows=k if k>=15 else 0,feature_semantics=c['feature_semantics'],taps=TAPS,feature_shape=[n,w],logit_scratch_deleted=True)
 write(out/'verification.json',receipt)
 (out/'logits.f32').unlink()
 return pin(out/'verification.json'),receipt


def run():
 argparse.ArgumentParser().parse_args()
 state=dict(status='preflight',started=now(),gpu_initially_idle=True,wait_for_other_gpu_job_seconds=0,sequences_captured=0,windows=0,unique_label_positions=0,output_paths=[str(P/'captures'),str(P/'manifest.json')],blocker=None)
 write(P/'status.json',state)
 manifest=dict(schema='he4_native_teacher_capture_v1',status='running',hardware='DESKTOP-T4JKGJB RTX 2080 Ti SM75',deadline_utc='2026-11-09T20:00:00Z',training_executed=False,target_sha256=PIN,taps=TAPS,feature_semantics='zero_based_pre_block_input_all_tokens',contract=dict(max_new_tokens=128,sampling='native raw greedy argmax lowest-id tie',consistent=32,kv='F16',window_length=15,stride=1,eog='include first EOG label; no post-EOG inputs/labels'),records=[],failures=[],domain_mix={},sequences=0,windows=0,unique_label_positions=0,unique_label_positions_in_windows=0,trainer=dict(index=str(P/'manifest.json'),record_receipt='records[].receipt.path',features='verification files layer-N.f32: little-endian F32 [feature_shape[0],2560]',inputs='inputs.npy int32',labels='labels.npy int32 native greedy targets',label_positions='label_positions.npy maps labels to feature row; each length15 overlapping label slice is one window',instruction='Verify manifest.sha256 and every receipt/file hash before import; TRAIN only; no KL needed; do not include failed/unverified directories'),exclusions=pin(P/'exclusions.json'),prompt_pool=pin(P/'prompt-pool.json'),runtime_receipt=pin(P/'runtime-receipt.json'))
 def checkpoint():
  manifest['updated_at']=now();write(P/'manifest.json',manifest);(P/'manifest.sha256').write_text(sha(P/'manifest.json')+'  manifest.json\n')
  state.update(updated_at=now(),sequences_captured=manifest['sequences'],windows=manifest['windows'],unique_label_positions=manifest['unique_label_positions'],free_disk_bytes=shutil.disk_usage(P).free);write(P/'status.json',state)
 try:
  assert sha(MODEL)==PIN
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip();assert gpu=='NVIDIA GeForce RTX 2080 Ti'
  wait=time.time()
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
   state['status']='waiting_for_existing_GPU_job';checkpoint()
   if time.time()>=DEADLINE:raise RuntimeError('GPU occupied through deadline')
   time.sleep(30)
  state['wait_for_other_gpu_job_seconds']=round(time.time()-wait,2)
  rows=json.loads((P/'prompt-pool.json').read_text())['records'];exclude=set(json.loads((P/'exclusions.json').read_text())['identities']);seen=set();i=0;b=0
  (P/'captures').mkdir(exist_ok=True)
  while time.time()<DEADLINE:
   row=rows[i] if i<len(rows) else broad(b)
   if i>=len(rows):b+=1
   i+=1
   raw=row['rendered_text'];h=hashlib.sha256(raw.encode()).hexdigest()
   normalized=' '.join(__import__('unicodedata').normalize('NFKC',raw).split())
   # Broad prompts also pass the frozen identity filters; never use held-out text.
   raw_user=raw.split('<|im_start|>user\n',1)[-1].split('<|im_end|>',1)[0]
   normalized_user=' '.join(__import__('unicodedata').normalize('NFKC',raw_user).split())
   hashes={h,hashlib.sha256(normalized.encode()).hexdigest(),hashlib.sha256(raw_user.encode()).hexdigest(),hashlib.sha256(normalized_user.encode()).hexdigest()}
   if hashes&exclude or h in seen:continue
   seen.add(h)
   root=P/'captures'/f'shard-{manifest["sequences"]//100:04d}';root.mkdir(exist_ok=True)
   entry=root/f'{i-1:06d}';entry.mkdir()
   write(entry/'source.json',row);(entry/'prompt.txt').write_text(raw)
   state.update(status='tokenizing',current_id=row['id'],phase=row['pool']);checkpoint()
   command([BIN,'--tokenize','--model',MODEL,'--text-file',entry/'prompt.txt','--out',entry/'tokenization'],entry/'tokenization.log',120)
   tokens=(entry/'tokenization/tokens.ids').read_text().split();needed=(len(tokens)+128)*2560*8*4+129*248320*4
   write(entry/'disk-estimate.json',dict(prompt_tokens=len(tokens),feature_bytes_upper_bound=(len(tokens)+128)*81920,full_logits_scratch_bytes_upper_bound=129*248320*4,min_free_after_bytes=30*1024**3))
   if len(tokens)+128>4096:
    manifest['failures'].append(dict(id=row['id'],reason='over context capacity; no cropping'));continue
   if shutil.disk_usage(P).free<needed+30*1024**3:
    state['blocker']='Disk safety reserve reached; retained verified shards preserved';break
   state['status']='capturing';checkpoint()
   argv=[BIN,'--model',MODEL,'--tokens',entry/'tokenization/tokens.ids','--out',entry/'capture','--cpu-library',LIB/'libggml-cpu.so.0.28.0','--cuda-library',CUDA,'--greedy-steps','128','--taps',','.join(map(str,TAPS)),'--prefill-chunk','32','--threads','6','--output-cap','32','--max-output-mib','1024']
   write(entry/'launch.json',dict(argv=list(map(str,argv)),environment={k:ENV[k] for k in ('CUDA_VISIBLE_DEVICES','GGML_CUDA_DECODE_CONSISTENT','LD_LIBRARY_PATH')}))
   command(argv,entry/'capture.log')
   receipt,record=verify(entry/'capture',row)
   manifest['records'].append(dict(id=row['id'],receipt=receipt,pool=row['pool'],domain=row['domain']))
   for key in ('sequences','windows','unique_label_positions','unique_label_positions_in_windows'):manifest[key]+=record[key]
   manifest['domain_mix'][row['domain']]=manifest['domain_mix'].get(row['domain'],0)+1
   checkpoint()
  state['status']='complete' if manifest['sequences'] else 'failed'
  if not manifest['sequences']:state['blocker']=state['blocker'] or 'No captures completed'
 except BaseException as e:
  state.update(status='failed',blocker=str(e));(P/'failure.txt').write_text(traceback.format_exc())
 finally:
  manifest['status']=state['status'];manifest['blocker']=state['blocker'];checkpoint()
  body=dict(status=state['status'],sequences_captured=manifest['sequences'],unique_label_positions=manifest['unique_label_positions'],windows=manifest['windows'],output_paths=state['output_paths'],blocker=state['blocker'])
  write(P/'completion.json',body)
  # Terminal event only: launch or an idle agent never sends this callback.
  for attempt in range(6):
   try:
    req=urllib.request.Request('https://api2.cursor.sh/automations/webhook/4115c6ba-94e3-59b0-b7c6-628f693830cd',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=30) as response:write(P/'webhook-receipt.json',dict(status=response.status,time=now(),body=body,response=response.read(4096).decode(errors='replace')))
    break
   except Exception as e:
    write(P/'webhook-error.json',dict(attempt=attempt+1,error=str(e)));time.sleep(10)
if __name__=='__main__':run()
