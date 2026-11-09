import sys,json,pathlib
P=pathlib.Path('/home/philip/he4-capture-2080ti');sys.path.insert(0,str(P/'project/scripts/he4_capture'));import run as r
row=r.broad(3);row['rendered_text']='<|im_start|>system\nYou are a helpful assistant. Answer directly and accurately.<|im_end|>\n<|im_start|>user\nReply with the single word hello.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
e=P/'smoke';e.mkdir();r.write(e/'source.json',row);(e/'prompt.txt').write_text(row['rendered_text'])
r.command([r.BIN,'--tokenize','--model',r.MODEL,'--text-file',e/'prompt.txt','--out',e/'tokenization'],e/'tokenization.log')
a=[r.BIN,'--model',r.MODEL,'--tokens',e/'tokenization/tokens.ids','--cpu-library',r.LIB/'libggml-cpu.so.0.28.0','--cuda-library',r.CUDA,'--greedy-steps','128','--taps',','.join(map(str,r.TAPS)),'--prefill-chunk','32','--output-cap','32','--max-output-mib','1024']
r.write(e/'launch.json',dict(argv=list(map(str,a))))
r.command(a+['--out',e/'capture'],e/'capture.log');print(r.verify(e/'capture',row),flush=True)
r.command(a+['--out',e/'no-features','--no-features'],e/'no-features.log')
c=json.loads((e/'capture/capture.json').read_text());d=json.loads((e/'no-features/capture.json').read_text());assert c['greedy_ids']==d['greedy_ids'];assert c['greedy_ids'][-1] in c['eog_ids'];assert not any(x in c['eog_ids'] for x in c['input_tokens'][c['prompt_tokens']:]);(e/'no-features/logits.f32').unlink();r.write(e/'parity.json',dict(taps_disabled_greedy_equal=True,early_eog_stop_proven=True,label_count=len(c['greedy_ids']),full_logits_deleted=True));print('SMOKE PASS',flush=True)
