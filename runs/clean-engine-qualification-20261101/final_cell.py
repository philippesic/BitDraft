"""Public vLLM API, ordinary HF checkpoints, no engine patches."""
from pathlib import Path
import dataclasses, hashlib, json, os, sys, time

def main():
    p=Path(__file__).resolve().parent
    mode,arm,rep,output=sys.argv[1:];rep=int(rep);out=Path(output);out.mkdir(exist_ok=False)
    assert mode in ('stock','invariant') and arm in ('target_only','eagle3') and rep in range(5)
    assert os.environ['VLLM_BATCH_INVARIANT']==('1' if mode=='invariant' else '0')
    freeze=json.loads((p/'final-freeze.json').read_text())
    for ref in freeze['sources']:assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
    receipt=json.loads((p/'final-inputs/receipt.json').read_text())
    assert receipt['freeze_sha256']==hashlib.sha256((p/'final-freeze.json').read_bytes()).hexdigest()
    for name,sha in receipt['files'].items():assert hashlib.sha256((p/'final-inputs'/name).read_bytes()).hexdigest()==sha
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0,str(p.parent/'upstream-consistency-20261101'));import quality_judge
    tokenizer=AutoTokenizer.from_pretrained(freeze['target'],local_files_only=True)
    prompts=json.loads((p/'final-inputs/requests.json').read_text());cases=json.loads((p/'final-inputs/quality-cases.json').read_text())
    assert len(prompts)==len(cases)==24
    prepared=[]
    for phase,items in [('quality',cases if rep==0 else []),('performance',prompts)]:
        for row in items:
            text=row['prompt'] if phase=='quality' else row['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
            ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            maximum=384 if phase=='quality' else 128;assert len(ids)+maximum<=2048
            prepared.append({'phase':phase,'id':row['id'],'prompt_token_ids':ids,'maximum':maximum,'case':row if phase=='quality' else None})
    (out/'canonical-prompts.json').write_text(json.dumps(prepared)+'\n')
    kwargs=freeze['engine_kwargs'].copy()
    if arm=='eagle3':kwargs['speculative_config']=freeze['speculative_config']
    (out/'kwargs.json').write_text(json.dumps(kwargs,indent=2)+'\n')
    begin=time.perf_counter();llm=LLM(**kwargs);startup=time.perf_counter()-begin
    (out/'loaded-library-maps.txt').write_text(Path('/proc/self/maps').read_text())
    results=[];warmup=0.
    performance=[r for r in prepared if r['phase']=='performance']
    work=[{**r,'phase':'warmup'} for r in performance[:2]]+(prepared if rep%2==0 else list(reversed(prepared)))
    for row in work:
        state=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']
        assert not state.get('pause_requested') and not state.get('stop_all_requested') and not (p/'STOP').exists()
        params=SamplingParams(temperature=0,max_tokens=row['maximum'],seed=42)
        before=[dataclasses.asdict(x) for x in llm.get_metrics()]
        start=time.perf_counter();outputs=llm.generate([{'prompt_token_ids':row['prompt_token_ids']}],params,use_tqdm=False);wall=time.perf_counter()-start
        after=[dataclasses.asdict(x) for x in llm.get_metrics()]
        answer=outputs[0].outputs[0];assert list(outputs[0].prompt_token_ids)==row['prompt_token_ids']
        assert answer.token_ids and len(answer.token_ids)<=row['maximum']
        result={'mode':mode,'arm':arm,'rep':rep,'phase':row['phase'],'id':row['id'],'token_ids':list(answer.token_ids),'text':answer.text,
            'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,'full_generate_wall_seconds':wall,'metrics_before':before,'metrics_after':after}
        if row['phase']=='warmup':warmup+=wall;continue
        if row['phase']=='quality':result['judgement']=quality_judge.judge(row['case'],answer.text)
        results.append(result)
        with (out/'requests.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    (out/'summary.json').write_text(json.dumps({'completed':True,'requests':len(results),'startup_seconds':startup,'excluded_warmup_seconds':warmup,
        'evaluation_generate_seconds':sum(r['full_generate_wall_seconds'] for r in results),'training_seconds':0},indent=2)+'\n')

if __name__=='__main__':main()
