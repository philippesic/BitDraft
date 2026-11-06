"""Use the ordinary public vLLM API; no engine source modifications."""
from pathlib import Path
import dataclasses, json, os, sys, time

def main():
    p=Path(__file__).resolve().parent
    arm=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    assert arm in ('target_only','eagle3')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    target=str(p.parents[1]/'models/hf/Qwen3-4B')
    draft=str(p.parents[1]/'models/hf/Qwen3-4B_eagle3')
    tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
    rows=json.loads((p/'gate-prompts.json').read_text())
    prepared=[]
    for row in rows:
        text=row['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
        ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert len(ids)+128<=2048
        prepared.append({'id':row['id'],'prompt_token_ids':ids})
    (out/'canonical-prompts.json').write_text(json.dumps(prepared)+'\n')
    kwargs={'model':target,'dtype':'bfloat16','tensor_parallel_size':1,'max_model_len':2048,
        'max_num_seqs':1,'max_num_batched_tokens':2048,'gpu_memory_utilization':0.8,
        'enable_prefix_caching':False,'attention_backend':'TRITON_ATTN','disable_log_stats':False,
        'seed':42,'trust_remote_code':False}
    if arm=='eagle3':kwargs['speculative_config']={'model':draft,'method':'eagle3','num_speculative_tokens':3,'draft_sample_method':'greedy','rejection_sample_method':'standard'}
    (out/'kwargs.json').write_text(json.dumps(kwargs,indent=2)+'\n')
    begin=time.perf_counter();llm=LLM(**kwargs);startup=time.perf_counter()-begin
    (out/'loaded-library-maps.txt').write_text(Path('/proc/self/maps').read_text())
    params=SamplingParams(temperature=0,max_tokens=128,seed=42)
    warmup=0.;results=[]
    for i,row in enumerate(prepared[:2]+prepared):
        state=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']
        assert not state.get('pause_requested') and not state.get('stop_all_requested') and not (p/'STOP').exists()
        prompt={'prompt_token_ids':row['prompt_token_ids']}
        before=[dataclasses.asdict(x) for x in llm.get_metrics()]
        start=time.perf_counter();outputs=llm.generate([prompt],params,use_tqdm=False);wall=time.perf_counter()-start
        after=[dataclasses.asdict(x) for x in llm.get_metrics()]
        answer=outputs[0].outputs[0]
        assert list(outputs[0].prompt_token_ids)==row['prompt_token_ids']
        result={'arm':arm,'id':row['id'],'token_ids':list(answer.token_ids),'text':answer.text,'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,'full_generate_wall_seconds':wall,'metrics_before':before,'metrics_after':after}
        if i<2:warmup+=wall
        else:
            results.append(result)
            with (out/'requests.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            print(json.dumps({'arm':arm,'requests':len(results),'output_tokens':len(answer.token_ids)}),flush=True)
    (out/'summary.json').write_text(json.dumps({'completed':True,'requests':len(results),'startup_seconds':startup,'excluded_warmup_seconds':warmup,'evaluation_generate_seconds':sum(r['full_generate_wall_seconds'] for r in results),'training_seconds':0},indent=2)+'\n')

if __name__=='__main__':main()
