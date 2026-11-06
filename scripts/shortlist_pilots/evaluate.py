"""Corrected-D5 full-request evaluation, using existing native request consumers.

The root's fresh sole-GPU guard owns runtime admission. This wrapper starts only
its own serial llama-server children, reaps them, and never trains or schedules.
Clean timing and optional round-cost profiling are separate invocations.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'runs/w1a8-algorithms-20261020'
BASE_TARGET = '05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6'
HELPERS = ('scripts/benchmark_native_eagle.py','scripts/benchmark_dspark_screen.py',
           'scripts/evaluate_nine_model_native.py','scripts/evaluate_paused_eagle_checkpoint.py',
           'src/w1a1_eagle/nine_model_pipeline.py')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def pin(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def read(path):return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def checked(ref):
    if sha(ref['path']) != ref['sha256']:raise ValueError('input pin changed: '+ref['path'])
    return Path(ref['path'])


def require(value,message):
    if not value:raise ValueError(message)


def validate_protocol(protocol):
    expected={'context_tokens':2048,'batch_tokens':32,'microbatch_tokens':32,
              'concurrency':1,'max_output_tokens':128,'seed':42,'temperature':0.,
              'enable_thinking':False,'warmups_per_cell':2,'repetitions':5}
    require(all(protocol.get(k)==v for k,v in expected.items()),'fixed request protocol changed')
    require(protocol['target_policy']=={'immutable':True,'kv':'f16','weights':'f16'},'F16 target/KV required')


def arm_order(arms, rep):
    # Existing precision/prefix runner's cyclic order: two arms alternate AB/BA.
    offset=rep%len(arms)
    return arms[offset:]+arms[:offset]


def select_prompts(prompts, limit, balanced=False):
    if not balanced:
        return prompts[:limit]
    domains=('code','prose','reasoning')
    require(limit % len(domains)==0,'balanced screen needs equal domain counts')
    selected=[]
    for domain in domains:
        rows=[p for p in prompts if p.get('domain')==domain]
        require(len(rows)>=limit//3,'balanced panel lacks domain prompts: '+domain)
        selected.extend(rows[:limit//3])
    require(len({p['id'] for p in selected})==limit,'duplicate balanced prompt IDs')
    return selected


def aggregate(rows):
    require(rows,'nonempty measured request rows required')
    results=[r['result'] for r in rows]
    for result in results:
        require(type(result['completion_tokens']) is int and result['completion_tokens']>0,
                'actual positive generated-token count required')
        require(result['completion_tokens']==len(result['generated_token_ids']),'token IDs/count mismatch')
        require(result['request_wall_s']>0,'positive full request wall required')
        require(all(type(result['speculative'][k]) is int and result['speculative'][k]>=0
                    for k in ('accepted','proposed','rounds')),'actual native counters required')
        require(result['speculative']['accepted']<=result['speculative']['proposed'],'native accepted exceeds proposed')
    tokens=sum(x['completion_tokens'] for x in results)
    seconds=sum(x['request_wall_s'] for x in results)
    counts={k:sum(x['speculative'][k] for x in results) for k in ('accepted','proposed','rounds')}
    return {'requests':len(results),'tokens':tokens,'full_request_seconds':seconds,
            'tokens_per_s':tokens/seconds,**counts,
            'acceptance_percent':100*counts['accepted']/counts['proposed'] if counts['proposed'] else None,
            'mean_accepted_per_round':counts['accepted']/counts['rounds'] if counts['rounds'] else None,
            'server_prompt_ms_sum':sum(x.get('server_prompt_ms') or 0 for x in results),
            'server_predicted_ms_sum':sum(x.get('server_predicted_ms') or 0 for x in results),
            'denominator':'sum actual output tokens / sum full request wall; loading and excluded warmups omitted'}


def summarize(records, models, repetitions, diagnostic=False):
    cells={}
    for arm in models:
        rows=[r for r in records if r['arm']==arm]
        cells[arm]={**aggregate(rows),
                    'by_domain':{d:aggregate([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})},
                    'by_repetition':{str(rep):aggregate([r for r in rows if r['repetition']==rep]) for rep in range(repetitions)}}
    comparisons={}
    for arm in models:
        if arm in ('q4_k_m','parent3504'):continue
        controls=['parent3504','q4_k_m']
        if models[arm].get('markov_mode','full')=='5' and 'q4_k_m_policy5' in cells and arm!='q4_k_m_policy5':
            controls.append('q4_k_m_policy5')
        comparisons[arm]={base:{'tokens_per_s_change_percent':100*(cells[arm]['tokens_per_s']/cells[base]['tokens_per_s']-1),
                              'acceptance_change_points':cells[arm]['acceptance_percent']-cells[base]['acceptance_percent']
                                  if cells[arm]['acceptance_percent'] is not None and cells[base]['acceptance_percent'] is not None else None,
                              'rounds_change_percent':100*(cells[arm]['rounds']/cells[base]['rounds']-1) if cells[base]['rounds'] else None}
                          for base in controls}
    return {'cells':cells,'comparisons':comparisons,'timing_eligible':not diagnostic,
            'scope':'separate diagnostic pass; traced wall rates cannot establish speed' if diagnostic else
                    'fixed24 panels x five alternating clean repetitions' if repetitions==5 else
                    'one-pass SCREEN; no small speed-gain claim',
            'strict_target_output':'all measured requests match frozen target IDs',
            'automatic_adoption':False}


def prepare(args):
    panel_paths=args.panel or [str(R/'openitems-markov-benchmark06-root/bindings-development.json'),
                              str(R/'openitems-markov-benchmark06-root/bindings-reserved.json')]
    panels={};source=None;plan=None;models=None;helpers=None
    for path in panel_paths:
        b=read(path)
        name='reserved' if 'reserved' in Path(path).stem else 'development'
        protocol=read(checked(b['protocol']));validate_protocol(protocol)
        if models is None:
            models={k:copy.deepcopy(b['models'][k]) for k in ('q4_k_m','parent3504')}
            source=b['source_root'];plan=read(checked(b['plan']))
            refs={x['path']:x for x in b['source_files']}
            helpers=[refs[str(Path(source)/n)] for n in HELPERS]
        require(b['models']['q4_k_m']['sha256']==models['q4_k_m']['sha256'] and
                b['models']['parent3504']['sha256']==models['parent3504']['sha256'],'panel baseline differs')
        panels[name]={'ancestry':pin(path),'protocol':b['protocol'],'prompts':protocol['primary_prompts'],
                      'target_reference':b['target_reference']}
    require(set(panels)=={'development','reserved'},'both existing fixed panels required')
    for text in args.candidate:
        name,path=text.split('=',1)
        require(name not in models and name.replace('_','').replace('-','').isalnum(),'unique safe candidate name required')
        models[name]={**pin(path),'custom_ffn':True,'draft_limit':5,'markov_mode':'full'}
    if args.arm_specs:
        for name,model in read(args.arm_specs).items():
            require(name not in models,'duplicate arm spec')
            models[name]=model
    if any(m.get('markov_mode')=='5' for m in models.values()) and not any(
            not m['custom_ffn'] and m.get('markov_mode')=='5' for m in models.values()):
        models['q4_k_m_policy5']={**copy.deepcopy(models['q4_k_m']),'markov_mode':'5'}
    build=read(args.build)
    config={'schema':'shortlist_fixed_request_evaluator_v1','build':pin(args.build),
            'binary':build['server'],'runtime_libraries':build['runtime_libraries'],
            'source_root':source,'source_files':helpers,'target':plan['target'],
            'gpu_uuid':plan['gpu_uuid'],'panels':panels,'models':models,
            'mode':args.mode,'prompt_limit':args.prompt_limit,'wall_seconds':args.wall_seconds,
            'balanced_screen':getattr(args,'balanced_screen',False),
            'port':18290,'target_width_mask':3,'repetitions':5 if args.mode=='claim' else 1,
            'warmups_per_cell':2,'draft_limit':5,
            'qualification':'root fresh boot/runtime/source/dispatch checks required externally; no old keeper reused',
            'environment':{'OMP_NUM_THREADS':'2'}}
    require(args.mode!='claim' or args.prompt_limit==24,'claim requires every frozen prompt')
    require(not config['balanced_screen'] or args.mode=='screen','balanced subset is a screen only')
    args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,config)


def load_config(path):
    b=read(path)
    require(b['schema']=='shortlist_fixed_request_evaluator_v1' and b['target_width_mask']==3 and
            b['draft_limit']==5 and b['warmups_per_cell']==2,'corrected D5 fixed protocol required')
    require(b['target']['sha256']==BASE_TARGET,'unchanged full-F16 target required')
    require(b['mode'] in ('screen','claim') and b['repetitions']==(5 if b['mode']=='claim' else 1),'mode/repetition mismatch')
    require(2<=b['prompt_limit']<=24 and (b['mode']!='claim' or b['prompt_limit']==24),'invalid prompt screen')
    require({'q4_k_m','parent3504'}<=b['models'].keys(),'matched standard Q4 and unchanged parent required')
    require(b['wall_seconds']>0 and 1024<=b['port']<=65535,'finite wall and valid loopback port required')
    for model in b['models'].values():
        require(model['draft_limit']==5 and type(model['custom_ffn']) is bool,'matched D5 models required')
        require(model.get('markov_mode','full') in ('full','5'),'unbound markov policy')
    for ref in [b['build'],b['binary'],b['target'],*b['source_files'],*b['runtime_libraries'],*b['models'].values()]:checked(ref)
    build=read(b['build']['path'])
    require(build['server']==b['binary'] and build['runtime_libraries']==b['runtime_libraries'],'runtime differs from bound build')
    sys.path[:0]=[str(Path(b['source_root'])/'scripts'),str(Path(b['source_root'])/'src')]
    return b


def environment(b, model, trace=None):
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','W1AX_','DSPARK_','DFLASH_','EAGLE_'))}
    env.update(b.get('environment',{}))
    env.update(CUDA_VISIBLE_DEVICES=b['gpu_uuid'],LD_LIBRARY_PATH=str(Path(b['binary']['path']).parent),
               DSPARK_REQUIRE_AUTHOR_LAYOUT='1',DSPARK_TARGET_WIDTH_MASK='3',
               DSPARK_MARKOV_ROWS=model.get('markov_mode','full'))
    if model['custom_ffn']:
        env.update(GGML_W1AX_DP4A='1',GGML_W1A8_ALGORITHM='imma_mix',GGML_W1A8_TOKEN7='1')
    # Optional comparable algorithm/head policy arms must state their settings.
    allowed={'GGML_W1A8_ALGORITHM','GGML_W1A8_TOKEN7','GGML_W1AX_GROUP_DP4A','GGML_W1AX_INPUT_GAINS'}
    for key,value in model.get('environment',{}).items():
        require(key in allowed,'unreviewed per-arm environment: '+key);env[key]=value
    if trace is not None:
        env['W1AX_ROUND_TRACE_JSONL']=str(trace)
    return env


def run(args):
    b=load_config(args.config)
    from benchmark_native_eagle import available_port,execute_request,request_body,wait_ready
    from benchmark_dspark_screen import stop_owned_server,round_summary
    from evaluate_nine_model_native import native_command
    from w1a1_eagle.nine_model_pipeline import load_opaque_prompts,stop_signals
    from evaluate_paused_eagle_checkpoint import wall_cap
    panels={}
    for name,panel in b['panels'].items():
        protocol=read(checked(panel['protocol']));validate_protocol(protocol)
        prompts=load_opaque_prompts(checked(panel['prompts']))
        require(len(prompts)==24 and all(p.get('split',protocol['split'])==protocol['split'] for p in prompts),'fixed24 prompt role differs')
        target=read(checked(panel['target_reference']))
        reference={(r['repetition'],r['prompt_id']):r['generated_token_ids'] for r in target['records']}
        prompts=select_prompts(prompts,b['prompt_limit'],b.get('balanced_screen',False))
        for rep in range(b['repetitions']):
            require(all((rep,p['id']) in reference for p in prompts),'target reference coverage missing')
        selected=copy.deepcopy(protocol);selected['draft_lengths']['dspark']=5
        panels[name]=(selected,prompts,reference)
    commands={}
    for name,(protocol,_,_) in panels.items():
        commands[name]={arm:native_command({'inputs':{'binary':b['binary'],'target':b['target']}},protocol,'dspark_a8',model,b['port'])
                        +['--log-verbosity',str(protocol['clean_log_verbosity'])] for arm,model in b['models'].items()}
    if args.preflight:
        print(json.dumps({'status':'PREFLIGHT_NO_GPU_LAUNCH','commands':commands,'configuration':pin(args.config),
                          'diagnostic_separate':args.diagnostic,'fresh_runtime_qualification_external':True},indent=2));return
    require(args.allow_native,'root sole operator must pass --allow-native')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'config.json',b)
    repetitions=1 if args.diagnostic else b['repetitions']
    records=[];summaries={}
    deadline=time.monotonic()+b['wall_seconds']
    def check_stop():
        require(time.monotonic()<deadline,'finite evaluator wall boundary')
        require(not args.stop.exists(),'evaluator STOP requested')
    with stop_signals(),wall_cap(b['wall_seconds']):
        for name,(protocol,prompts,reference) in panels.items():
            request_config={'evaluation':{'max_output_tokens':128,'temperature':0.,'seed':42,'enable_thinking':False}}
            for rep in range(repetitions):
                for arm in arm_order(list(b['models']),rep):
                    check_stop();require(available_port('127.0.0.1',b['port']),'evaluation port occupied')
                    cell=args.output/name/f'rep-{rep:02d}'/arm;cell.mkdir(parents=True)
                    trace=cell/'rounds.jsonl'
                    env=environment(b,b['models'][arm],trace.resolve() if args.diagnostic else None)
                    write(cell/'command.json',{'argv':commands[name][arm],'environment':{k:v for k,v in env.items() if k.startswith(('GGML_','W1AX_','DSPARK_','CUDA_','OMP_','LD_'))}})
                    proc=None
                    with (cell/'server.log').open('wb') as log:
                        try:
                            masked={signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGALRM}
                            prior=signal.pthread_sigmask(signal.SIG_BLOCK,masked)
                            try:
                                proc=subprocess.Popen(commands[name][arm],cwd=b['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                                      preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,prior))
                            finally:
                                signal.pthread_sigmask(signal.SIG_SETMASK,prior)
                            write(cell/'process.json',{'pid':proc.pid,'started_unix':time.time()})
                            wait_ready(proc,f"http://127.0.0.1:{b['port']}",min(protocol['startup_wall_seconds'],max(1,deadline-time.monotonic())))
                            mapped=Path(f'/proc/{proc.pid}/maps').read_text();(cell/'loaded-library-maps.txt').write_text(mapped)
                            for ref in b['runtime_libraries']:
                                filename=Path(ref['path']).name
                                if filename.startswith(('libllama','libggml','libmtmd')):
                                    require(str(Path(ref['path']).resolve()) in mapped,'loaded native library differs: '+ref['path'])
                            for i,prompt in enumerate(prompts[:2]):
                                check_stop();execute_request(f"http://127.0.0.1:{b['port']}",request_body(request_config,prompt),min(180,max(1,deadline-time.monotonic())),cell/f'warmup-{i:04d}')
                            offset=len(trace.read_text().splitlines()) if trace.exists() else 0
                            for i,prompt in enumerate(prompts):
                                check_stop()
                                result=execute_request(f"http://127.0.0.1:{b['port']}",request_body(request_config,prompt),min(180,max(1,deadline-time.monotonic())),cell/f'prompt-{i:04d}')
                                require(result['generated_token_ids']==reference[(rep,prompt['id'])],'strict target reference mismatch: '+arm+' '+prompt['id'])
                                row={'panel':name,'arm':arm,'repetition':rep,'prompt_id':prompt['id'],
                                     'domain':prompt.get('domain','unknown'),'result':result,'warmup_excluded':True}
                                if args.diagnostic:
                                    trace_rows=[json.loads(s) for s in trace.read_text().splitlines()]
                                    selected=trace_rows[offset:];offset=len(trace_rows)
                                    emitted=[t for x in selected if x['status']!='checkpoint_replay' for t in x['emitted_token_ids']]
                                    require(emitted in (result['generated_token_ids'],result['generated_token_ids'][1:]),'diagnostic trace/request token mapping differs')
                                    row['round_costs']=round_summary(selected,5)
                                records.append(row)
                                with (args.output/'requests.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                        finally:
                            if proc is not None:
                                stop_owned_server(proc)
                                write(cell/'release.json',{'owned_server_reaped':proc.poll() is not None,'returncode':proc.returncode})
                    if not args.diagnostic:
                        text=(cell/'server.log').read_text(errors='replace')
                        require(not any(marker in text for marker in ('WIDTH_DIAG_','TARGET_WIDTH_NODE','TARGET_WIDTH_POLICY','W1AX_ADMISSION_TRACE')),'diagnostic output contaminated timing')
            panel_rows=[r for r in records if r['panel']==name]
            summaries[name]=summarize(panel_rows,b['models'],repetitions,args.diagnostic)
    write(args.output/'summary.json',{'status':'COMPLETE','panels':summaries,'diagnostic':args.diagnostic,
          'configuration':pin(args.config),'strict_target_mismatches':0,'warmups_per_cell':2,
          'root_runtime_release_required':True,'profile_costs_in':'requests.jsonl round_costs' if args.diagnostic else None,
          'fresh_model_kernel_qualification':'external root responsibility; request token exactness is not hidden fidelity'})


def main():
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--build',type=Path,default=R/'residual30-build04/complete.json')
    prep.add_argument('--panel',action='append',help='existing corrected benchmark binding; supply development and reserved')
    prep.add_argument('--candidate',action='append',default=[],help='name=/absolute/exported.gguf; custom FFN full7 default')
    prep.add_argument('--arm-specs',type=Path,help='optional extra model pins/explicit comparable policy')
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--mode',choices=('screen','claim'),default='screen')
    prep.add_argument('--prompt-limit',type=int,default=24)
    prep.add_argument('--balanced-screen',action='store_true',help='equal code/prose/reasoning subset for brief screening')
    prep.add_argument('--wall-seconds',type=float,default=1200)
    execute=sub.add_parser('run')
    execute.add_argument('--config',type=Path,required=True)
    execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--stop',type=Path,default=R/'STOP')
    execute.add_argument('--preflight',action='store_true')
    execute.add_argument('--diagnostic',action='store_true')
    execute.add_argument('--allow-native',action='store_true')
    args=parser.parse_args()
    prepare(args) if args.command=='prepare' else run(args)


if __name__=='__main__':main()
