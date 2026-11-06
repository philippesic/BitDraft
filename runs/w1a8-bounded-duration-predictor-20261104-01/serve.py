from pathlib import Path
import sys,json,copy,argparse,signal,time,hashlib
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');N=P/'runs/w1a8-bounded-duration-predictor-20261104-01';sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
from modern_requests import Requests,atomic,pin

def read(p):return json.loads(Path(p).read_text())
training=read(N/'result.json');assert training['status']=='OLD_ONLY5760_TRAINED_OLD3840_REPRODUCTION_PASS_NOT_NATIVE_ADMITTED'
assert read(N/'old3840-parity.json')['all_exact']
for key in ['model','checkpoint','training_result','parity']:
 assert pin(training[key]['path'])==training[key]
m=read(N/'models-pre-endpoint.json');arm=copy.deepcopy(next(a for a in m['arms'] if a['id']=='expanded5760'));arm['id']='old5760';arm['model']=training['model'];arm['runtime']['log_verbosity']=4;arm['runtime'].pop('loader_qualification',None);m['arms'].append(arm);m['source_pins'] += [pin(N/'frozen-study.json'),pin(N/'result.json'),pin(N/'old3840-parity.json'),pin(__file__)];verbose=N/'models-new-endpoint-verbose.json';atomic(verbose,m)
inputs=P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json';out=N/'old5760-http-qualification01';x=Requests(argparse.Namespace(models=verbose,inputs=inputs,output=out,repetitions=5,allow_cuda=True));x.stopping=False;x.records=[];signal.signal(signal.SIGTERM,x.request_stop);signal.signal(signal.SIGINT,x.request_stop);x.authenticate();x.cell(0,arm);atomic(out/'result.json',dict(status='OLD5760_NATIVE_QUALIFIED_NO_TIMING_CLAIM',records=x.records,models=pin(verbose),wall_seconds=time.perf_counter()-x.started))
cell=out/'rep-00-old5760';release=read(cell/'release.json');assert release['owned_process_reaped'] and release['owned_process_group_absent'] and release['returncode']==0
assert len(x.records)==14 and all(r['token_text_stop_parity'] for r in x.records)
q=dict(schema='w1a8_model_runtime_native_loader_qualification_v1',model=arm['model'],runtime={k:v for k,v in arm['runtime'].items() if k not in ['log_verbosity','loader_qualification']},native_outputs_audited=14,raw_log=pin(cell/'server.log'),release=pin(cell/'release.json'),command=pin(cell/'command.json'),maps=[pin(cell/'loaded-images-warm.json'),pin(cell/'loaded-images-final.json')],models=pin(verbose),responses=[r['response'] for r in x.records]);qp=N/'old5760-qualification.json';atomic(qp,q);arm['runtime']['log_verbosity']=3;arm['runtime']['loader_qualification']=pin(qp);quiet=N/'models-quiet.json';atomic(quiet,m)
x=Requests(argparse.Namespace(models=quiet,inputs=inputs,output=N/'repeated-serving01',repetitions=5,allow_cuda=True));x.run()
