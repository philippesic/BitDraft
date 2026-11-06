from pathlib import Path
import sys,json,copy,argparse,signal,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01'
sys.path[:0]=[str(P/'scripts/design_readiness'),str(P/'scripts/w1a8_generalization')]
from native_panel import Panel
from modern_requests import atomic,pin

def read(p):return json.loads(Path(p).read_text())
assert read(R/'stage2-result.json')['status']=='THREE_EXTENSIONS_TRAINED_EXPORTED_NOT_NATIVE_ADMITTED'
f=read(R/'freeze.json');C=Path(f['output']);m=read(R/'models-stage1-quiet.json');m['arms']=[a for a in m['arms'] if a['id'] in ['target_only','dflash_Q4KM','w1a8_initial','paid5760','control6720']]
template=copy.deepcopy(next(a for a in m['arms'] if a['id']=='w1a8_initial'))
recipes=read(R/'stage2-selection.json')['recipes']
endpoints={name+'7680':C/(name+'7680') for name in recipes}
for name,path in endpoints.items():
 result=read(path/'result.json');a=copy.deepcopy(template);a['id']=name;a['model']=result['model'];assert pin(a['model']['path'])==a['model'];a['runtime']['log_verbosity']=4;a['runtime'].pop('loader_qualification',None);m['arms'].append(a)
m['source_pins'] += [pin(R/'stage2-result.json'),pin(R/'stage2-selection.json'),pin(__file__)]
verbose=R/'models-stage2-verbose.json';atomic(verbose,m)
inputs=P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json';out=R/'qualification-stage2'
x=Panel(argparse.Namespace(models=verbose,inputs=inputs,output=out,repetitions=5,allow_cuda=True));x.stopping=False;x.records=[];signal.signal(signal.SIGTERM,x.request_stop);signal.signal(signal.SIGINT,x.request_stop);x.authenticate()
for a in m['arms']:
 if a['kind']!='w1a8' or a['runtime'].get('log_verbosity')!=4:continue
 before=len(x.records);x.cell(0,a);records=x.records[before:];cell=out/('rep-00-'+a['id']);rel=read(cell/'release.json');assert rel['owned_process_reaped'] and rel['owned_process_group_absent'] and rel['returncode']==0
 assert len(records)==14 and all(row['token_text_stop_parity'] for row in records)
 q=dict(schema='w1a8_model_runtime_native_loader_qualification_v1',model=a['model'],runtime={k:v for k,v in a['runtime'].items() if k not in ['log_verbosity','loader_qualification']},native_outputs_audited=14,raw_log=pin(cell/'server.log'),release=pin(cell/'release.json'),command=pin(cell/'command.json'),maps=[pin(cell/'loaded-images-warm.json'),pin(cell/'loaded-images-final.json')],models=pin(verbose),responses=[r['response'] for r in records]);qp=R/(a['id']+'-qualification.json');atomic(qp,q);a['runtime']['log_verbosity']=3;a['runtime']['loader_qualification']=pin(qp)
atomic(out/'result.json',dict(status='THREE_ENDPOINTS_NATIVE_LOADER_AND_ROUND_METRICS_PASS',records=x.records,wall_seconds=time.perf_counter()-x.started,timing_claim=False))
quiet=R/'models-stage2-quiet.json';atomic(quiet,m)
Panel(argparse.Namespace(models=quiet,inputs=inputs,output=R/'serving-stage2',repetitions=5,allow_cuda=True)).run()
