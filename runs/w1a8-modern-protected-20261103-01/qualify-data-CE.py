from pathlib import Path
import json,copy,argparse,sys,signal,time
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
from modern_requests import Requests,atomic,pin
m=json.loads((R/'selected-lr-models-verbose.json').read_text());template=next(a for a in m['arms'] if a['kind']=='w1a8');m['arms']=[a for a in m['arms'] if not a['id'].startswith('w1a8_sign_lr')]
for name,root in [('oldCE',C/'w1a8-expanded-ce-control3840-01'),('newCE',C/'w1a8-expanded-ce-resident-new24-3840-01')]:
 d=json.loads((root/'result.json').read_text());arm=copy.deepcopy(template);arm.update(id='w1a8_'+name+'3840',model=d['model']);m['arms'].insert(-1,arm);m['source_pins'].append(pin(root/'result.json'))
m['data_usage']={'scope':'two14response qualification cells on reused12DEV/2FIT warmups; no repeatedspeed/freshclaim'};models=R/'data-CE-models-verbose.json';atomic(models,m)
x=Requests(argparse.Namespace(models=models,inputs=P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json',output=R/'data-CE-http-qualification01',repetitions=5,allow_cuda=True));x.stopping=False;x.records=[];signal.signal(signal.SIGTERM,x.request_stop);signal.signal(signal.SIGINT,x.request_stop);x.authenticate()
for arm in [a for a in x.arms if a['id'] in ['w1a8_oldCE3840','w1a8_newCE3840']]:x.cell(0,arm)
atomic(x.output/'result.json',{'status':'MATCHED_OLD_NEW_CE_HTTP_QUALIFIED_NOT_REPEATED_TIMING','records':x.records,'models':pin(models),'wall_seconds':time.perf_counter()-x.started})
