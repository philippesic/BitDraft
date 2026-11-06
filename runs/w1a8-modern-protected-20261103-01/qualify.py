from pathlib import Path
import argparse,sys,json,signal,time
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01'
sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
from modern_requests import Requests,atomic,pin
x=Requests(argparse.Namespace(models=R/'models-qualification.json',inputs=P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json',output=R/'qualification01',repetitions=5,allow_cuda=True));x.stopping=False;x.records=[]
signal.signal(signal.SIGTERM,x.request_stop);signal.signal(signal.SIGINT,x.request_stop)
x.authenticate()
for arm in x.arms[3:]:x.cell(0,arm)
atomic(x.output/'result.json',{'status':'THREE_NATIVE_MODELS_QUALIFIED_NOT_REPEATED_TIMING','records':x.records,'models':pin(R/'models-qualification.json'),'wall_seconds':time.perf_counter()-x.started})
