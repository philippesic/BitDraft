from pathlib import Path
import json,os,sys,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,pin
R=P/'runs/execution-campaign-20261105-01';S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';lib=R/'libproposal-cost-trace.so'
cmd=['c++','-O2','-shared','-fPIC','-std=c++17','-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'-I/usr/local/cuda/include',str(ROOT/'scripts/execution_campaign/proposal_trace.cpp'),'-ldl','-pthread','-o',str(lib)]
subprocess.run(cmd,check=True);(R/'proposal-trace-build.json').write_text(json.dumps(dict(command=cmd,source=pin(ROOT/'scripts/execution_campaign/proposal_trace.cpp'),library=pin(lib)),indent=2)+'\n')
arms=json.load(open(R/'arms.json'));a={'target':None,'w1_original':arms['w1'],'w1_optimized':arms['w1'],'q4':arms['q4'],'q4_optimized':arms['q4']};(R/'final-arms.json').write_text(json.dumps(a,indent=2)+'\n');runtime=R/'runtime-combined-last02';mapping={x:str(runtime) for x in ['w1_optimized','q4_optimized']};(R/'final-runtime-map.json').write_text(json.dumps(mapping,indent=2)+'\n');env={**os.environ,'EXEC_TRACE_LIBRARY':str(lib)}
argv=[sys.executable,str(ROOT/'scripts/execution_campaign/native.py'),'--arms',str(R/'final-arms.json'),'--runtime',str(F),'--runtime-map',str(R/'final-runtime-map.json'),'--out',str(R/'native-final'),'--cohort','natural12','--warmups','2','--reps','5','--paired-trace']
subprocess.run(argv,env=env,check=True)
# Independent short instrumented attribution, after clean final throughput requests.
argv=[sys.executable,str(ROOT/'scripts/execution_campaign/native.py'),'--arms',str(R/'final-arms.json'),'--runtime',str(F),'--runtime-map',str(R/'final-runtime-map.json'),'--out',str(R/'profile-final'),'--cohort','natural12','--limit','1','--output-cap','32','--warmups','2','--reps','1','--profile','--trace']
subprocess.run(argv,env=env,check=True)
