"""Finish the two reserved interactions and two matched one-window continuations."""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.precision_benchmark import pin,save
from scripts.dflash_baseline.prototype import resolve,digest,native_admit
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.checkpoint_io import archive
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';runtime=P/'runs/precision-native-20261104-04'
def prepare():
 plans=[]
 for name,base,components in [('G8D8-learned-clip','G8D8',{}),('G4D8-learned-small','G4D8',{'fusion':'Q4_0','attention_qk':'Q4_0','attention_vo':'Q4_0'})]:
  c=copy.deepcopy(json.load(open(R/'locked-configs'/(base+'.json'))));c['quantizer']=dict(mode='learned',clip=1.);c['components']=components
  native_admit(resolve(c));config=R/'locked-configs'/(name+'.json')
  if config.exists() and json.load(open(config))!=c:raise ValueError('interaction config changed')
  save(config,c);plans.append(dict(name=name,kind='interaction',out=str(R/'interactions'/name),config=pin(config),updates=1024,max_windows=1024,history='Direct common frozen foundation; reset Adam; same TRAIN order/cursor as all standard arms'))
 for name in ('G1D8','G8D8'):
  plans.append(dict(name=name,kind='recovery',out=str(R/'extensions'/name),config=pin(R/'locked-configs'/(name+'.json')),resume=str(R/'adapted'/name/'checkpoint.pt.bsz'),updates=2048,max_windows=1024,history='Exact optimizer/RNG/data continuation; unchanged configuration; additional1024 updates; matched candidate and A8 control'))
 record=dict(status='RESERVED_INITIAL_ALLOCATION_LOCKED',arms=plans,additional_physical_updates=4096,interactions=2,recovery_windows=2,rationale='G4 learned clipping improves acceptance but precision and bound-policy effects are confounded: test learned A8 control. Test its smaller combined protected-weight interaction. G1 dynamic acceptance is still improving: extend it with an identical-exposure A8 control. No new feature tier. No heldout opened.',unselected='Calibrated A1/A2 clipping and fixed-vs-learned ablations remain untested initial variants; defer until full initial chart. No general A1/A2 inferiority claim.')
 path=R/'reserved-allocation.json'
 if path.exists() and json.load(open(path))!=record:raise ValueError('reserved allocation changed')
 save(path,record);return record
def run(args):subprocess.run(list(map(str,args)),env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src')),cwd=ROOT,check=True)
def execute():
 plan=json.load(open(R/'reserved-allocation.json'))
 registry=json.load(open(R/'tranche.json'))
 for name,spec in registry['models'].items():
  if spec['adapt'] and not (R/'adapted'/name/'panel/result.json').exists():raise ValueError('initial standard arm unfinished: '+name)
 # Exercise actual same-process warmups before the expensive final protocol.
 smoke=R/'warmup-protocol-smoke';af=R/'warmup-smoke-arms.json'
 save(af,{'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),'foundation-zero':str(R/'zero-arms/G8D8.gguf')})
 if not (smoke/'result.json').exists():run([sys.executable,ROOT/'scripts/dflash_baseline/precision_benchmark.py','--arms',af,'--runtime',runtime,'--out',smoke,'--cohort','capture3','--warmups','2','--paired-trace'])
 for arm in plan['arms']:
  cfg=Path(arm['config']['path']);out=Path(arm['out']);out.parent.mkdir(exist_ok=True)
  if pin(cfg)['sha256']!=arm['config']['sha256']:raise ValueError('locked config changed')
  if not (out/'result.json').exists():
   args=[sys.executable,'-m','scripts.dflash_baseline.pipeline',cfg,'--out',out,'--updates',arm['updates'],'--max-windows',arm['max_windows'],'--device','cuda','--native-points']
   args+=['0','256','--adapt'] if arm['kind']=='interaction' else ['1536','--resume',arm['resume']]
   run(args)
  result=json.load(open(out/'result.json'))
  if result['status']!='COMPLETE' or result['progress']['updates']!=arm['updates']:raise ValueError('reserved arm incomplete')
  if not (out/'model-final.gguf').exists():export(out/'checkpoint.pt',out/'model-final.gguf')
  if not (out/'checkpoint.pt.bsz').exists():archive(out/'checkpoint.pt')
  models={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf')}
  if arm['kind']=='interaction':models.update({arm['name']+'-'+label:str(out/file) for label,file in [('zero','model-0.gguf'),('256','model-256.gguf'),('1024','model-final.gguf')]})
  else:models.update({arm['name']+'-'+label:str(file) for label,file in [('1024',R/'adapted'/arm['name']/'model-final.gguf'),('1536',out/'model-1536.gguf'),('2048',out/'model-final.gguf')]})
  af=out/'panel-arms.json';save(af,models)
  if not (out/'panel/result.json').exists():run([sys.executable,ROOT/'scripts/dflash_baseline/precision_benchmark.py','--arms',af,'--runtime',runtime,'--out',out/'panel','--cohort','natural12','--paired-trace'])
  print('Reserved native arm complete',arm['name'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args();prepare() if a.mode=='prepare' else execute()
