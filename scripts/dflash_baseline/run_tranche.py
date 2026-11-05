"""Serial GPU ownership for finite zero screens and common-parent matched arms."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.checkpoint_io import archive
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';runtime=P/'runs/precision-native-20261104-03';env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'))
def run(argv):subprocess.run(list(map(str,argv)),env=env,check=True)
def benchmark(arms,out,cohort='capture3',paired=True):
 args=[sys.executable,ROOT/'scripts/dflash_baseline/precision_benchmark.py','--arms',arms,'--runtime',runtime,'--out',out,'--cohort',cohort]
 if paired:args+=['--paired-trace']
 run(args)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['zero','anchors','mutations']);a=ap.parse_args()
 assert json.load(open(r/'foundation-panel/result.json'))['status']=='COMPLETE'
 if a.stage=='zero':
  # Requalify every width/mode against the final common libraries, not the older build.
  with (r/'final-runtime-ops.jsonl').open('w') as output:subprocess.run([str(P/'runs/precision-native-20261104-01/op-check'),str(r/'final-runtime-ops-maps.txt')],env={**env,'LD_LIBRARY_PATH':str(runtime/'bin')},stdout=output,check=True)
  benchmark(r/'zero-arms.json',r/'zero-panel',paired=True);return
 registry=json.load(open(r/'tranche.json'));chosen=[(name,c) for name,c in registry['models'].items() if c['adapt'] and ((c['adapt']['kind']=='anchor')==(a.stage=='anchors'))]
 for name,spec in chosen:
  cfg=r/'locked-configs'/(name+'.json');out=r/'adapted'/name;out.parent.mkdir(exist_ok=True)
  if not (out/'result.json').exists():
   run([sys.executable,'-m','scripts.dflash_baseline.pipeline',cfg,'--out',out,'--updates','1024','--max-windows','1024','--device','cuda','--adapt','--native-points','256'])
  record=json.load(open(out/'result.json'))
  if record['status']!='COMPLETE' or record['progress']['updates']!=1024:raise RuntimeError('incomplete arm '+name)
  if not (out/'model-final.gguf').exists():export(out/'checkpoint.pt',out/'model-final.gguf')
  if not (out/'checkpoint.pt.bsz').exists():archive(out/'checkpoint.pt')
  arms={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),name+'-256':str(out/'model-256.gguf'),name+'-1024':str(out/'model-final.gguf')}
  af=out/'panel-arms.json';af.write_text(json.dumps(arms,indent=2)+'\n')
  if not (out/'panel/result.json').exists():benchmark(af,out/'panel',cohort='natural12',paired=True)
  print('Matched native arm complete',name,flush=True)
if __name__=='__main__':main()
