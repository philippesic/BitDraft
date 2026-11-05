"""Keep short/medium/near4096 cache comparisons in separate matched panels."""
import argparse,json,subprocess,sys
from .common import *
from .measure import completed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arms',nargs='+');a=ap.parse_args();registry=completed()
 models=dict(target=None,q4=str(Q4),B0=str(B/'model-final.gguf'),**{n:registry[n]['model'] for n in a.arms});cache={n:registry[n]['spec'].get('kv',['f16','f16']) for n in a.arms}
 save(R/'panels/final-context-arms.json',models);save(R/'panels/final-context-cache.json',cache)
 for length in (128,1536,3840):
  out=R/'measurements'/('final-context-'+str(length))
  if not (out/'result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.native','--arms',str(R/'panels/final-context-arms.json'),'--draft-cache-map',str(R/'panels/final-context-cache.json'),'--runtime',str(R/'runtime04'),'--out',str(out),'--cohort','capture3','--stress-length',str(length),'--reps','5','--warmups','2'],cwd=ROOT,check=True)
 print('FINAL_CONTEXT_COMPLETE',flush=True)

if __name__=='__main__':main()
