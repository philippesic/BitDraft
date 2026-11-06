from pathlib import Path
import subprocess,json,sys,shutil
P=Path(__file__).resolve().parent;PY=str(P.parent.parent/'.venv/bin/python');rt=P/'runtime';m=P/'models';results=[]
def run(name,cmd):
 with (P/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=rt,stdout=f,stderr=subprocess.STDOUT)
 results.append({'name':name,'argv':cmd,'returncode':r.returncode});(P/'conversion-results.json').write_text(json.dumps(results,indent=2));return r.returncode==0
for size in ['4B','9B']:
 for typ in ['eagle','dspark']:
  src=m/f'hf-{typ}-{size}';target=m/f'hf-target-{size}'
  if not src.exists():continue
  if typ=='eagle':
   c=json.loads((src/'config.json').read_text());shutil.copyfile(src/'config.json',src/'config.source.json')
   # Normalize the published nested metadata to the converter's documented key, retaining source.
   ids=c.get('eagle_config',{}).get('eagle_aux_hidden_state_layer_ids')
   if ids:c['eagle_aux_hidden_state_layer_ids']=ids
   (src/'config.json').write_text(json.dumps(c,indent=2))
  prefix=f'Qwen3.5-{size}-'+('eagle3' if typ=='eagle' else 'dspark')
  bf=m/(prefix+'-BF16.gguf')
  ok=run(prefix+'-convert',[PY,str(rt/'convert_hf_to_gguf.py'),str(src),'--target-model-dir',str(target),'--outfile',str(bf),'--outtype','bf16'])
  if ok:
   for fmt in ['Q8_0','Q4_K_M']:run(prefix+'-'+fmt,[str(P/'build/bin/llama-quantize'),str(bf),str(m/(prefix+'-'+fmt+'.gguf')),fmt,'8'])
