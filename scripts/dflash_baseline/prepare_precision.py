"""Freeze the finite tranche before screening; export zero-update native arms."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.prototype import resolve,digest,native_admit
from scripts.dflash_baseline.pipeline import sha
from scripts.dflash_baseline.checkpoint_io import load
from scripts.dflash_baseline.export_precision import export
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';parent=r/'foundation/checkpoint.pt.zst'
def main():
 base=json.load(open(r/'qualification-config.json'));base['parent']=dict(path=str(parent),sha256=sha(parent));state=load(parent);plans={};matched={}
 for g in (8,4,2,1):
  preset={8:'essential-a8',4:'gate-up-a4',2:'gate-up-a2',1:'gate-up-a1'}[g]
  for d in (8,4,2,1):
   name=f'G{g}D{d}';c=dict(base,preset=preset,overrides={f'blk.{i}.ffn_down':dict(weight='W1',bits=d) for i in range(6)});plans[name]=c
  anchor=f'G{g}D8';matched[anchor]=dict(family=f'G{g}',kind='anchor',control='G8D8')
 for g,d,kind,comp in [(8,4,'down',{}),(8,8,'attention-vo',{'attention_vo':'Q4_0'}),(4,4,'down',{}),(4,8,'learned-clip',{}),(2,2,'down',{}),(2,8,'fusion',{'fusion':'Q4_0'}),(1,1,'down',{}),(1,8,'private-head',{'head':'Q4_0'})]:
  anchor=f'G{g}D8';name=f'G{g}D{d}' if kind=='down' else anchor+'-'+kind
  if name not in plans:plans[name]=dict(plans[anchor],components=comp)
  if kind=='learned-clip':plans[name]['quantizer']=dict(mode='learned',clip=1.)
  matched[name]=dict(family=f'G{g}',kind=kind,control=anchor)
 # Cost/sensitivity-only probes of fixed protected formats; adaptation stays capped.
 for comp in ('fusion','attention_qk','attention_vo','head'):
  for format in ('Q8_0','Q4_0'):
   name='G8D8-probe-'+comp+'-'+format;plans[name]=dict(plans['G8D8'],components={comp:format})
 zero=r/'zero-arms';zero.mkdir(exist_ok=True);configs=r/'locked-configs';configs.mkdir(exist_ok=True);arms={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf')}
 resolved={name:resolve(c) for name,c in plans.items()}
 registry=dict(status='LOCKED_BEFORE_PRECISION_SCREEN',parent=base['parent'],matched_updates=1024,supervised_positions=15360,trainable='sign/scale/midpoint/shared-gain/down-gain/original-norm; no protected masters',data=base['data'],models={name:dict(request=plans[name],resolved_hash=digest(c),adapt=matched.get(name),native_contract=native_admit(c)) for name,c in resolved.items()},blocked=[dict(arm='private-embedding-reduction',reason='Target shared embedding is immutable; private native embedding support unqualified. No target change.'),dict(arm='packed-activation-binary-speed',reason='No new A1/A2 packed activation kernel qualified; actual execution remains INT8-code DP4A.')])
 existing=r/'tranche.json'
 if existing.exists() and json.load(open(existing))!=registry:raise ValueError('locked tranche changed')
 existing.write_text(json.dumps(registry,indent=2)+'\n')
 for name,c in resolved.items():
  (configs/(name+'.json')).write_text(json.dumps(plans[name],indent=2)+'\n');dest=zero/(name+'.gguf')
  if not dest.exists():export(parent,dest,config=c,state=state,checkpoint_pin=base['parent'])
  arms[name+'-zero']=str(dest);print('Zero export',name,flush=True)
 (r/'zero-arms.json').write_text(json.dumps(arms,indent=2)+'\n')
if __name__=='__main__':main()
