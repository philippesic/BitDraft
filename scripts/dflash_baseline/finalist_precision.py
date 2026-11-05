"""Lock four family recipes, then honor reserved matched2048-update confirmation."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.precision_benchmark import pin,save
from scripts.dflash_baseline.prototype import digest
from scripts.dflash_baseline.checkpoint_io import archive as checkpoint_archive
from scripts.dflash_baseline.native_archive import archive as native_archive,dictionary,available
from scripts.dflash_baseline.export_precision import export
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01'
def prepare(names):
 if len(names)!=4:raise ValueError('four family confirmations were reserved')
 if json.load(open(P/'runs/precision-reserved-20261105-01/state.json')).get('exit_code')!=0:raise ValueError('finite search must finish before locking confirmation')
 registry=json.load(open(R/'tranche.json'));parent=registry['parent'];data=registry['data'];arms={};families=[]
 for name in names:
  source=next((R/folder/name for folder in ('adapted','interactions') if (R/folder/name/'result.json').exists()),None)
  if source is None:raise ValueError('unknown completed standard recipe '+name)
  cfg=json.load(open(source/'resolved.json'));family=cfg['coverage']['blk.0.ffn_gate']['bits'];families.append(family)
  if cfg['parent']!=parent or cfg['data']!=data:raise ValueError('confirmation must share frozen foundation/data')
  prior=json.load(open(source/'result.json'))
  if prior['status']!='COMPLETE' or prior['progress']['updates']!=1024:raise ValueError('incomplete standard exposure')
  continuation=R/'extensions'/name
  reuse=(continuation/'result.json').exists() and json.load(open(continuation/'result.json'))['progress']['updates']==2048 and digest(json.load(open(continuation/'resolved.json')))==digest(cfg)
  out=continuation if reuse else R/'finalists'/name
  arms[name]=dict(config=pin(R/'locked-configs'/(name+'.json')),resolved_hash=digest(cfg),source_result=pin(source/'result.json'),resume=pin(source/'checkpoint.pt.bsz'),out=str(out),reuse_exact_matching_2048=reuse,additional_updates=0 if reuse else 1024)
 if sorted(families)!=[1,2,4,8]:raise ValueError('one predeclared recipe per activation family')
 plan=dict(status='FINALIST_CONFIRMATION_CONFIGS_LOCKED',parent=parent,data=data,updates=2048,supervised_positions=30720,arms=arms,physical_additional_updates=sum(x['additional_updates'] for x in arms.values()),policy='Exact constant-schedule resume from1024 to2048 equals uninterrupted common-parent2048 training. Same TRAIN order/cursor/exposure; initial Adam reset preserved. No new configuration or heldout input after this lock.',attribution='Four dynamic1024 anchors isolate logical gate/up precision. These family finalists are explicit recipes when clipping/protected formats differ; no sole-precision claim.',rationale='Preserve all four activation families. Learned A8 is the clipping-policy control; compact learned A4 tests the practical size frontier; dynamic A2 and recovering A1 retain bounded extreme-precision confirmation. Freeze before confirmation, never tune on heldout.')
 path=R/'finalist-training-plan.json'
 if path.exists() and json.load(open(path))!=plan:raise ValueError('finalist training selection changed')
 save(path,plan)
def run(args):subprocess.run(list(map(str,args)),cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src')),check=True)
def execute():
 if json.load(open(P/'runs/precision-archive-20261105-01/state.json')).get('exit_code')!=0:raise ValueError('lossless archival/native admission must qualify first')
 plan=json.load(open(R/'finalist-training-plan.json'));records={};basis=R/'zero-arms/G8D8.gguf'
 for name,arm in plan['arms'].items():
  cfg=arm['config'];resume=arm['resume'];out=Path(arm['out']);out.parent.mkdir(exist_ok=True)
  if pin(cfg['path'])!=cfg or pin(resume['path'])!=resume:raise ValueError('locked finalist config/checkpoint changed')
  if not arm['reuse_exact_matching_2048']:
   if not (out/'result.json').exists():run([sys.executable,'-m','scripts.dflash_baseline.pipeline',cfg['path'],'--out',out,'--updates','2048','--max-windows','1024','--resume',resume['path'],'--device','cuda','--native-points','1536'])
   result=json.load(open(out/'result.json'))
   if result['status']!='COMPLETE' or result['progress']['updates']!=2048:raise ValueError('confirmation interrupted')
   if not available(out/'model-final.gguf'):export(out/'checkpoint.pt',out/'model-final.gguf')
   if not (out/'checkpoint.pt.bsz').exists():checkpoint_archive(out/'checkpoint.pt')
   d=dictionary(basis)
   for model in sorted(out.glob('*.gguf')):native_archive(model,basis,prepared=d)
   del d
  result=json.load(open(out/'result.json'));resolved=json.load(open(out/'resolved.json'));progress=result['progress'];foundation=json.load(open(R/'foundation/result.json'))['progress']
  if digest(resolved)!=arm['resolved_hash'] or progress['cursor']!=foundation['cursor']+2048 or progress['supervised_tokens']!=30720 or progress['order']!=foundation['order']:raise ValueError('final matched history mismatch')
  records[name]=dict(model=pin(out/'model-final.gguf'),checkpoint=pin(out/'checkpoint.pt.bsz'),result=pin(out/'result.json'),updates=2048,supervised_positions=30720,cursor=progress['cursor'],native_evaluation='required in locked repeated confirmation, not inferred from training')
  print('Finalist2048 exposure complete',name,flush=True)
 additional={}
 for name in ('G8D8-learned-clip','G4D8-learned-small'):
  out=R/'interactions'/name
  for label,file in [('zero','model-0.gguf'),('256','model-256.gguf'),('1024','model-final.gguf')]:additional[name+'-'+label]=str(out/file)
 for name in ('G1D8','G8D8'):
  for label,file in [('1536','model-1536.gguf'),('2048','model-final.gguf')]:additional[name+'-'+label]=str(R/'extensions'/name/file)
 for name,arm in plan['arms'].items():
  for label,file in [('1536','model-1536.gguf'),('2048','model-final.gguf')]:additional[name+'-'+label]=str(Path(arm['out'])/file)
 save(R/'finalist-training-confirmation.json',dict(status='FOUR_MATCHED_2048_RECIPES_COMPLETE',plan=pin(R/'finalist-training-plan.json'),records=records))
 save(R/'search-lock.json',dict(status='INITIAL_SEARCH_LOCKED_FOR_CONFIRMATION',tranche=pin(R/'tranche.json'),reserved=pin(R/'reserved-allocation.json'),finalist_plan=pin(R/'finalist-training-plan.json'),finalist_training=pin(R/'finalist-training-confirmation.json'),additional_confirmations=additional,policy='All configurations, training histories and repeated DEV arms locked before formal confirmation. Shortlist model pins will lock before any fresh heldout payload is selected.'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('names',nargs='*');a=ap.parse_args();prepare(a.names) if a.mode=='prepare' else execute()
