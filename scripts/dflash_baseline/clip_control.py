"""One post-publication fixed/learned range control; frozen initial charts remain intact."""
import argparse,copy,json,os,shutil,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.precision_benchmark import pin,save,pause,auth
from scripts.dflash_baseline.prototype import resolve,digest,native_admit
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.native_archive import archive,dictionary,available
from scripts.dflash_baseline.prepare_holdout import CONTEXT_POLICY
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';U=R/'clip-control-followup';runtime=P/'runs/precision-native-20261104-04';stopping=False

def stop(*_):
 global stopping;stopping=True

def run(argv,extra=None):
 pause()
 if stopping:raise RuntimeError('supervisor requested stop')
 subprocess.run(list(map(str,argv)),cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'),**(extra or {})),check=True)
 if stopping:raise RuntimeError('supervisor requested stop after child recovery')

def script(name,*args):run([sys.executable,ROOT/'scripts/dflash_baseline'/name,*args])
def panel(arms,out,cohort='natural12',reps=1,prompts=None,label=None):
 af=U/(out.name+'-arms.json');save(af,arms)
 if (out/'result.json').exists():return
 argv=[sys.executable,ROOT/'scripts/dflash_baseline/precision_benchmark.py','--arms',af,'--runtime',runtime,'--out',out,'--cohort',cohort,'--reps',reps,'--warmups',2,'--paired-trace']
 if prompts:argv+=['--prompts',prompts,'--cohort-label',label]
 run(argv)

def prepare():
 initial=R/'campaign-completion.json'
 if json.load(open(initial))['status']!='FULL_INITIAL_COMPARISON_COMPLETE':raise ValueError('initial charts must publish before further allocation')
 U.mkdir(exist_ok=False);c=copy.deepcopy(json.load(open(R/'locked-configs/G4D8-learned-small.json')));c['quantizer']['mode']='fixed';resolved=resolve(c);native_admit(resolved)
 learned=json.load(open(R/'interactions/G4D8-learned-small/resolved.json'));expected=copy.deepcopy(learned);expected['quantizer']['mode']='fixed'
 if resolved!=expected:raise ValueError('range control changes more than learned versus fixed mode')
 save(U/'fixed-config.json',c)
 record=dict(status='POST_INITIAL_CLIP_CONTROL_LOCKED',created_at_utc=datetime.now(timezone.utc).isoformat(),initial_publication=pin(initial),config=pin(U/'fixed-config.json'),resolved_hash=digest(resolved),parent=resolved['parent'],data=resolved['data'],matched_learned_result=pin(R/'interactions/G4D8-learned-small/result.json'),additional_updates=1024,max_windows=1024,supervised_positions=15360,zero_qualification_updates=0,finalists=['G4D8-learned-small-1024','G4D8-fixed-small-1024'],prior_consumptions=[pin(R/'heldout24/consumption.json')],rationale='Initial chart is fully published while user remains absent. Unit-initialized learned range recipes dominate W1 recovery, but learned-step effects were not isolated from initialization/range policy. One fixed-bound compact G4 control isolates bound learning under the same1024 parent/order/exposure. No new feature tier, target, loss or architecture.',budget='One1024-update window; no recovery extension. Zero native admission, one paired intermediate DEV smoke, final4-arm DEV12 and new24-case heldout with2same-process warmups/5alternating paired repetitions. No tuning on prior heldout.',storage='Opt-in streamed checkpoint avoids raw+archive duplication; full CPU optimizer/RNG native-export recovery uses owned RAM temporary. Existing paid checkpoints and Windows artifacts remain unchanged.')
 save(U/'plan.json',record);print('Post-initial one-window clipping control locked',flush=True)

def execute():
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);plan=json.load(open(U/'plan.json'));auth(plan['initial_publication']);c=resolve(auth(plan['config']))
 if digest(c)!=plan['resolved_hash']:raise ValueError('locked control config changed')
 out=U/'training'
 if not (out/'result.json').exists() and shutil.disk_usage(P).free<8*2**30+5_300_000_000+800_000_000:raise RuntimeError('insufficient checkpoint/output headroom with8GiB reserve')
 basis=R/'zero-arms/G8D8.gguf';base={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf')};learn=R/'interactions/G4D8-learned-small';out=U/'training'
 if not available(U/'model-zero.gguf'):export(c['parent']['path'],U/'model-zero.gguf',config=c)
 panel({**base,'G4D8-learned-small-zero':str(learn/'model-0.gguf'),'G4D8-fixed-small-zero':str(U/'model-zero.gguf')},U/'zero-admission','capture3')
 rows=json.load(open(U/'zero-admission/result.json'))['rows']
 for x in [x for x in rows if x['arm']=='G4D8-fixed-small-zero']:
  y=next(y for y in rows if y['arm']=='G4D8-learned-small-zero' and all(x[k]==y[k] for k in ('rep','prompt','instrumented')))
  if any(x['native'].get(k)!=y['native'].get(k) for k in ('accepted','proposed','rounds','prefix_survival_counts')):raise ValueError('fixed/learned unit-initialized native zero differs')
 if not (out/'result.json').exists():run([sys.executable,'-m','scripts.dflash_baseline.pipeline',plan['config']['path'],'--out',out,'--updates',1024,'--max-windows',1024,'--device','cuda','--adapt','--native-points',256,1024,'--checkpoint-archive'])
 result=json.load(open(out/'result.json'))
 if result['status']!='COMPLETE' or result['progress']['updates']!=1024:raise ValueError('control training not complete; retain recoverable state')
 if not (U/'checkpoint-recovery.json').exists():
  ram=Path('/dev/shm/precision-clip-control-20261105-01');ram.mkdir(exist_ok=False)
  raw=json.load(open(out/'checkpoint.pt.json'))['raw_bytes'];available_kib=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
  if shutil.disk_usage(ram).free<raw+(1<<30) or available_kib*1024<raw+(4<<30):raise RuntimeError('insufficient owned RAM recovery headroom')
  before=shutil.disk_usage('/dev/shm').free
  try:run([sys.executable,ROOT/'scripts/dflash_baseline/recover_clip_checkpoint.py'],dict(TMPDIR=str(ram)))
  finally:
   if not any(ram.iterdir()):ram.rmdir()
  save(U/'ram-recovery-storage.json',dict(path=str(ram),before_free_bytes=before,after_free_bytes=shutil.disk_usage('/dev/shm').free,owned_directory_released=not ram.exists(),scope='Owned CPU decode temporary only; full paid archive persists on WSL. No Windows move.'))
 d=dictionary(basis)
 for model in [U/'model-zero.gguf',*sorted(out.glob('*.gguf'))]:
  if model.exists():archive(model,basis,prepared=d)
 del d
 script('audit_component_weights.py',out/'model-final.gguf','--out',U/'component-code-audit.json')
 panel({**base,'G4D8-learned-small-zero':str(learn/'model-0.gguf'),'G4D8-fixed-small-zero':str(U/'model-zero.gguf'),'G4D8-learned-small-256':str(learn/'model-256.gguf'),'G4D8-fixed-small-256':str(out/'model-256.gguf')},U/'intermediate-dev12')
 arms={**base,'G4D8-learned-small-1024':str(learn/'model-final.gguf'),'G4D8-fixed-small-1024':str(out/'model-final.gguf')};panel(arms,U/'dev12',reps=5)
 selected=U/'selection.json'
 if not selected.exists():
  freeze=json.load(open(U/'dev12/freeze.json'));save(selected,dict(status='SHORTLIST_LOCKED_BEFORE_HELDOUT',heldout_opened=False,created_at_utc=datetime.now(timezone.utc).isoformat(),candidates={k:pin(v) for k,v in arms.items() if k not in ('target','q4')},target=freeze['target'],baseline=freeze['arms']['q4'],runtime=freeze['runtime'],training_data=plan['data'],context_policy=CONTEXT_POLICY,prior_consumptions=plan['prior_consumptions'],followup_plan=pin(U/'plan.json'),dev_confirmation=pin(U/'dev12/result.json'),selection_policy='Both predeclared1024 fixed/learned controls; no heldout-derived model, epoch or range selection. Initial24-case consumed roster excluded before new whole-component selection.'))
 if not (U/'heldout24/prompts.json').exists():script('prepare_holdout.py',selected,U/'heldout24')
 old=json.load(open(R/'heldout24/consumption.json'))['records'];new=json.load(open(U/'heldout24/consumption.json'))['records'];old_ids={v for x in old for v in x['identities'].values()};old_groups={v for x in old for v in (x['group_id'],x['original_group'])};old_topics={v for x in old for v in x['original_topics']}
 if any(old_ids.intersection(x['identities'].values()) or old_groups.intersection((x['group_id'],x['original_group'])) or old_topics.intersection(x['original_topics']) for x in new):raise ValueError('followup source components overlap initial heldout')
 save(U/'cohort-disjointness.json',dict(status='NEW_HELDOUT_COMPONENTS_DISJOINT',old=pin(R/'heldout24/consumption.json'),new=pin(U/'heldout24/consumption.json'),identity_group_topic_overlap=False))
 panel(arms,U/'heldout-panel','natural24',5,U/'heldout24/prompts.json','clip-heldout24')
 save(U/'native-measurements.json',dict(status='FOLLOWUP_NATIVE_MEASUREMENTS_COMPLETE',plan=pin(U/'plan.json'),dev=pin(U/'dev12/result.json'),heldout=pin(U/'heldout-panel/result.json'),recovery=pin(U/'checkpoint-recovery.json'),scope='Raw audit and followup chart publication still required; initial artifacts stay frozen.'))
 print('Bounded fixed-versus-learned control measured; audit/charts remain',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args();prepare() if a.mode=='prepare' else execute()
