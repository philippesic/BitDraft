"""Audit and publish only the bounded post-initial clipping control."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.precision_benchmark import pin,save,auth
from scripts.dflash_baseline.audit_precision import audit_panel
from scripts.dflash_baseline.prototype import digest
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';U=R/'clip-control-followup'
def paired_effect(panel):
 import numpy as np
 rows=auth(pin(Path(panel)/'result.json'))['rows'];fixed='G4D8-fixed-small-1024';learned='G4D8-learned-small-1024'
 groups={name:{(x['rep'],x['prompt']):x for x in rows if x['arm']==name and not x['warmup'] and not x['instrumented']} for name in (fixed,learned)}
 if groups[fixed].keys()!=groups[learned].keys():raise ValueError('paired effect lacks matching cases')
 reps=sorted({r for r,p in groups[fixed]});prompts=sorted({p for r,p in groups[fixed]})
 def stats(keys):
  a=[groups[fixed][k] for k in keys];b=[groups[learned][k] for k in keys]
  rate=lambda xs:sum(x['tokens'] for x in xs)/sum(x['wall_seconds'] for x in xs)
  decode=lambda xs:sum(x['tokens'] for x in xs)/sum(x['timings']['predicted_ms'] for x in xs)
  acceptance=lambda xs:sum(x['native']['accepted'] for x in xs)/sum(x['native']['proposed'] for x in xs)
  return [rate(a)/rate(b),decode(a)/decode(b),acceptance(a)-acceptance(b)]
 observed=stats(list(groups[fixed]));rng=np.random.default_rng(811);samples=[]
 for _ in range(2000):
  rr=rng.choice(reps,len(reps));pp=rng.choice(prompts,len(prompts));samples.append(stats([(r,p) for r in rr for p in pp]))
 intervals=np.quantile(samples,[.025,.975],axis=0)
 return dict(request_fixed_over_learned=observed[0],request_ratio95=intervals[:,0].tolist(),decode_fixed_over_learned=observed[1],decode_ratio95=intervals[:,1].tolist(),acceptance_fixed_minus_learned=observed[2],acceptance_difference95=intervals[:,2].tolist(),seed=811,resamples=2000,method='Paired two-way resampling of whole prompt components and repetitions; prompts sampled once across selected repetitions. No independent-training variance.')

def bound_metadata(path):
 from scripts.dflash_baseline.native_archive import hydrate,release
 from scripts.dflash_baseline.model import GGUFReader
 import math
 path=Path(path);created=hydrate(path)
 try:
  reader=GGUFReader(path);bounds={name.removesuffix('.w1ax.clip'):float(field.contents()) for name,field in reader.fields.items() if name.endswith('.w1ax.clip')};modes={name.removesuffix('.w1ax.mode'):int(field.contents()) for name,field in reader.fields.items() if name.endswith('.w1ax.mode')}
  if len(bounds)!=18 or len(modes)!=18 or not all(v>0 and math.isfinite(v) for v in bounds.values()):raise ValueError('incomplete positive range metadata')
  del reader
  return dict(model=pin(path),bounds=bounds,modes=modes,min_bound=min(bounds.values()),max_bound=max(bounds.values()))
 finally:release(path,created)

def data():
 measurements=auth(pin(U/'native-measurements.json'));plan=auth(measurements['plan']);initial=auth(plan['initial_publication'])
 if measurements['status']!='FOLLOWUP_NATIVE_MEASUREMENTS_COMPLETE' or initial['status']!='FULL_INITIAL_COMPARISON_COMPLETE':raise ValueError('measured endpoints missing')
 for ref in initial['artifacts'].values():
  if pin(ref['path'])!=ref:raise ValueError('frozen initial artifact changed')
 panels={};expected={'zero-admission':(4,3,1),'intermediate-dev12':(6,12,1),'dev12':(4,12,5),'heldout-panel':(4,24,5)}
 for name,(arms,prompts,reps) in expected.items():
  p=U/name;result=auth(pin(p/'result.json'));freeze=auth(pin(p/'freeze.json'));rows=result['rows'];warm=freeze['warmup_policy']
  if not result['paired_trace_counters_equal'] or len(rows)!=arms*2*reps*(prompts+2) or len(freeze['arms'])!=arms or not (warm['same_process'] and warm['per_server_cell']==2 and warm['measured_repetitions']==reps):raise ValueError('followup native protocol/coverage changed')
  panels[name]=audit_panel(p)
 dev=auth(pin(U/'dev12/freeze.json'));held=auth(pin(U/'heldout-panel/freeze.json'));old=auth(pin(R/'confirmation-dev12/freeze.json'))
 if held['cohort']!='clip-heldout24' or dev['cohort']!='natural12':raise ValueError('distinct heldout cohort labels required')
 for current in (dev,held):
  if current['runtime']!=old['runtime'] or current['target']!=old['target'] or any(current['baseline_registry'][k]!=old['baseline_registry'][k] for k in ('sha256','bytes')) or current['arms']['q4']!=old['arms']['q4']:raise ValueError('matched runtime/target/Q4 control changed')
 selected=auth(pin(U/'selection.json'))
 if selected['prior_consumptions']!=plan['prior_consumptions'] or selected['candidates']!={k:v for k,v in dev['arms'].items() if k not in ('target','q4')}:raise ValueError('shortlist or exclusion lock changed')
 if auth(pin(U/'cohort-disjointness.json'))['status']!='NEW_HELDOUT_COMPONENTS_DISJOINT':raise ValueError('fresh cohort overlaps')
 fixed=auth(pin(U/'training/result.json'));learned=auth(plan['matched_learned_result']);resolved=auth(pin(U/'training/resolved.json'));expected_config=auth(pin(R/'interactions/G4D8-learned-small/resolved.json'));expected_config['quantizer']['mode']='fixed'
 if digest(resolved)!=plan['resolved_hash'] or resolved!=expected_config or any(fixed['progress'][k]!=learned['progress'][k] for k in ('updates','cursor','order','supervised_tokens')) or fixed['progress']['updates']!=1024 or fixed['progress']['supervised_tokens']!=15360:raise ValueError('fixed/learned matching failed')
 new=[json.loads(x) for x in (U/'training/updates.jsonl').read_text().splitlines()];prior=[json.loads(x) for x in (R/'interactions/G4D8-learned-small/updates.jsonl').read_text().splitlines()]
 if len(new)!=1024 or len(prior)!=1024 or any((a['sequence'],a['anchor'],a['update'],a['cursor'])!=(b['sequence'],b['anchor'],b['update'],b['cursor']) for a,b in zip(new,prior)):raise ValueError('actual example/update order differs')
 recovery=auth(measurements['recovery'])
 if recovery['status']!='FULL_STREAMED_CUDA_CHECKPOINT_ROUNDTRIP_PASS' or recovery['optimizer_states']!=recovery['trainable_parameters'] or any(recovery['model'][k]!=recovery['resident_model'][k] for k in ('sha256','bytes')):raise ValueError('full state/export recovery failed')
 if pin(recovery['checkpoint']['path'])!=recovery['checkpoint']:raise ValueError('recovery checkpoint changed')
 components=auth(pin(U/'component-code-audit.json'))
 if len(components)!=1 or components[0]['status']!='DEPLOYED_COMPONENT_CODES_EXACT' or pin(components[0]['model'])['sha256']!=components[0]['sha256']:raise ValueError('protected component codes changed')
 ram=auth(pin(U/'ram-recovery-storage.json'))
 if not ram['owned_directory_released'] or Path(ram['path']).exists():raise ValueError('owned recovery RAM temporary retained')
 effects={name:paired_effect(U/name) for name in ('dev12','heldout-panel')};bounds={'fixed':bound_metadata(U/'training/model-final.gguf'),'learned':bound_metadata(R/'interactions/G4D8-learned-small/model-final.gguf')}
 if set(bounds['fixed']['modes'].values())!={1} or set(bounds['learned']['modes'].values())!={2} or set(bounds['fixed']['bounds'].values())!={1.}:raise ValueError('fixed/learned range metadata invalid')
 evidence=dict(status='FOLLOWUP_NATIVE_DATA_AUDITED',paired_effects=effects,range_metadata=bounds,completed_at_utc=datetime.now(timezone.utc).isoformat(),plan=pin(U/'plan.json'),initial_publication=plan['initial_publication'],panels=panels,matched_history=dict(updates=1024,supervised_positions=15360,cursor=fixed['progress']['cursor'],actual_examples_identical=True,additional_update_seconds=sum(x['seconds'] for x in new),additional_physical_updates=len(new)),checkpoint_recovery=measurements['recovery'],component_codes=pin(U/'component-code-audit.json'),cohort_disjointness=pin(U/'cohort-disjointness.json'),initial_artifacts_unchanged=True,source=pin(__file__),scope='Only one fixed-bound1024 control. Four formal models per followup cohort; new heldout remains distinct from initial24. Charts/publication separate.')
 save(U/'data-completion.json',evidence);print('Followup raw responses, matched actual examples, recovery and frozen-initial audit passed',flush=True)
def publish(out):
 out=Path(out);evidence=auth(pin(U/'data-completion.json'));comparison=auth(pin(out/'comparison.json'))
 if evidence['status']!='FOLLOWUP_NATIVE_DATA_AUDITED' or comparison['status']!='FULL_INITIAL_AND_CLIP_CONTROL_COMPARISON_COMPLETE' or not all(x['correct'] for x in comparison['points']):raise ValueError('followup charts incomplete')
 for name,(count,reps) in {'zero-admission':(4,1),'intermediate-dev12':(6,1),'dev12':(4,5),'heldout-panel':(4,5)}.items():
  rows=[p for p in comparison['points'] if p['panel']==str(U/name)]
  if len(rows)!=count or any(p['repetitions']!=reps for p in rows):raise ValueError('followup chart omissions')
 initial=auth(evidence['initial_publication'])
 for ref in initial['artifacts'].values():
  if pin(ref['path'])!=ref:raise ValueError('initial artifact changed during followup')
 names=['comparison.json','comparison.csv','domains.csv','full-table.md','comparison.html','generator.py']+[n+s for n in ('acceptance-latency','throughput','size-memory','recovery','precision-grid') for s in ('.png','.svg')];artifacts={name:pin(out/name) for name in names}
 if any(x['bytes']==0 for x in artifacts.values()):raise ValueError('empty followup artifact')
 if (U/'completion.json').exists():raise FileExistsError('followup charts already frozen')
 save(U/'completion.json',dict(status='FULL_INITIAL_AND_CLIP_CONTROL_COMPARISON_COMPLETE',completed_at_utc=datetime.now(timezone.utc).isoformat(),data=pin(U/'data-completion.json'),charts_directory=str(out.resolve()),artifacts=artifacts,initial_publication=evidence['initial_publication'],point_count=len(comparison['points']),scope='Initial measured comparison plus completed bounded fixed/learned control, no broader goal completion.'))
 print('Followup comparison published:',out,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['data','publish']);ap.add_argument('--charts',type=Path);a=ap.parse_args()
 if a.mode=='data':data()
 else:
  if a.charts is None:ap.error('--charts required')
  publish(a.charts)
