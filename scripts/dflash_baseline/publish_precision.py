"""Gate native data first; publish completion only after real charts validate."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dflash_baseline.precision_benchmark import pin,save,auth
from scripts.dflash_baseline.audit_precision import campaign
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01'
def panel(path,arms,prompts):
 result=json.load(open(path/'result.json'));freeze=json.load(open(path/'freeze.json'))
 if result['status']!='COMPLETE' or not result['paired_trace_counters_equal'] or not all(x['correct'] for x in result['rows']):raise ValueError('native panel incomplete/incorrect')
 policy=freeze['warmup_policy']
 if not (policy['per_server_cell']==2 and policy['measured_repetitions']==5 and policy['same_process']):raise ValueError('formal warmup/repetition protocol')
 rows=[x for x in result['rows'] if not x['warmup']];keys={(x['arm'],x['rep'],x['instrumented']) for x in rows}
 expected={(arm,rep,traced) for arm in arms for rep in range(5) for traced in (False,True)}
 if keys!=expected or len(rows)!=len(expected)*prompts:raise ValueError('native panel lacks complete arm/repetition coverage')
 for key in keys:
  found=[x for x in rows if (x['arm'],x['rep'],x['instrumented'])==key]
  if len(found)!=prompts or len({x['prompt'] for x in found})!=prompts:raise ValueError('native prompt coverage differs')
 return freeze
def data():
 dev_arms=json.load(open(R/'confirmation-arms.json'));held_arms=json.load(open(R/'heldout-arms.json'));dev=panel(R/'confirmation-dev12',dev_arms,12);held=panel(R/'heldout-panel',held_arms,24)
 if len(dev_arms)!=53 or len(held_arms)!=6 or dev['runtime']!=held['runtime'] or dev['target']!=held['target'] or any(dev['baseline_registry'][k]!=held['baseline_registry'][k] for k in ('sha256','bytes')):raise ValueError('matched lane/runtime changed')
 if json.load(open(R/'slot16-qualification/result.json'))['status']!='GAINED_STREAM_CUDA36_N16_BYTEEXACT':raise ValueError('actual16-slot arithmetic unqualified')
 if json.load(open(R/'tokenizer/qualification.json'))['status']!='NATIVE_VOCAB_ADMISSION_QUALIFIED':raise ValueError('heldout context admission unqualified')
 component_refs={name:pin(R/name) for name in ('component-code-audit.json','final-component-code-audit.json')}
 for name,count in (('component-code-audit.json',20),('final-component-code-audit.json',2)):
  document=auth(component_refs[name]);records=document.get('models') if isinstance(document,dict) else document
  if isinstance(document,dict) and document.get('status')!='DEPLOYED_COMPONENT_CODES_EXACT':raise ValueError('component audit summary failed')
  if len(records)!=count or any(x['status']!='DEPLOYED_COMPONENT_CODES_EXACT' or pin(x['model'])['sha256']!=x['sha256'] for x in records):raise ValueError('deployed protected component-code audit incomplete')
 final_models={str(R/'finalists/G4D8-learned-small'/name) for name in ('model-1536.gguf','model-final.gguf')}
 if {x['model'] for x in auth(component_refs['final-component-code-audit.json'])}!=final_models:raise ValueError('final component audit covers wrong exports')
 audit=campaign(R);pre=json.load(open(R/'pre-confirmation-audit.json'))
 if len(audit['finalists'])!=4 or audit['physical_training_updates']!=pre['physical_training_updates']:raise ValueError('training changed after predeclared confirmation lock')
 save(R/'initial-native-audit.json',audit)
 record=dict(status='INITIAL_NATIVE_DATA_AUDITED',component_codes=component_refs,sources=dict(publisher=pin(__file__),auditor=pin(ROOT/'scripts/dflash_baseline/audit_precision.py'),component_auditor=pin(ROOT/'scripts/dflash_baseline/audit_component_weights.py')),completed_at_utc=datetime.now(timezone.utc).isoformat(),dev=pin(R/'confirmation-dev12/result.json'),heldout=pin(R/'heldout-panel/result.json'),audit=pin(R/'initial-native-audit.json'),selection=pin(R/'heldout-selection.json'),slot16=pin(R/'slot16-qualification/result.json'),tokenizer=pin(R/'tokenizer/qualification.json'),scope='Native measurements and raw response/ancestry audit complete. Charts/publication are separate; this file is not campaign completion.')
 save(R/'native-data-completion.json',record)
def publish(out):
 out=Path(out);native=json.load(open(R/'native-data-completion.json'))
 if native['status']!='INITIAL_NATIVE_DATA_AUDITED':raise ValueError('native data gate incomplete')
 for key in ('dev','heldout','audit','selection','slot16','tokenizer'):auth(native[key])
 for ref in native['component_codes'].values():auth(ref)
 for ref in native['sources'].values():
  if pin(ref['path'])!=ref:raise ValueError('auditor source changed')
 comparison=json.load(open(out/'comparison.json'))
 if comparison['status']!='FULL_INITIAL_COMPARISON_COMPLETE' or not all(p['correct'] for p in comparison['points']):raise ValueError('charts incomplete or incorrect')
 for name,count in (('confirmation-dev12',53),('heldout-panel',6)):
  points=[p for p in comparison['points'] if Path(p['panel']).name==name]
  if len(points)!=count or any(p['repetitions']!=5 or p['scope']!='confirmation' for p in points):raise ValueError('charts omit formal measurements')
 names=['comparison.json','comparison.csv','domains.csv','full-table.md','comparison.html','generator.py']+[name+suffix for name in ('acceptance-latency','throughput','size-memory','recovery','precision-grid') for suffix in ('.png','.svg')]
 artifacts={name:pin(out/name) for name in names}
 if any(ref['bytes']==0 for ref in artifacts.values()):raise ValueError('empty chart artifact')
 marker=R/'campaign-completion.json'
 if marker.exists():raise FileExistsError('initial comparison is already frozen')
 save(marker,dict(status='FULL_INITIAL_COMPARISON_COMPLETE',completed_at_utc=datetime.now(timezone.utc).isoformat(),native_data=pin(R/'native-data-completion.json'),charts_directory=str(out.resolve()),artifacts=artifacts,scope='Full initial4B/Q4 lane only; frozen vanilla/reference and any later followup remain separate. Full plots/tables exist; no metadata-only completion.'))
 print('Full initial native comparison published:',out,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['data','publish']);ap.add_argument('--charts',type=Path);a=ap.parse_args()
 if a.mode=='data':data()
 else:
  if a.charts is None:ap.error('--charts required for publish')
  publish(a.charts)
