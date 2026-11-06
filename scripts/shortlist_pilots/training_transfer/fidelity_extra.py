"""Unchanged reference fidelity heuristics, with full-panel and stratum scopes explicit."""
from pathlib import Path
import json
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def stats(deltas):
 n=len(deltas);different=sum(d!=0 for d in deltas);mae=sum(abs(d) for d in deltas)/n
 return {'windows':n,'prefix_disagreements':different,'fraction':different/n,'mean_absolute_prefix_error':mae,'native_minus_actor_prefix':sum(deltas),'reference_heuristics_pass':different/n<=5/96 and mae<=.1}
def main():
 windows=json.loads((RUN/'historical-matrix.json').read_text())['windows'];parent=json.loads((RUN/'historical-native/parent3504/result.json').read_text())['rows'];serving=stats([r['greedy']['prefix']-min(5,w['accepted_drafts']) for w,r in zip(windows,parent)]);result={'thresholds':{'disagreement_fraction':5/96,'mean_absolute_prefix_error':.1,'origin':'unchanged historical allocation heuristics; descriptive per-stratum values are not new acceptance/adoption gates'},'historical_states_parent_native_vs_serving':serving,'historical_full_panel_actor_native':{},'current_full_panel_actor_native':{}}
 for arm in ('initializer','8hour','16hour','parent3504','EAL2160'):
  base=RUN/('historical-actor' if arm in ('initializer','8hour','16hour') else 'historical-current-actor');ap=base/arm/'result.json'
  if not ap.exists():continue
  a=json.loads(ap.read_text())['rows'];n=json.loads((RUN/'historical-native'/arm/'result.json').read_text())['rows'];result['historical_full_panel_actor_native'][arm]=stats([x['greedy']['prefix']-y['greedy']['prefix'] for x,y in zip(n,a)])
 for arm in ('parent3504','EAL2160'):
  a=json.loads((RUN/'current-actor'/arm/'result.json').read_text())['rows'];n=json.loads((RUN/'current-native'/arm/'result.json').read_text())['rows'];result['current_full_panel_actor_native'][arm]={'both_panels_288':stats([x['greedy']['prefix']-y['greedy']['prefix'] for x,y in zip(n,a)]),'enriched_original_144':stats([x['greedy']['prefix']-y['greedy']['prefix'] for x,y in zip(n[:144],a[:144])])}
 result['status']='FIDELITY_SCOPE_AUDIT_COMPLETE';(RUN/'fidelity-extra.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
