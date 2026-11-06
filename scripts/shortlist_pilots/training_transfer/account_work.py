"""Count actual native outputs and preserve exact new capture requests/counters."""
from pathlib import Path
import json,hashlib,subprocess
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def main():
 counts={};partial=[]
 for base in ('current-native','historical-native'):
  for d in (RUN/base).iterdir():
   if not d.is_dir():continue
   count=0
   for p in d.glob('cell-*/rows.jsonl'):
    for i,line in enumerate(p.read_text().splitlines()):
     try:json.loads(line);count+=1
     except json.JSONDecodeError:partial.append({'path':str(p),'line':i})
   counts[base+'/'+d.name]=count
 cap=RUN/'capture05';complete=json.loads((cap/'complete.json').read_text());rounds=[json.loads(l) for l in (cap/'rounds.jsonl').read_text().splitlines()];records=[]
 for rec in complete['records']:
  request=json.loads(Path(rec['response']['path']).read_text());rr=[r for r in rounds if r['task_id']==rec['task_id']];ids=request['response']['tokens'];records.append({'case_index':rec['case_index'],'group_id':request['case']['group_id'],'domain':request['case']['domain'],'role':request['case']['role'],'body':request['body'],'generated_token_ids':ids,'actual_output_tokens':len(ids),'diagnostic_full_request_seconds':request['diagnostic_wall_seconds'],'accepted':sum(r['accepted_drafts'] for r in rr),'proposed':sum(len(r['draft_token_ids']) for r in rr),'rounds':len(rr),'source_request':rec['response'],'scope':'Instrumented feature/logit capture; no throughput acceptance claim or independently repeated target-only assertion on this24group panel.'})
 (RUN/'new-capture-requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 command=json.loads((cap/'command.json').read_text());env=command['environment'];command['environment']={k:v for k,v in env.items() if k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE','OMP_NUM_THREADS','GGML_W1AX_DP4A','GGML_W1A8_TOKEN7','GGML_W1A8_ALGORITHM','DSPARK_REQUIRE_AUTHOR_LAYOUT','DSPARK_QUALITY_CAPTURE') or k.startswith('EAGLE_CAPTURE_') or k=='EAGLE_RECORD_ROUNDS_JSONL'};(RUN/'new-capture-command.json').write_text(json.dumps(command,indent=2))
 head=len((RUN/'head-controls/native/rows.jsonl').read_text().splitlines());result={'planned_scientific_native_case_evaluations':2016,'scientific_cases_include_declared_panel_overlap':True,'observed_native_case_rows_by_arm':counts,'observed_native_case_rows_including_controls':sum(counts.values())+head,'head_control_cases':head,'partial_non_JSON_native_rows':partial,'actor_model_window_pairs':1536,'actor_forward_calls_two_conditionings':3072,'new_instrumented_native_requests':len(records),'new_capture_actual_output_tokens':sum(r['actual_output_tokens'] for r in records),'new_capture_diagnostic_request_wall_seconds':sum(r['diagnostic_full_request_seconds'] for r in records),'new_capture_accepted':sum(r['accepted'] for r in records),'new_capture_proposed':sum(r['proposed'] for r in records),'new_capture_rounds':sum(r['rounds'] for r in records),'new_training_updates':0,'new_TRAIN_seconds':0,'reused_current_HTTP_rows_recomputed_on_Mac':72,'reused_historical_HTTP_scope':'Original five-repetition DSparkD7/EAGLEQ4_0/target-only protocol, retained research-only strict failure; no new historical requests.', 'gxx':subprocess.check_output(['g++','--version'],text=True),'uname':subprocess.check_output(['uname','-a'],text=True)};(RUN/'native-work-accounting.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
