from pathlib import Path
import json,hashlib,sys
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-actual-parent-capture-20261103-01/capture01';s=json.loads((R/'summary.json').read_text());sys.path.insert(0,str(P/'scripts/w1a8_generalization'));from modern_serving_trace import audit_trace
counts={};filecount=0;rawbytes=0
for g in s['groups']:
 assert g['status']=='ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS',g
 cell=R/g['id'];observed=json.loads((cell/'response.json').read_text());gold=json.loads((P/'runs/w1a8-modern-generalization-capture-20261103-02/serving'/g['id']/'response.json').read_text());assert all(observed[k]==gold[k] for k in ['tokens','content','stop','stop_type','stopping_word','truncated','tokens_predicted'])
 saved=json.loads((cell/'protection.json').read_text());again=audit_trace((cell/'server.log').read_text(),observed,expected_depth=2)
 for k in ['protections','protection_mask','draft_proposed','draft_verified_accepted','emitted_draft_successes','rounds']:assert saved[k]==json.loads(json.dumps(again[k])),(g['id'],k)
 for protection in saved['protections']:assert protection['token']==observed['tokens'][protection['response_index']]
 assert sum(saved['protection_mask'])==g['emitted_draft_successes'];release=json.loads((cell/'release.json').read_text());assert release['owned_process_reaped'] and release['owned_process_group_absent'];ident=json.loads((cell/'identity.json').read_text());assert not Path('/proc',str(ident['pid'])).exists()
 for ref in saved['native_output_pair_audit']['trace_files']:
  path=Path(ref['path'])
  with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ref['sha256']
  filecount+=1;rawbytes+=path.stat().st_size
 out=counts.setdefault(g['role'],{'groups':0,'response_tokens':0,'draft_proposed':0,'native_successes':0});out['groups']+=1;out['response_tokens']+=g['response_tokens'];out['draft_proposed']+=g['draft_proposed'];out['native_successes']+=g['emitted_draft_successes']
r={'status':'ROOT_ACTUAL_PARENT_RAW_AUDIT_PASS','counts':counts,'trace_files_rehashed':filecount,'trace_bytes_rehashed':rawbytes,'full_response_parity_all36':True,'native_success_tokens_join_raw_responses':True,'all_owned_servers_reaped_groups_absent':True,'collection_wall_seconds':s['wall_seconds'],'free_bytes_after_collection':s['free_bytes_after'],'scope':'Actual native parent histories and successful continuations for pilot; not trained generalization, quality, performance or all36CPUactor qualification'}
(R/'root-independent-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
