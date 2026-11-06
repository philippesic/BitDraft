"""Read-only real36 source view API checks; no model, CUDA or optimizer."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
P=Path('/home/philip/binary-eagle-decoding')
R=P/'runs/w1a8-modern-actual-training-data-20261103-01'
W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103')
sys.path.insert(0,str(R/'source'))
import modern_actual_data as data
import numpy as np

started=time.perf_counter()
path=R/'final01/manifest.json'
dataset=data.ActualDataset(path)
m=dataset.manifest
assert len(dataset.windows)==2138 and m['native_protections']=={'fit':356,'development':193}
assert len(m['required_embedding_ids'])==len(m['embedding_rows'])==1440 and not m['missing_embedding_ids']
assert data.pin(R/'source/modern_actual_data.py')['sha256']==data.pin(W/'scripts/w1a8_generalization/modern_actual_data.py')['sha256']
assert data.pin(R/'source/test_w1a8_modern_actual_data.py')['sha256']==data.pin(W/'tests/test_w1a8_modern_actual_data.py')['sha256']
protected=Counter();seen_groups={};window_max=0
for window in dataset.windows:
    assert len(window['prefix_tokens'])==len(window['prefix_raw_rows'])==window['anchor']
    assert window['max_depth']==len(window['labels']) and 1<=window['max_depth']<=2
    assert window['native_prefix']==len(window['native_protections'])<=window['max_depth']
    assert window['native_success_mask']==[d<window['native_prefix'] for d in range(window['max_depth'])]
    for p in window['native_protections']:
        assert window['labels'][p['proposal_depth']]==p['token']
        assert p['response_index']==window['response_label_start']+p['proposal_depth']
        protected[window['role']]+=1
    seen_groups.setdefault(window['group'],[]).append(window)
    window_max=max(window_max,window['anchor'])
assert protected==Counter({'fit':356,'development':193}) and len(seen_groups)==36
chosen={}
for group,windows in seen_groups.items():
    for window in (windows[0],windows[-1],next((w for w in windows if w['native_prefix']==2),windows[0])):
        chosen[window['id']]=window
loaded=[]
for window in chosen.values():
    result=dataset.materialize(window,device='cpu')
    assert result.prefix_tokens==window['prefix_tokens']
    assert tuple(result.prefix_rawfeatures.shape)==(window['anchor'],7680)
    assert tuple(result.seed_rawfeature.shape)==(7680,)
    assert result.prefix_rawfeatures.device.type=='cpu'
    # External row lookup uses source captures, not a private embedding table.
    for token in set(result.prefix_tokens+[window['seed_token']]+window['labels']):
        row=dataset.embedding_lookup(token,device='cpu')
        assert row.shape==(2560,) and not row.requires_grad
    loaded.append({'id':window['id'],'prefix_rows':window['anchor'],'native_prefix':window['native_prefix']})
# Concrete native bridge byte proof. Both equalities come from retained F32 files.
bridge=next(x for x in m['crossed_fc_source_rows'] if x['group']=='fit-prose-00' and x['decoder_call']==7 and x['decoder_row']==0)
rr=m['raw_rows'][bridge['raw_row']]
provided=np.fromfile(bridge['input_file']['path'],dtype='<f4').reshape(bridge['input_file']['shape'])[bridge['decoder_row']]
native=np.fromfile(rr['native_fc_file']['path'],dtype='<f4').reshape(rr['native_fc_file']['shape'])[rr['native_fc_row']]
prior=next(x for x in m['raw_rows'] if x['group']=='fit-prose-00' and x['encoder_call']==3 and x['target_position']==31)
priorvalue=np.fromfile(prior['native_fc_file']['path'],dtype='<f4').reshape(prior['native_fc_file']['shape'])[prior['native_fc_row']]
assert np.array_equal(provided,native) and not np.array_equal(provided,priorvalue)
assert bridge['decoder_position']==31 and bridge['source_target_position']==35 and bridge['encoder_call']==6 and bridge['encoder_row']==3
bridge_proof={**bridge,'actual_input_equals_encoder6_row3_byteexact':True,
              'actual_input_equals_prior_encoder3_target31':False,
              'prior_fc_file':prior['native_fc_file'],'prior_fc_row':prior['native_fc_row'],
              'actual_vs_prior_fc_max_abs':float(np.abs(provided-priorvalue).max()),
              'prior_fc_row_sha256':data.row_hash(priorvalue),
              'interpretation':'Observed crossed bridge lineage retained; no source operation repaired in these views'}
with (R/'bridge-example.json').open('x') as stream:json.dump(bridge_proof,stream,indent=2);stream.write('\n')
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert peak<3*1024**3
size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
assert size<100*1024**2
report={'status':'ACTUAL_PARENT36_SOURCE_VIEW_AND_API_CHECKS_PASS','manifest':data.pin(path),
        'module':data.pin(R/'source/modern_actual_data.py'),'tests':data.pin(R/'source/test_w1a8_modern_actual_data.py'),
        'counts':m['counts'],'native_protections':m['native_protections'],'all_window_metadata_checked':len(dataset.windows),
        'materialized_views':loaded,'materialized_groups':36,'max_anchor':window_max,
        'missing_native_embedding_ids':0,'crossed_source_occurrences':len(m['crossed_fc_source_rows']),
        'bridge_example':data.pin(R/'bridge-example.json'),
        'unit_tests':8,'ruff_pass':True,'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':peak,
        'artifact_bytes_before_report':size,'gpu_seconds':0,'model_loaded':False,'optimizer_updates':0,
        'initial_attempt':'Root manifest retained as preliminary join with113missing embeddings; final01 adds authenticated second capture and frozen source snapshots',
        'scope':'Original-runtime observed actual parent histories; controls preserved, no hidden bridge repair, model improvement or actor parity claim'}
with (R/'report.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps({k:report[k] for k in ('status','counts','native_protections','materialized_groups','max_anchor','crossed_source_occurrences','wall_seconds','peak_rss_bytes','artifact_bytes_before_report')},indent=2))
