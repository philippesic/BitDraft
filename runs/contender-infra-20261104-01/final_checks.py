import sys,json,time,copy
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';S=R/'source-v2';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01');sys.path[:0]=[str(S),'/home/philip/.codex/worktrees/w1a8-actual-training-20261103/scripts/w1a8_generalization','/home/philip/.codex/worktrees/w1a8-actual-training-20261103/src']
from runner import read,pin,atomic
from comparison import summarize
from bundle import create,verify
from manifest_stream import ManifestStream
from sharded_rows import ShardRows,ExposureSampler
from modern_actual_data import ActualDataset
import torch;torch.set_num_threads(2)
# Actual measured native panel through the common comparator.
root=R/'native-retry/serving';native=read(root/'result.json');models=read(root/'models.json');inputs=read(root/'inputs.json');byid={i['id']:i for i in inputs['rows']};runtimes={}
for arm in models['arms']:
 path=R/(arm['id']+'-runtime.json');atomic(path,arm['runtime']);runtimes[arm['id']]=pin(path)
panel=dict(schema='contender_comparison_v1',tier='contender',repetitions=5,cases=[r['id'] for r in inputs['rows']],arms=[a['id'] for a in models['arms']],baseline='dflash_Q4KM',model_pins={a['id']:a['model'] or models['target'] for a in models['arms']},runtime_pins=runtimes,target_pin=models['target'],workload_pin=pin(root/'inputs.json'),admission_pin=pin(R/'native-retry/audit.json'),semantics='greedy_exact_target',records=[])
for row in native['records']:
 if row['warmup']:continue
 panel['records'].append(dict(arm=row['arm'],case=row['id'],repetition=row['repetition'],wall_seconds=row['full_http_wall_seconds'],tokens=row['completion_tokens'],response=row['response'],reference=byid[row['id']]['reference'],request=row['request']))
atomic(R/'common-panel.json',panel);result=summarize(panel);atomic(R/'common-comparison.json',result)
# Check bytes, contract differences and meaningful updates independently.
a=read(C/'continuous/result.json');b=read(C/'resumed/result.json');c=read(C/'optimized/result.json')
assert len({v['model']['sha256'] for v in [a,b,c]})==1
ac=copy.deepcopy(a['contract']);cc=copy.deepcopy(c['contract']);ac.pop('freeze');cc.pop('freeze');assert ac==cc
assert all(sum(v['changed_scale_elements'].values())>0 for v in [a,b,c])
atomic(R/'export-and-contract.json',dict(status='IDENTICAL_EXPORT_AND_SCIENTIFIC_CONTRACT',export_sha256=a['model']['sha256'],changed_elements={k:sum(v['changed_scale_elements'].values()) for k,v in [('continuous',a),('resumed',b),('optimized',c)]},allowed_contract_difference='freeze pin only: explicitly recorded workflow helper/shard source pins'))
# Stream original independently admitted member metadata with a one-manifest cache.
parent=torch.load('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/checkpoint-00005760.pt',map_location='cpu',weights_only=False);union=read(parent['contract']['union_manifest']['path'])
registry=dict(schema='training_manifest_stream_v1',contract={'target':union['target']},shards=[])
for member in union['members']:
 value=read(member['manifest']['path']);windows=[w['id'] for w in value['windows'] if w['role']=='fit']
 registry['shards'].append(dict(manifest=member['manifest'],admission=pin(R/'data-probe-v2.json'),contract=registry['contract'],windows=windows))
atomic(R/'stream-registry.json',registry)
def factory(path):
 data=ActualDataset(path);data._rows=ShardRows(R/'shards/manifest.json',max_maps=2);return data
stream=ManifestStream(R/'stream-registry.json',factory,max_manifests=1);sampler=ExposureSampler(stream.ids,421104);sample_ids=[sampler.next() for _ in range(12)];saved=sampler.state();next_ids=[sampler.next() for _ in range(12)];restored=ExposureSampler(stream.ids,421104,saved);assert next_ids==[restored.next() for _ in range(12)]
checked=0
for key in sample_ids+next_ids:
 data,w=stream.window(key);actual=stream.materialize(key);reference=data.materialize(w)
 assert torch.equal(actual.prefix_rawfeatures,reference.prefix_rawfeatures) and torch.equal(actual.seed_rawfeature,reference.seed_rawfeature);checked+=1
atomic(R/'manifest-stream.json',dict(status='CURRENT_FIT_MANIFEST_STREAM_AND_SAMPLER_PASS',reader=stream.report(),windows_materialized=checked,sampler_resume_exact=True,scope='Existing FIT members only; no new source prompts selected. All-window byte equivalence is separately covered in data-probe-v2.'))
# Package existing verified reference, with no winner claim.
selection=dict(schema='contender_release_selection_v1',candidate='candidate_0',supported_workloads=['Current frozen4B Q4_K_M mode32 greedy consumed DEV; rehearsal only'],model=a['model'],target=models['target'],runtime_dependencies=[runtimes['candidate_0']],evidence={'native_audit':pin(R/'native-retry/audit.json'),'recovery_exact':pin(R/'exact.json'),'storage_exact':pin(R/'optimized-exact.json')},correctness='greedy_exact_target',claim='reference_rehearsal',settings=read(runtimes['candidate_0']['path']))
atomic(R/'bundle-selection.json',selection);create(R/'bundle-selection.json',R/'bundle');verify(R/'bundle')
atomic(R/'final-checks.json',dict(status='COMMON_COMPARISON_STREAM_EXPORT_AND_BUNDLE_PASS',comparison=pin(R/'common-comparison.json'),stream=pin(R/'manifest-stream.json'),exports=pin(R/'export-and-contract.json'),bundle=pin(R/'bundle/result.json')))
print('FINAL_CHECKS_PASS')
