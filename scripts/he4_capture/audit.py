"""Verify the live manifest and immutable receipts, sample retained tensors."""
import hashlib,json,pathlib,sys,numpy as np
p=pathlib.Path(sys.argv[1]);manifest=json.loads(p.read_text());counts={k:0 for k in ('sequences','windows','unique_label_positions','unique_label_positions_in_windows')}
def check(pin):
 f=pathlib.Path(pin['path']);assert f.stat().st_size==pin['bytes'];assert hashlib.sha256(f.read_bytes()).hexdigest()==pin['sha256'];return f
rows=manifest['records'];sample={0,len(rows)//2,len(rows)-1};samples=[]
for i,row in enumerate(rows):
 proof=json.loads(check(row['receipt']).read_text())
 for key in counts:counts[key]+=proof[key]
 assert proof['greedy_vs_full_logits_argmax'] and proof['no_post_eog_labels']
 assert not pathlib.Path(proof['full_logits']['path']).exists()
 if i not in sample:continue
 for spec in proof['files']:check(spec)
 out=pathlib.Path(proof['metadata']['path']).parent;c=json.loads(check(proof['metadata']).read_text())
 a=np.load(out/'labels.npy');positions=np.load(out/'label_positions.npy');inputs=np.load(out/'inputs.npy')
 assert a.tolist()==c['greedy_ids'];assert positions.tolist()==c['logit_positions'];assert inputs.tolist()==c['input_tokens'];assert not any(t in c['eog_ids'] for t in a[:-1]);assert not any(t in c['eog_ids'] for t in inputs[c['prompt_tokens']:]);assert len(a)<=128
 assert c['taps']==[2,6,10,14,18,22,26,30]
 assert c['feature_semantics']=='zero_based_pre_block_input_all_tokens'
 assert proof['windows']==max(0,len(a)-14)
 samples.append(dict(id=row['id'],labels=len(a),windows=proof['windows'],eog=proof['eog_observed'],taps=c['taps']))
for key in counts:assert manifest[key]==counts[key]
print(json.dumps(dict(status='PASS',snapshot_updated_at=manifest['updated_at'],counts=counts,samples=samples),indent=2))
