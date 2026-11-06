import json,sys,copy
from pathlib import Path
ROOT=Path('/home/philip/binary-eagle-decoding');OUT=ROOT/'runs/dspark-a8-full-pool-20261006-01';sys.path.insert(0,str(ROOT/'runs/checkouts/dspark-full-training-20261006-01/src'))
from w1a1_eagle.block_shard_lifecycle import pin,checked
from w1a1_eagle.block_data import BlockDataset
from w1a1_eagle.nine_model_pipeline import atomic_json
old=OUT/'preparation-merged/manifest.json';manifest=json.loads(old.read_text());producer=json.loads(checked(manifest['producer']['receipt']).read_text())
records=json.loads((OUT/'preparation-records.json').read_text());expected={};provenance=[]
for record in records:
 source=json.loads(checked(record['manifest']).read_text());receipt=json.loads(checked(source['producer']['receipt']).read_text())
 for chain in source['chains']:
  cid=chain['chain_id'];assert cid not in expected;expected[cid]=receipt['chains'][cid]
 provenance.append({'manifest':record['manifest'],'admission':record['admission'],'receipt':source['producer']['receipt']})
assert set(expected)=={c['chain_id'] for c in manifest['chains']}
for chain in manifest['chains']:
 cid=chain['chain_id'];wanted=expected[cid];actual=producer['chains'][cid]
 assert set(wanted)-set(actual)=={'label_policy'} and actual=={k:v for k,v in wanted.items() if k!='label_policy'}
 assert chain['label_policy']==wanted['label_policy']=='all_available_v1'
 producer['chains'][cid]=copy.deepcopy(wanted)
new=OUT/'preparation-merged-v2';assert not new.exists();new.mkdir()
# Preserve individual native receipt/tensor bytes and copy only corrected joins.
atomic_json(new/'original-train-inventory.json',json.loads(checked(manifest['train_inventory']).read_text()))
manifest['train_inventory']=pin(new/'original-train-inventory.json');producer['train_inventory_sha256']=manifest['train_inventory']['sha256']
atomic_json(new/'aggregate-producer-receipt.json',producer);manifest['producer']['receipt']=pin(new/'aggregate-producer-receipt.json')
atomic_json(new/'manifest.json',manifest)
ds=BlockDataset(new/'manifest.json',expected_sha256=pin(new/'manifest.json')['sha256'],allow_synthetic=False)
try:ds.write_admission(new/'completed-admission.json')
finally:ds.close()
receipt={'schema':'block_preparation_label_join_repair_v1','status':'PASS','original_failed_manifest':pin(old),'original_records':provenance,'repair_producer':pin(Path(__file__)),'only_changed_join_field':'label_policy','chains':len(manifest['chains']),'native_receipt_or_tensor_bytes_changed':False,'data':pin(new/'manifest.json'),'admission':pin(new/'completed-admission.json')}
atomic_json(new/'repair-provenance.json',receipt);print(json.dumps(receipt),flush=True)
