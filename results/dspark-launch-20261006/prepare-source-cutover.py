import argparse,json,sys,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();C=a.checkout;R=Path('/home/philip/binary-eagle-decoding');OUT=R/'runs/dspark-a8-full-pool-20261006-01';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.block_shard_lifecycle import pin
from w1a1_eagle.nine_model_pipeline import atomic_json
assert subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip()==a.commit
assert not subprocess.check_output(['git','-C',str(C),'status','--porcelain'],text=True).strip()
v=json.loads((OUT/'packet-inputs.json').read_text());v['runtime']=pin(OUT/'runtime.inference-v2.json')
for role in ['capture','restore']:
 producer=C/'scripts/capture_nine_model_train_data.py'
 # Preserve the actual original helper spelling and arguments.
 old=Path(v['shard_lifecycle'][role]['producer']['path']);producer=C/'scripts'/old.name
 assert pin(producer)['sha256']==v['shard_lifecycle'][role]['producer']['sha256']
 v['shard_lifecycle'][role]['producer']=pin(producer);v['shard_lifecycle'][role]['argv']=[str(R/'.venv/bin/python'),str(producer),'--request','{request}','--output','{output}']
 if role=='restore':
  old=Path(v['shard_lifecycle'][role]['replay_producer']['path']);new=C/'scripts'/old.name;assert pin(new)['sha256']==v['shard_lifecycle'][role]['replay_producer']['sha256'];v['shard_lifecycle'][role]['replay_producer']=pin(new)
q=OUT/'packet-inputs.source-cutover-v3.json';assert not q.exists();atomic_json(q,v)
command=[str(R/'.venv/bin/python'),str(C/'scripts/prepare_block_lane_packet.py'),'prepare','--inputs',str(q),'--inputs-sha256',pin(q)['sha256'],'--output',str(OUT/'packet-v3')]
print(json.dumps({'inputs':pin(q),'command':command,'paid_updates':0}),flush=True);subprocess.run(command,check=True)
