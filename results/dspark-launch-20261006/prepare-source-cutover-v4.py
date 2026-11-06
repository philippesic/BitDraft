import json,sys,subprocess
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';O=R/'runs/dspark-a8-full-pool-20261006-01';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.block_shard_lifecycle import pin
from w1a1_eagle.nine_model_pipeline import atomic_json
v=json.loads((O/'packet-inputs.json').read_text());v['runtime']=pin(O/'runtime.inference-v2.json')
p=O/'packet-inputs.source-cutover-v4.json';assert not p.exists();atomic_json(p,v)
subprocess.run([str(R/'.venv/bin/python'),str(C/'scripts/prepare_block_lane_packet.py'),'prepare','--inputs',str(p),'--inputs-sha256',pin(p)['sha256'],'--output',str(O/'packet-v4')],check=True)
print(json.dumps(json.loads((O/'packet-v4/commands.json').read_text())),flush=True)
