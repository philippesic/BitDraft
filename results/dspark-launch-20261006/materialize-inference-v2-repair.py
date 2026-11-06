import json,hashlib,subprocess,sys
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-full-training-20261006-01';OUT=R/'runs/dspark-a8-full-pool-20261006-01';B=R/'runs/build/native-seven-5eb221-sm120-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import Files,atomic_json
from w1a1_eagle.block_shard_lifecycle import pin
f=Files();prov=R/'results/dspark-launch-20261006/native-seven-build-provenance.json';proof=json.loads(prov.read_text());assert proof['status']=='PASS' and proof['native_commit']=='5eb221ec53cb4317aef963e3bdec3cc69401eca4';[f.check({k:p[k] for k in ('path','sha256')}) for p in proof['files']]
runtime=json.loads((OUT/'runtime.packet.json').read_text());old=dict(runtime['inputs']);assert old['teacher_binary']['sha256']=='63eacbb8e600488c76122dfd29d2da168a8a371db7c244b2b10d90d1d21c478f';f.check(old['teacher_binary'])
runtime['inputs']={k:v for k,v in old.items() if not k.startswith('runtime_library_')};runtime['inputs'].update(binary=pin(B/'bin/llama-server'),backend_binary=pin(B/'bin/test-backend-ops'),block_native_binary=pin(B/'bin/test-block-binary'))
for p in sorted((B/'bin').glob('*.so*')):
 if p.is_file():runtime['inputs']['runtime_library_'+p.name.replace('.','_').replace('-','_')]=pin(p)
runtime.update(inference_native_source_revision=proof['native_commit'],inference_build_provenance=pin(prov),capture_runtime_preserved={'runtime':pin(OUT/'runtime.packet.json'),'capture_revision':runtime['native_source_revision'],'teacher_binary':old['teacher_binary']})
assert not (OUT/'runtime.inference-v2.json').exists();atomic_json(OUT/'runtime.inference-v2.json',runtime)
qa=R/'results/dspark-launch-20261006/portable-qa-v2/qa-ledger.json';f.check(pin(qa));d=json.loads((OUT/'packet/production-inputs.json').read_text());d['qa_ledger']=pin(qa)
for key in list(d['inputs']):
 if key.startswith('runtime_library_'):del d['inputs'][key]
for key,value in runtime['inputs'].items():
 if key in ['binary','backend_binary','block_native_binary'] or key.startswith('runtime_library_'):d['inputs'][key]=value
# Native source revision remains the exact frozen teacher revision required by fresh capture portability.
# The distinct serving graph repair is bound by its own complete build proof/source commit.
d['preflight'].setdefault('build_provenance',{})['inference_seven_reserve']=pin(prov)
d['preflight']['build_provenance']['capture_runtime_preserved']=pin(OUT/'runtime.packet.json')
assert not (OUT/'production-inputs.inference-v2.json').exists();atomic_json(OUT/'production-inputs.inference-v2.json',d)
print(json.dumps({'runtime':pin(OUT/'runtime.inference-v2.json'),'production_inputs':pin(OUT/'production-inputs.inference-v2.json'),'paid_updates':0}))
