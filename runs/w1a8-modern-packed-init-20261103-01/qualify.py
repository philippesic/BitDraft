"""Zero-update native packed initialization; CPU serialization qualification only."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time
import numpy as np
from safetensors import safe_open

P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-packed-init-20261103-01'
sys.path.insert(0,str(P/'scripts'))
from export_recurrent_binary import SOURCE_NAMES,export_model
from audit_eagle_w1a1_gguf import audit
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
source=P/'runs/spec-shortlist-20261030/models/hf-eagle-4B/model.safetensors'
base=P/'runs/qwen35-correctness-matrix-20261101-01/models/Qwen3.5-4B-eagle3-sourcefaithful-BF16.gguf'
repair_path=P/'runs/qwen35-correctness-matrix-20261101-01/eagle-sourcefaithful-repair.json'
repair=read(repair_path);record=next(r for r in repair['records'] if r['size']=='4B')
assert pin(base)['sha256']==record['output_sha256']
assert shutil.disk_usage(R).free>3.2*1024**3
started=time.monotonic();phases={};arrays={};shapes={}
with safe_open(source,framework='pt',device='cpu') as state:
 for name in SOURCE_NAMES.values():
  weight=state.get_tensor(name);assert weight.ndim==2
  latent=weight.float().numpy();assert np.isfinite(latent).all()
  arrays[name+'.latent']=latent
  arrays[name+'.scale']=np.mean(np.abs(latent),axis=1,dtype=np.float32)
  shapes[name]=list(latent.shape)
phases['source_read_initialization_seconds']=time.monotonic()-started
npz=R/'initialization.npz';np.savez(npz,**arrays)
scratch_pin=pin(npz);del arrays
manifest={'schema_version':2,'base_gguf_sha256':record['output_sha256'],
 'checkpoint_sha256':scratch_pin['sha256'],'scale_layout':'row','activation_bits':8,
 'activation_rule':'a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive',
 'weight_rule':'hard_sign_zero_positive_clipped_identity_ste','qk_row_order':'original_checkpoint',
 'export_status':'row_w1ax_requires_native_validation','objective':'hard_ce',
 'projections':{b:{'checkpoint_name':n,'shape':shapes[n]} for b,n in SOURCE_NAMES.items()}}
mp=R/'initialization-manifest.json';write(mp,manifest)
freeze={'scope':'Deterministic zero-update hard-sign/row-meanabs initializer from retained BF16 source; NOT TRAINED and not native-loader/serving qualified',
 'training_updates':0,'training_seconds':0,'gpu_seconds':0,'planned_future_objective':'hard_ce (no optimization performed)',
 'source':pin(source),'base':pin(base),'repair_receipt':pin(repair_path),'recipe':pin(__file__),
 'exporter':pin(P/'scripts/export_recurrent_binary.py'),'auditor':pin(P/'scripts/audit_eagle_w1a1_gguf.py'),
 'initialization_scratch':scratch_pin,'initialization_scratch_retention':'Reproducible zero-update NPZ, retire only after independent source serialization audit; retained weights and recipe reconstruct it',
 'free_bytes_before_export':{p:shutil.disk_usage(p).free for p in ['.','/mnt/c']}}
write(R/'freeze.json',freeze)
stage=time.monotonic();report=export_model(base,npz,mp,R/'model.gguf');phases['export_seconds']=time.monotonic()-stage;write(R/'export.json',report)
stage=time.monotonic();independent=audit(source,R/'model.gguf',rows_per_chunk=128);phases['independent_audit_seconds']=time.monotonic()-stage;write(R/'independent-audit.json',independent)
assert independent['packed_tensors']==9 and report['serialization_audit_passed']
assert independent['qk_layout']['blk.0.attn_q']=={'heads':16,'head_dim':160,'rotary_dim':40}
# This file contains no trained/optimizer state. Its exact retained source and
# recipe reconstruct it; only own zero-update scratch is retired after proof.
assert pin(npz)==scratch_pin
write(R/'scratch-retirement.json',{'scratch':scratch_pin,'reconstructible_from':freeze['source'],'recipe':freeze['recipe'],'independent_audit':pin(R/'independent-audit.json'),'training_updates':0,'retirement_prepared':True})
npz.unlink()
result={'status':'CPU_SERIALIZATION_PASS','initialization_only':True,'training_updates':0,'training_seconds':0,'native_loaded':False,'serving_qualified':False,
 'model':pin(R/'model.gguf'),'export_report':pin(R/'export.json'),'independent_audit':pin(R/'independent-audit.json'),
 'audited_rows':independent['audited_rows'],'phases':phases,'wall_seconds':time.monotonic()-started,'scratch_retired':True,
 'free_bytes_after':{p:shutil.disk_usage(p).free for p in ['.','/mnt/c']}}
write(R/'result.json',result);print(json.dumps(result),flush=True)
