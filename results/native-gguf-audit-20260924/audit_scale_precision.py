import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
R=Path.cwd();sys.path.insert(0,str(R/"third_party/llama.cpp/gguf-py"))
import gguf
gg=gguf.GGUFReader(str(R/"models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf"))
t=next(x for x in gg.tensors if x.name=="output.w1a1_scale")
actual=np.asarray(t.data,dtype=np.float32)
with safe_open(R/"models/hf/Qwen3-4B_eagle3/model.safetensors",framework="pt",device="cpu") as f:
    w=f.get_tensor("lm_head.weight").float().abs()
    f32=w.mean(dim=1).numpy()
    f64=w.double().mean(dim=1).float().numpy()
diff=np.abs(f32.view(np.int32).astype(np.int64)-f64.view(np.int32).astype(np.int64))
adiff=np.abs(f32.astype(np.float64)-f64.astype(np.float64))
summary={"rows":len(f32),"gguf_scale_sha256":hashlib.sha256(actual.tobytes()).hexdigest(),"remote_f32_mean_sha256":hashlib.sha256(f32.tobytes()).hexdigest(),"f64_accumulate_round_f32_sha256":hashlib.sha256(f64.tobytes()).hexdigest(),"gguf_equals_remote_f32_mean":bool(np.array_equal(actual,f32)),"rows_f32_vs_f64_different":int(np.count_nonzero(diff)),"max_ulp_difference":int(diff.max()),"max_abs_difference":float(adiff.max()),"local_mps_scale_sha256_reference":"93c714eb6c21ce41ee27e8aeb6aad877f63e398a874167ce5f65f3a9cd2c56ea"}
p=R/"results/native-gguf-audit-20260924/scale-precision-audit.json";p.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
assert summary["gguf_equals_remote_f32_mean"]

