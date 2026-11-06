import hashlib,json,sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/"third_party/llama.cpp"),str(ROOT/"third_party/llama.cpp/gguf-py")]
import gguf
from conversion.llama import pack_w1a1_head
model_path=ROOT/"models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf"
source_path=ROOT/"models/hf/Qwen3-4B_eagle3/model.safetensors"
reader=gguf.GGUFReader(str(model_path))
tensors={t.name:t for t in reader.tensors}
with safe_open(source_path,framework="pt",device="cpu") as f:
    weight=f.get_tensor("lm_head.weight")
expected_p,expected_s=pack_w1a1_head(weight)
packed=tensors["output.w1a1_packed"].data
scale=tensors["output.w1a1_scale"].data
packed_match=None
for candidate in (expected_p,expected_p.T):
    if candidate.shape==packed.shape and np.array_equal(candidate.view(np.uint32),np.asarray(packed).view(np.uint32)):
        packed_match="direct" if candidate is expected_p else "transpose"
        break
scale_match=scale.shape==expected_s.shape and np.array_equal(scale,expected_s)
meta={}
for name,field in reader.fields.items():
    if name.startswith("eagle3.w1a1_head."):
        try: value=field.contents()
        except Exception: value=str(field.data)
        if isinstance(value,bytes): value=value.decode(errors="replace")
        meta[name]=value
summary={"gguf_path":str(model_path),"gguf_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),"source_head_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),"source_head_shape":list(weight.shape),"source_head_dtype":str(weight.dtype),"packed_shape":list(packed.shape),"packed_dtype":str(packed.dtype),"packed_type":tensors["output.w1a1_packed"].tensor_type.name,"scale_shape":list(scale.shape),"scale_dtype":str(scale.dtype),"scale_type":tensors["output.w1a1_scale"].tensor_type.name,"packed_matches_source":packed_match,"scales_exact_match_source":bool(scale_match),"dense_output_weight_present":"output.weight" in tensors,"packed_metadata":meta,"tensor_names":sorted(tensors)}
out=ROOT/"results/native-gguf-audit-20260924/packed-head-audit.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(summary,indent=2,default=str)+"\n")
print(json.dumps(summary,indent=2,default=str))
assert packed_match is not None and scale_match and not summary["dense_output_weight_present"]

