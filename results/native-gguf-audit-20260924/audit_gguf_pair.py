import hashlib,json
from pathlib import Path
import numpy as np
import gguf
root=Path.cwd()
ordinary=root/"models/gguf/Qwen3-4B-eagle3-f16.gguf"
packed=root/"models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf"
def read(path):
    reader=gguf.GGUFReader(str(path))
    return reader,{t.name:t for t in reader.tensors}
ro,to=read(ordinary); rp,tp=read(packed)
def digest(t):
    a=np.asarray(t.data)
    return hashlib.sha256(a.tobytes()).hexdigest()
p=tp["output.w1a1_packed"];s=tp["output.w1a1_scale"]
shared=sorted(set(to)&set(tp))
changed=[]
for name in shared:
    a,b=to[name],tp[name]
    if a.tensor_type!=b.tensor_type or a.shape.tolist()!=b.shape.tolist() or not np.array_equal(a.data,b.data):
        changed.append(name)
report={
 "ordinary_file_sha256":hashlib.sha256(ordinary.read_bytes()).hexdigest(),
 "packed_file_sha256":hashlib.sha256(packed.read_bytes()).hexdigest(),
 "ordinary_tensor_count":len(to),"packed_tensor_count":len(tp),
 "shared_tensor_count":len(shared),"shared_tensor_mismatches":changed,
 "ordinary_output_weight_present":"output.weight" in to,
 "packed_output_weight_present":"output.weight" in tp,
 "packed_tensor":{"shape":p.shape.tolist(),"type":p.tensor_type.name,"data_sha256":digest(p),"matches_local_reference":"7e3c5642ec462e7a01691db84c4d25bbbd4a262977dc59794e441fc502a49b1b"==digest(p)},
 "scale_tensor":{"shape":s.shape.tolist(),"type":s.tensor_type.name,"data_sha256":digest(s),"matches_local_reference":"93c714eb6c21ce41ee27e8aeb6aad877f63e398a874167ce5f65f3a9cd2c56ea"==digest(s)},
}
out=root/"results/native-gguf-audit-20260924/gguf-pair-audit.json"
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
assert not changed and not report["ordinary_output_weight_present"] is False
assert not report["packed_output_weight_present"]

