"""Authenticate the declared precision exceptions against original released tensors."""
import hashlib
import json
import numpy as np
from safetensors import safe_open
from scripts.dflash_baseline.model import GGUFReader
from gguf import GGMLQuantizationType as Type
from scripts.component_campaign.common import pin, save
from scripts.dflash2_smoke.train import R


def main():
    source=R/'models/dflash2-4b-Q4_K_M.gguf';fallback=R/'models/dflash2-4b-Q4-F32down.gguf'
    old=GGUFReader(source);new=GGUFReader(fallback);lookup={t.name:t for t in new.tensors}
    assert set(lookup)=={t.name for t in old.tensors}
    records=[]
    with safe_open(R/'models/dflash2-4b/model.safetensors',framework='pt',device='cpu') as released:
        for tensor in old.tensors:
            actual=lookup[tensor.name]
            exception=tensor.name.startswith('blk.') and tensor.name.endswith('.ffn_down.weight')
            if exception:
                layer=int(tensor.name.split('.')[1]);expected=released.get_tensor(f'layers.{layer}.mlp.down_proj.weight').float().numpy()
                assert actual.tensor_type==Type.F32 and np.array_equal(actual.data,expected)
            else:
                assert tensor.tensor_type==actual.tensor_type and np.array_equal(tensor.data,actual.data)
            records.append(dict(name=tensor.name,exception=exception,type=actual.tensor_type.name,
                                bytes=actual.data.nbytes,sha256=hashlib.sha256(actual.data).hexdigest()))
    for name,field in old.fields.items():
        if name.startswith('GGUF.'):continue
        assert field.contents()==new.fields[name].contents()
    assert sum(r['exception'] for r in records)==5
    save(R/'release-audit/fallback-byte-audit.json',dict(status='ALL_76_UNCHANGED_TENSORS_AND_5_EXACT_RELEASED_F32_DOWN_MATRICES_PASS',
        original=pin(source),fallback=pin(fallback),tensors=records,source=pin(__file__),
        semantic_change='Declared storage/native input arithmetic precision for FFN down only; no learned values or model capacity invented.'))
    print('FALLBACK_TENSOR_IDENTITY_PASS',len(records),flush=True)


if __name__=='__main__':main()
