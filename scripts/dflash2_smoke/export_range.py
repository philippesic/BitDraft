"""Declared inference precision fallback for finite, very wide DFlash2 FFN inputs."""
import numpy as np
from safetensors import safe_open
from scripts.dflash_baseline.model import GGUFReader
from gguf import GGUFWriter, GGMLQuantizationType as Type
from scripts.component_campaign.common import pin, save
from scripts.dflash2_smoke.train import R


def main():
    source=R/'models/dflash2-4b-Q4_K_M.gguf';weights=R/'models/dflash2-4b/model.safetensors'
    destination=R/'models/dflash2-4b-Q4-F32down.gguf'
    if destination.exists():raise FileExistsError(destination)
    reader=GGUFReader(source);writer=GGUFWriter(destination,'dflash')
    for key,field in reader.fields.items():
        if key.startswith('GGUF.') or key=='general.architecture':continue
        writer.add_key_value(key,field.contents(),field.types[0],field.types[-1] if len(field.types)>1 else None)
    writer.add_string('dflash.smoke.down_precision','F32 original BF16 values; native F32 input arithmetic to avoid legacy F16/Q8_1 range overflow')
    exceptions=[]
    with safe_open(weights,framework='pt',device='cpu') as original:
        for tensor in reader.tensors:
            if tensor.name.startswith('blk.') and tensor.name.endswith('.ffn_down.weight'):
                layer=int(tensor.name.split('.')[1]);value=original.get_tensor(f'layers.{layer}.mlp.down_proj.weight').float().numpy()
                assert list(value.shape)==list(tensor.shape[::-1]) and np.isfinite(value).all()
                writer.add_tensor(tensor.name,value,raw_dtype=Type.F32)
                exceptions.append(dict(name=tensor.name,source_type=tensor.tensor_type.name,execution_type='F32',bytes=value.nbytes))
            else:writer.add_tensor(tensor.name,tensor.data,raw_dtype=tensor.tensor_type)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    assert len(exceptions)==5
    save(destination.with_suffix('.fallback.json'),dict(status='EXPORTED_NOT_YET_QUALIFIED',
        original=pin(source),released_weights=pin(weights),model=pin(destination),exceptions=exceptions,
        unchanged='All other native tensor bytes and metadata preserved; no training or selector weights invented.',
        reason='Observed finite FFN GLU values~1.1e8 overflow legacy F16/Q8_1 down-input arithmetic; F32 down weight selects F32 input path.',source=pin(__file__)))
    print('F32_DOWN_FALLBACK_EXPORTED',flush=True)


if __name__=='__main__':main()
