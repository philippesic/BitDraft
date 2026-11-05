"""Stream the modern initializer into a packed zero-update artifact.

This is serialization only; native admission is deliberately a separate gate.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(Path('/home/philip/binary-eagle-decoding/third_party/llama.cpp/gguf-py'))]
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Type
from gguf.quants import dequantize
from export_recurrent_binary import pack, raw_hash, same_tensor, sha256


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    reader = GGUFReader(source)
    expected = {'dflash.block_count': 6, 'dflash.block_size': 16,
                'dflash.target_layers': [2,6,10,14,18,22,26,30],
                'dflash.feed_forward_length': 9216}
    for key, value in expected.items():
        if reader.fields[key].contents() != value:
            raise ValueError(key)
    selected = {f'blk.{i}.ffn_{p}.weight' for i in range(6) for p in ('gate','up','down')}
    if not selected <= {t.name for t in reader.tensors}:
        raise ValueError('missing FFN projection')
    temporary = destination.with_suffix('.partial.gguf')
    if temporary.exists():
        raise FileExistsError(temporary)
    writer = GGUFWriter(temporary, 'dflash')
    for key, field in reader.fields.items():
        if not key.startswith('GGUF.') and key != 'general.architecture':
            writer.add_key_value(key, field.contents(), field.types[0], field.types[-1] if len(field.types)>1 else None)
    writer.add_uint32('dflash.w1ax.version', 1)
    writer.add_uint32('dflash.w1ax.activation_bits', 8)
    for key, value in {'profile':'ffn18', 'bit_order':'little', 'sign_rule':'nonnegative_is_one', 'scale_rule':'f32_learned_nonnegative'}.items():
        writer.add_string('dflash.w1ax.'+key, value)
    writer.add_array('dflash.w1ax.tensors', sorted(selected))
    audit = {}
    for tensor in reader.tensors:
        if tensor.name not in selected:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            continue
        weight = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(tensor.shape[::-1])).astype(np.float32)
        scale = np.abs(weight).mean(axis=1)
        packed = pack(weight)
        base = tensor.name.removesuffix('.weight')
        writer.add_tensor(base+'.w1a1_packed', packed, raw_dtype=Type.I32)
        writer.add_tensor(base+'.w1a1_scale', scale, raw_dtype=Type.F32)
        audit[base] = {'packed_sha256':raw_hash(packed), 'scale_sha256':raw_hash(scale), 'shape':list(weight.shape)}
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    observed = {t.name:t for t in GGUFReader(temporary).tensors}
    for tensor in reader.tensors:
        if tensor.name not in selected and not same_tensor(tensor, observed[tensor.name]):
            raise ValueError('protected tensor changed: '+tensor.name)
    for base, item in audit.items():
        for suffix in ('packed','scale'):
            if raw_hash(observed[base+'.w1a1_'+suffix].data) != item[suffix+'_sha256']:
                raise ValueError('packed roundtrip mismatch')
    temporary.rename(destination)
    return {'source_sha256':sha256(source), 'output_sha256':sha256(destination),
            'output':str(destination), 'projections':audit, 'updates':0,
            'native_admission':False, 'preserved_precision':'original BF16 matrices/F32 norms; shared target head and embedding',
            'latent_initialization':'original BF16 values decoded to F32; sign >=0 positive; row mean absolute scales'}

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('output'); p.add_argument('audit')
    a=p.parse_args(); result=export(a.source,a.output)
    Path(a.audit).write_text(json.dumps(result,indent=2)+'\n')
