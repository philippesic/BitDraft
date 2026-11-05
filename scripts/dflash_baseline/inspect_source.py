"""Inspect released GGUF topology without loading model tensors onto a device."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'third_party/llama.cpp/gguf-py'))
from gguf import GGUFReader


def inspect(path):
    reader = GGUFReader(path)
    fields = {k: v.contents() for k, v in reader.fields.items()
              if k.startswith('dflash.') or k == 'general.architecture'}
    tensors = [{'name': t.name, 'shape': t.shape.tolist(), 'type': t.tensor_type.name}
               for t in reader.tensors]
    ffn = [t for t in tensors if any(t['name'].endswith(f'.ffn_{p}.weight')
                                   for p in ('gate', 'up', 'down'))]
    with open(path, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': digest, 'metadata': fields, 'ffn_projections': ffn,
            'tensors': tensors, 'native_admission': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = inspect(args.source)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
