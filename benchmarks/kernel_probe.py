import math
from pathlib import Path

from w1a1_eagle.numerical.gguf_readback import Reader


def extract_matrix(source: Path, name: str, output: Path) -> tuple[int, ...]:
    with source.open("rb") as stream:
        reader = Reader(stream, source.stat().st_size)
        _, count, alignment = reader.header(False)
        tensors = {}
        for _ in range(count):
            key = reader.text()
            dimensions = reader.uint(4)
            if not 1 <= dimensions <= 4:
                raise ValueError("unsupported tensor dimensions")
            shape = tuple(reader.uint(8) for _ in range(dimensions))
            tensors[key] = (shape, reader.uint(4), reader.uint(8))
        beginning = (stream.tell() + alignment - 1) // alignment * alignment
        shape, kind, offset = tensors[name]
        if kind not in {0, 26}:
            raise ValueError("probe requires F32 or packed I32 tensors")
        stream.seek(beginning + offset)
        output.write_bytes(reader.read(math.prod(shape) * 4))
        return shape


def runtime_kernels(source: Path) -> str:
    text = source.read_text()
    beginning = text.index("static __device__ float gained_value")
    end = text.index("void ggml_cuda_w1ax_pack(")
    return text[beginning:end]
