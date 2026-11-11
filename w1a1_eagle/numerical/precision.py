import argparse
import json
from pathlib import Path

import torch

from w1a1_eagle.contracts import Record
from w1a1_eagle.data import Artifact


def screen(source: Artifact, device: str) -> Record:
    source.verify()
    if source.size == 0 or source.size > 32 * 1024 * 1024 or source.size % 4:
        raise ValueError("nonempty F32 payload at most 32 MiB required")
    if device not in {"cpu", "cuda"}:
        raise ValueError("explicit cpu/cuda device required")
    values = torch.frombuffer(bytearray(source.path.read_bytes()), dtype=torch.float32).to(device)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite reference features")
    rounded = values.half().float()
    finite = bool(torch.isfinite(rounded).all())
    absolute = (rounded - values).abs()
    denominator = float(values.double().square().sum().sqrt())
    return {
        "schema": 1,
        "source_sha256": source.sha256,
        "source_bytes": source.size,
        "values": values.numel(),
        "device": str(values.device),
        "torch": str(torch.__version__),
        "finite_roundtrip": finite,
        "overflow_values": int((~torch.isfinite(rounded)).sum()),
        "zeroed_nonzero_values": int(((rounded == 0) & (values != 0)).sum()),
        "max_absolute_error": float(absolute.max()) if finite else None,
        "relative_l2_error": float((rounded - values).double().square().sum().sqrt()) / denominator
        if finite and denominator
        else None,
        "max_reference_magnitude": float(values.abs().max()),
        "qualification": "numeric_screen_only",
        "learning_qualified": False,
        "native_acceptance_qualified": False,
    }


class Arguments(argparse.Namespace):
    path: Path
    sha256: str
    bytes: int
    device: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(namespace=Arguments())
    print(
        json.dumps(
            screen(Artifact(args.path, args.sha256, args.bytes), args.device), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
