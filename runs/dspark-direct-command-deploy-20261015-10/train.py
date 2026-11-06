#!/usr/bin/env python3
"""Train or resume the configured DSpark run on its GPU host."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value in arguments for value in ("-h", "--help")):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
        parser.add_argument("--run-dir", type=Path)
        parser.add_argument(
            "--status", action="store_true", help="show saved progress without GPU queries"
        )
        parser.add_argument(
            "--check", action="store_true", help="check sources without GPU/model access"
        )
        parser.add_argument(
            "--detach", action="store_true", help="start supervised training and return"
        )
        parser.print_help()
        return 0
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
        options, _ = parser.parse_known_args(arguments)
        config = json.loads(options.config.read_text())
        interpreter = Path(config["python"])
        if not interpreter.is_file():
            raise SystemExit(
                f"Run this command on the GPU host; Python is configured at {interpreter}"
            )
        if os.path.abspath(sys.executable) != os.path.abspath(interpreter):
            os.execv(
                str(interpreter), [str(interpreter), str(Path(__file__).resolve()), *arguments]
            )
    from w1a1_eagle.direct_training_runtime import main as train

    return train(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
