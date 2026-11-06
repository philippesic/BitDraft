#!/usr/bin/env python3
"""Train or resume the configured DSpark run on its GPU host."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def bound_config_arguments(arguments, config_path):
    """Preserve the selected public config across an exec into another root."""
    forwarded = list(arguments)
    end = forwarded.index("--") if "--" in forwarded else len(forwarded)
    for index in range(end - 1, -1, -1):
        if forwarded[index] == "--config":
            forwarded[index + 1] = str(config_path)
            return forwarded
        if forwarded[index].startswith("--config="):
            forwarded[index] = "--config=" + str(config_path)
            return forwarded
    forwarded[end:end] = ["--config", str(config_path)]
    return forwarded


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--configure-unattended" in arguments:
        from w1a1_eagle.training_supervisor import configure_main

        return configure_main(arguments)
    control_arguments = arguments[:arguments.index("--")] if "--" in arguments else arguments
    actions = {value for value in control_arguments
               if value in {"--status", "--check", "--detach", "--stop"}}
    if len(actions) > 1:
        raise SystemExit("Choose one action: --status, --check, --detach, or --stop")
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
        parser.add_argument("--stop", action="store_true", help="request graceful checkpoint stop")
        parser.print_help()
        return 0
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
        options, _ = parser.parse_known_args(arguments)
        # A public selector may point at an immutable admitted configuration.
        # Its consumer authenticates the canonical locator and rejects symlinks.
        canonical_config = options.config.resolve(strict=True)
        if canonical_config != options.config.absolute():
            arguments = bound_config_arguments(arguments, canonical_config)
        options.config = canonical_config
        config = json.loads(options.config.read_text())
        # Inspection is a local metadata operation, not a launch into the
        # configured (possibly unavailable) host interpreter/private source.
        # Mutating actions still take the authenticated execution path below.
        inspection_module = ("training_supervisor" if config.get("schema") == "unattended_training_v1"
                             else "direct_training_runtime")
        local_reader = ROOT / "src" / "w1a1_eagle" / (inspection_module + ".py")
        # A dispatcher-only host has no local package. In that case the
        # authenticated private entry below supplies its own passive reader.
        if "--status" in actions and local_reader.is_file() and not local_reader.is_symlink():
            if config.get("schema") == "unattended_training_v1":
                from w1a1_eagle.training_supervisor import main as inspect
            else:
                from w1a1_eagle.direct_training_runtime import main as inspect
            return inspect(arguments)
        interpreter = Path(config["python"])
        if not interpreter.is_file():
            raise SystemExit(
                f"Run this command on the GPU host; Python is configured at {interpreter}"
            )
        selected_root = config.get("source_root")
        if config.get("schema") == "unattended_training_v1" and selected_root is not None:
            source = Path(selected_root)
            if source != ROOT:
                entry = source / "train.py"
                try:
                    if (not source.is_absolute() or source != source.resolve(strict=True)
                            or not source.is_dir() or entry.is_symlink()
                            or not entry.is_file() or entry != entry.resolve(strict=True)
                            or entry.samefile(Path(__file__))):
                        raise ValueError("canonical separate source root and entry required")
                    if hashlib.sha256(entry.read_bytes()).hexdigest() != config.get("source_files", {}).get("train.py"):
                        raise ValueError("entry SHA differs from operational source pin")
                except (OSError, ValueError) as error:
                    raise SystemExit("Invalid pinned private training entry: " + str(error)) from error
                forwarded = bound_config_arguments(arguments, canonical_config)
                os.execv(str(interpreter), [str(interpreter), str(entry), *forwarded])
        if os.path.abspath(sys.executable) != os.path.abspath(interpreter):
            os.execv(
                str(interpreter), [str(interpreter), str(Path(__file__).resolve()), *arguments]
            )
    if config.get("schema") == "unattended_training_v1":
        from w1a1_eagle.training_supervisor import main as train
    else:
        from w1a1_eagle.direct_training_runtime import main as train

    return train(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
