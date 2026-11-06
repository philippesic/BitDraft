"""Read-only bridge to the campaign's admitted full48 native-A8 trainer.

Imports use the campaign's own source and guard/restore code. This is a new
finite branch; no historical bindings or paid checkpoint are edited.
"""
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
CAMPAIGN = PROJECT/"runs/w1a8-algorithms-20261020"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def configure_imports():
    source = CAMPAIGN/"training-source06/source"
    paths = [source/"src", source/"scripts", source/"third_party/llama.cpp/gguf-py"]
    paths += [CAMPAIGN/name for name in (
        "command-approx-kd20003", "command-sign-eal01", "command-coverage-train02",
        "command-loop2-answer-data01", "command-loop2-answer-interleave01",
        "command-loop2-firsterror-calib03")]
    sys.path[1:1] = list(map(str, paths))


def open_training_data():
    configure_imports()
    from cohort import open_answers
    from order import InterleavedTargets
    provider = open_answers()
    try:
        return provider, InterleavedTargets(CAMPAIGN/"loop2-answer-broad-data01", provider)
    except BaseException:
        provider.close()
        raise


def load_full_model(setup, parent, *, source_checkpoint=None):
    """parent pins exact checkpoint/model; same Adam, RNG and original guards.

    Returns model, optimizer, DeviceSignBaseline, metadata. Metadata contains
    original initial_norms, initial_sign_hashes, RNG and parent TRAIN_seconds.
    Existing restore authenticates zero-update GGUF == parent model.
    """
    configure_imports()
    scripts = str(PROJECT/"scripts")
    if scripts not in sys.path:
        sys.path.append(scripts)
    from shortlist_pilots.binary.parent_loader import load_full_model as restore_current
    return restore_current(setup, parent, source_checkpoint=source_checkpoint)


def native_export(model, output, *, parent=None):
    from joint_tools import export_joint
    donor = (parent or read(CAMPAIGN/"loop2-answer-broad01-p1/bindings.json")["parent"])["model"]
    return export_joint(model, donor["path"], donor["sha256"], output)


def select_pairs(targets, count):
    if not 1 <= count <= 729:
        raise ValueError("1..729 original fit pairs required")
    return [targets.pair(i) for i in range(count)]


def validate_spec(spec):
    if (spec.get("schema") != "shortlist_objective_pilot_v1"
            or spec["pilot"] not in ("control", "dpal", "vat", "edr")
            or not 0 < spec["max_train_seconds"] <= 840
            or not 0 < spec["max_updates"] <= 729
            or not 1 <= spec["pairs"] <= 729):
        raise ValueError("declared finite objective pilot required")
    if spec.get("synthetic") is not False:
        raise ValueError("authentic TRAIN only")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CPU-only prepare: no target/model/GPU loading")
    parser.add_argument("--pilot", choices=("control", "dpal", "vat", "edr"), required=True)
    parser.add_argument("--parent-binding", type=Path,
                        default=CAMPAIGN/"loop2-answer-broad01-p1/bindings.json")
    parser.add_argument("--parent-checkpoint", type=Path,
                        help="authenticated relocation; original binding hash remains required")
    parser.add_argument("--source-checkpoint", type=Path,
                        help="original sign-reference relocation; original hash remains required")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-seconds", type=float, default=120)
    parser.add_argument("--max-updates", type=int, default=72)
    parser.add_argument("--pairs", type=int, default=18)
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--edr-native-replay", type=Path)
    args = parser.parse_args()
    binding = read(args.parent_binding)
    parent = {k:dict(binding["parent"][k]) for k in ("checkpoint", "model")}
    if args.parent_checkpoint:
        if sha(args.parent_checkpoint) != parent["checkpoint"]["sha256"]:
            raise ValueError("relocated checkpoint differs from original bound parent")
        parent["checkpoint"]["path"] = str(args.parent_checkpoint.resolve())
    spec = {"schema": "shortlist_objective_pilot_v1", "pilot": args.pilot,
            "parent": parent,
            "parent_binding": pin(args.parent_binding), "max_train_seconds": args.max_train_seconds,
            "max_updates": args.max_updates, "pairs": args.pairs, "synthetic": False,
            "dpal_floor": .1, "vat_beta": .1, "edr_horizon": 12,"edr_cache_max_seconds":120,
            "teacher_manifest": pin(args.teacher_manifest) if args.teacher_manifest else None,
            "edr_native_replay": pin(args.edr_native_replay) if args.edr_native_replay else None}
    if args.source_checkpoint:
        spec["source_checkpoint"] = pin(args.source_checkpoint)
    validate_spec(spec)
    write(args.output, spec)


if __name__ == "__main__":
    main()
