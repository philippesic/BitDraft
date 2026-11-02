"""Freeze the historical coverage x retention experiment; no GPU work."""
import argparse
import hashlib
import json
from pathlib import Path


def pin(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031/base-spec04.json")
    original = json.loads(base.read_text())
    policies = json.loads((args.policies / "prepared.json").read_text())["policies"]
    sources = [pin(p) for p in sorted(Path(__file__).parent.glob("*.py"))]
    arms = {}
    for policy in ("legacy_errors", "occupancy"):
        for coefficient in (0., .1):
            arm = policy + ("_retention" if coefficient else "_control")
            spec = {"schema": "w1a8_generalization_v1", "project": str(project),
                    "retained_source": str(project / "runs/5080-broader-training-20261031/source03"),
                    "base_spec": pin(base), "sources": sources,
                    "manifest": policies[policy]["manifest"], "order": policies[policy]["order"],
                    "policy": policy, "retention_coefficient": coefficient, "margin_cap": 2.,
                    "horizon_updates": 19200, "readout_milestones": [96, 480, 1920, 4800, 9600],
                    "fit_readout_stride": 10, "required_free_bytes": 7 * 1024**3,
                    "stop_path": str(output.parent / "STOP"),
                    "scientific_protocol": original["scientific_protocol"],
                    "precision": original["precision"],
                    "initialization": "exact inherited parent3504 Adam/RNG; original 48 parameters and 13 binary buffers",
                    "training_scope": "all captured serving TRAIN groups; inherited exposure is not unseen",
                    "first_stage_updates": 1920,
                    "stage_note": "Assistant-selected discriminating checkpoint, not a user time limit or a quick-screen rejection threshold",
                    "comparison": "2x2 data coverage / parent-prefix retention; same update count, group split, optimizer and native deployment",
                    "promotion": "native development serving and retained-prefix benefit, followed by new untouched confirmation; no fit-only promotion"}
            path = output / (arm + ".json")
            write(path, spec)
            arms[arm] = pin(path)
    write(output / "design.json", {"schema": "w1a8_generalization_design_v1", "arms": arms,
          "phase": 1, "first_stage_updates": 1920, "later_phases_gated": ["quick-screen qualification", "98-segment backlog screening"],
          "untouched_panel_generated": False, "modern_W1A8_candidate_admitted": False,
          "no_cross_target_performance_comparison": True,
          "practical_numeric_policy": "native target data plus bounded native decision/trajectory checks; record residual drift; investigate if it changes labels, decisions, gradients or conclusions",
          "costs": "record preparation, parent restoration/reference, optimizer work, diagnostics, checkpoint/export and native request timing separately"})
    print(json.dumps({"arms": list(arms), "first_stage_updates": 1920, "GPU_used": False}))


if __name__ == "__main__":
    main()
