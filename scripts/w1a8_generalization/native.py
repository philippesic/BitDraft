"""Bounded same-state native decision gate for the historical experiment.

This is not a full-request throughput measurement or an untouched release test.
It runs the existing native consumer and independently recomputes raw logits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from data import materialize, pin, read, write


def prepare(args):
    manifest = read(args.manifest)
    selected = []
    for domain in ("code", "prose", "reasoning"):
        for successful in (False, True):
            eligible = [w for w in manifest["windows"] if w["role"] == "development" and w["domain"] == domain
                        and (w["accepted_drafts"] == 5) == successful]
            eligible.sort(key=lambda w: hashlib.sha256((w["group_id"] + ":" + str(w["anchor"])).encode()).digest())
            if len(eligible) < 8:
                raise ValueError("eight development windows per domain/success stratum required")
            selected.extend(eligible[:8])
    subset = {**manifest, "windows": selected, "source_manifest": pin(args.manifest),
              "selection": "fixed hashed eight per domain x prior full5 success; development diagnostic, not occupancy estimate"}
    print(json.dumps(materialize(subset, args.output, max_bytes=768 * 1024**2)))


def evaluate(args):
    sys.path[1:1] = [str(args.project / "runs/5080-broader-training-20261031/source03/scripts"), str(args.project / "scripts")]
    from shortlist_pilots.objectives.runtime import configure_imports
    configure_imports()
    from training_system.native import evaluate as existing_evaluate, verify_raw
    source = Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031/parent-native-config01.json")
    config = read(source)
    config.update(model=pin(args.model), manifest=pin(args.manifest))
    result = existing_evaluate(config, args.output)
    verified = verify_raw(result)
    windows = read(args.manifest)["windows"]
    rows = []
    for window, record in zip(windows, result["records"]):
        rows.append({"group_id": window["group_id"], "anchor": window["anchor"], "domain": window["domain"],
                     "serving_parent_prefix5": window["accepted_drafts"],
                     "native_prefix5": min(5, record["first_rejection"]),
                     "native_predictions": [r["argmax"] for r in record["rows"][:5]]})
    report = {"status": "NATIVE_RAW_RECOMPUTED", "rows": rows, "raw_arrays": verified,
              "result": pin(args.output / "result.json"), "scope": "historical same-state development decision gate only"}
    if args.actor:
        actor = read(args.actor)
        index = {(r["group_id"], r["anchor"]): r for r in actor["rows"]}
        differences = []
        for row in rows:
            a = index[(row["group_id"], row["anchor"])]["prefix5"]
            differences.append(abs(a - row["native_prefix5"]))
            row["actor_prefix5"] = a
        report.update(actor_disagreements=sum(d != 0 for d in differences),
                      mean_absolute_prefix_error=sum(differences) / len(differences),
                      practical_gate_passed=sum(d != 0 for d in differences) / len(differences) <= 5 / 96
                      and sum(differences) / len(differences) <= .1,
                      practical_gate="retained bounded 5/96 decision disagreements and .1 mean prefix error; no bit-exact claim")
    write(args.output / "decision-gate.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "raw_arrays")}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--actor", type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
