"""Matched local development requests; final panels require a separate frozen plan."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path[:0] = [str(P), str(ROOT / "scripts")]
import compare_target_references as h
from benchmark_native_eagle import counter_delta, metrics_text

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--panel", type=Path, required=True)
parser.add_argument("--repetitions", type=int, default=5)
parser.add_argument("--prompt-count", type=int, default=24)
parser.add_argument("--variants", nargs="+", default=["conservative32", "quant4group"])
parser.add_argument("--models", nargs="+", default=list(h.MODELS))
parser.add_argument("--nine-b-mode", choices=["dflash", "mtp"], default="mtp")
parser.add_argument("--draft-max", type=int, default=16)
args = parser.parse_args()
assert 1 <= args.repetitions and 1 <= args.prompt_count <= 24
assert set(args.variants) <= {"conservative32", "attention2col", "quant4group"}
assert set(args.models) <= set(h.MODELS)
out = P / args.output
out.mkdir(exist_ok=False)
h.OUT = out
h.PORT = 18939
lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
h.check_pause()
assert h.available_port("127.0.0.1", h.PORT)
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
old = ROOT / "runs/spec-shortlist-20261030"
prompts = json.loads(args.panel.read_text())[:args.prompt_count]
assert len(prompts) == 24 and args.repetitions >= 5 and args.nine_b_mode == "mtp"
assert args.variants == ["conservative32", "quant4group"]
h.json_write(out / "prompts.json", prompts)
h.json_write(out / "plan.json", {"scope": "new frozen final confirmation panel", "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    "source": h.pin(__file__), "prompt_manifest": h.pin(out / "prompts.json"),
    "output_limit": 256, "temperature": 0, "seed": 1234, "context": 4096,
    "batch": 512, "ubatch": 512, "kv": "f16", "flash_attention": "on", "cache_prompt": False,
    "order": "alternating forward/reverse cell and prompt order; fresh server per cell/repetition",
    "warmups_per_cell": 2, "counts": "returned generation token IDs; full HTTP request wall excludes metrics reads"})
rows = []
started = time.perf_counter()
excluded_startup = 0.0
excluded_warmup = 0.0
for model in args.models:
    cells = [(variant, arm) for variant in args.variants for arm in ("target_only", "speculative")]
    for rep in range(args.repetitions):
        for variant, arm in cells if rep % 2 == 0 else reversed(cells):
            h.check_pause()
            cell = out / model / f"rep-{rep}-{variant}-{arm}"
            cell.mkdir(parents=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "DSPARK_", "DFLASH_", "EAGLE_", "W1AX_"))
                   and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
            env.update(GGML_CUDA_DECODE_CONSISTENT="32", CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
                       LD_LIBRARY_PATH=str(P / (variant + "-overlay")) + ":" + str(old / "build/bin"))
            cmd = [str(old / "build/bin/llama-server"), "-m", str(old / "models" / model),
                   "-ngl", "99", "-c", "4096", "-np", "1", "-b", "512", "-ub", "512", "-t", "4",
                   "-fa", "on", "--cache-type-k", "f16", "--cache-type-v", "f16", "--host", "127.0.0.1",
                   "--port", str(h.PORT), "--metrics", "--no-webui", "--no-warmup"]
            if arm == "target_only":
                cmd += ["--spec-type", "none"]
            elif "4B-" in model or "4b-" in model or args.nine_b_mode == "mtp":
                cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
            else:
                cmd += ["--spec-type", "draft-dflash", "--spec-draft-n-max", str(args.draft_max),
                        "-md", str(old / "models/qwen35-9b-dflash-Q4_K_M.gguf"), "-ngld", "99"]
            h.json_write(cell / "command.json", {"command": cmd, "environment": {k: v for k, v in env.items() if k.startswith(("GGML_", "CUDA_", "OMP_", "LD_"))}})
            with (cell / "server.log").open("w") as log:
                try:
                    begin = time.perf_counter()
                    h.server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    h.wait_ready(h.server, f"http://127.0.0.1:{h.PORT}", 180)
                    excluded_startup += time.perf_counter() - begin
                    maps = Path(f"/proc/{h.server.pid}/maps").read_text()
                    (cell / "loaded-library-maps.txt").write_text(maps)
                    loaded = sorted({line.split()[-1] for line in maps.splitlines() if "libggml-cuda.so" in line})
                    assert loaded == [str((P / (variant + "-overlay/libggml-cuda.so.0.28.0")).resolve())], loaded
                    (cell / "gpu.csv").write_text(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm", "--format=csv"], text=True))
                    for warmup in prompts[:2]:
                        _, elapsed = h.request({"prompt": warmup["prompt"], "n_predict": 256, "temperature": 0,
                                               "seed": 1234, "cache_prompt": False, "return_tokens": True})
                        excluded_warmup += elapsed
                    for prompt in prompts if rep % 2 == 0 else reversed(prompts):
                        h.check_pause()
                        before = metrics_text(f"http://127.0.0.1:{h.PORT}")
                        result, elapsed = h.request({"prompt": prompt["prompt"], "n_predict": 256, "temperature": 0,
                                                   "seed": 1234, "cache_prompt": False, "return_tokens": True, "stream": False})
                        after = metrics_text(f"http://127.0.0.1:{h.PORT}")
                        assert result["timings"]["cache_n"] == 0 and result["tokens"]
                        assert result["tokens_predicted"] == len(result["tokens"])
                        measured_counts = counter_delta(before, after)
                        assert all(type(v) is int and v >= 0 for v in measured_counts.values())
                        assert measured_counts["accepted"] <= measured_counts["proposed"]
                        if arm == "speculative":
                            assert measured_counts["accepted"] == result["timings"]["draft_n_accepted"]
                            assert measured_counts["proposed"] == result["timings"]["draft_n"]
                        row = {"model": model, "variant": variant, "arm": arm, "rep": rep,
                               "prompt_id": prompt["id"], "full_request_seconds": elapsed,
                               "output_tokens": len(result["tokens"]), "speculative": counter_delta(before, after), "result": result}
                        rows.append(row)
                        with (out / "requests.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                    print(json.dumps({"model": model, "variant": variant, "arm": arm, "rep": rep,
                                      "requests_complete": len(rows)}), flush=True)
                finally:
                    h.stop()
by = {(r["model"], r["variant"], r["arm"], r["rep"], r["prompt_id"]): r for r in rows}
comparisons = []
for model in args.models:
    for rep in range(args.repetitions):
        for prompt in prompts:
            reference = by[model, args.variants[0], "target_only", rep, prompt["id"]]["result"]["tokens"]
            for variant, arm in [(v, a) for v in args.variants for a in ("target_only", "speculative")]:
                tokens = by[model, variant, arm, rep, prompt["id"]]["result"]["tokens"]
                candidate_result = by[model, variant, arm, rep, prompt["id"]]["result"]
                reference_result = by[model, args.variants[0], "target_only", rep, prompt["id"]]["result"]
                first_repeat = by[model, variant, arm, 0, prompt["id"]]["result"]
                if (tokens != reference or candidate_result["content"] != reference_result["content"]
                        or tokens != first_repeat["tokens"]
                        or any(candidate_result.get(k) != reference_result.get(k) for k in ("stop_type", "stopping_word", "truncated"))):
                    comparisons.append({"model": model, "rep": rep, "prompt_id": prompt["id"], "variant": variant, "arm": arm,
                                        "lengths": [len(reference), len(tokens)]})
cells = {}
for model in args.models:
    cells[model] = {}
    for variant in args.variants:
        for arm in ("target_only", "speculative"):
            rr = [r for r in rows if (r["model"], r["variant"], r["arm"]) == (model, variant, arm)]
            count = sum(r["output_tokens"] for r in rr)
            wall = sum(r["full_request_seconds"] for r in rr)
            cells[model][variant + "/" + arm] = {"requests": len(rr), "tokens": count, "full_request_seconds": wall,
                "tokens_per_second": count / wall,
                "by_repetition": {str(rep): sum(r["output_tokens"] for r in rr if r["rep"] == rep) /
                    sum(r["full_request_seconds"] for r in rr if r["rep"] == rep) for rep in range(args.repetitions)},
                "speculative": {key: sum(r["speculative"][key] for r in rr) if all(r["speculative"][key] is not None for r in rr) else None
                                for key in ("accepted", "proposed", "rounds")}}
h.json_write(out / "summary.json", {"completed": True, "scope": "matched final confirmation, candidate/settings frozen before panel generation",
    "cells": cells, "token_mismatches": comparisons, "all_tokens_equal": not comparisons,
    "costs": {"training_seconds": 0, "evaluation_full_request_seconds": sum(r["full_request_seconds"] for r in rows),
              "excluded_server_startup_seconds": excluded_startup, "excluded_warmup_seconds": excluded_warmup,
              "total_wall_seconds": time.perf_counter() - started}})
