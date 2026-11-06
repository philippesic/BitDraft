"""Independent, small development answer-quality pass; never a speed claim."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(P))
import compare_target_references as harness
import quality_judge

variant = sys.argv[1]
assert variant in ("original", "conservative32", "attention2col", "quant4group")
out = P / ("quality-" + variant + "-01")
out.mkdir(exist_ok=False)
harness.OUT = out
harness.PORT = 18938
lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
harness.check_pause()
assert harness.available_port("127.0.0.1", harness.PORT)
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
old = ROOT / "runs/spec-shortlist-20261030"
library_path = str(old / "build/bin")
if variant != "original":
    library_path = str(P / (variant + "-overlay")) + ":" + library_path
env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_") and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
env.update(LD_LIBRARY_PATH=library_path, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4")
if variant != "original":
    env["GGML_CUDA_DECODE_CONSISTENT"] = "32"
cases = quality_judge.cases()
harness.json_write(out / "quality-cases.json", cases)
harness.json_write(out / "judge-source.json", harness.pin(quality_judge.__file__))
rows = []
started = time.perf_counter()
for model in harness.MODELS:
    harness.check_pause()
    cell = out / model
    cell.mkdir()
    cmd = [str(old / "build/bin/llama-server"), "-m", str(old / "models" / model),
           "-ngl", "99", "-c", "4096", "-np", "1", "-b", "512", "-ub", "512", "-t", "4",
           "-fa", "on", "--cache-type-k", "f16", "--cache-type-v", "f16", "--host", "127.0.0.1",
           "--port", str(harness.PORT), "--spec-type", "none", "--no-webui", "--no-warmup"]
    harness.json_write(cell / "command.json", {"command": cmd, "variant": variant, "libraries": library_path})
    with (cell / "server.log").open("w") as log:
        try:
            harness.server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            harness.wait_ready(harness.server, f"http://127.0.0.1:{harness.PORT}", 180)
            maps = Path(f"/proc/{harness.server.pid}/maps").read_text()
            (cell / "loaded-library-maps.txt").write_text(maps)
            for case in cases:
                harness.check_pause()
                prompt = ("<|im_start|>system\nFollow the requested answer format precisely.<|im_end|>\n"
                          "<|im_start|>user\n" + case["prompt"] + "<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n\n</think>\n\n")
                body = {"prompt": prompt, "n_predict": 384, "temperature": 0, "seed": 1234,
                        "cache_prompt": False, "return_tokens": True, "stream": False}
                result, seconds = harness.request(body)
                assert result["timings"]["cache_n"] == 0
                judgement = quality_judge.judge(case, result["content"])
                row = {"model": model, "variant": variant, "case_id": case["id"], "result": result,
                       "judgement": judgement, "full_request_seconds": seconds}
                rows.append(row)
                with (out / "requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps({"model": model, "case_id": case["id"], "judgement": judgement}), flush=True)
        finally:
            harness.stop()
summary = {"completed": True, "variant": variant, "requests": len(rows),
           "by_model": {m: {"passed": sum(r["judgement"]["passed"] for r in rows if r["model"] == m),
                            "cases": sum(r["model"] == m for r in rows)} for m in harness.MODELS},
           "evaluation_full_request_seconds": sum(r["full_request_seconds"] for r in rows),
           "total_wall_seconds": time.perf_counter() - started, "training_seconds": 0,
           "scope": "independent development checks; no untouched final or performance acceptance"}
harness.json_write(out / "summary.json", summary)
