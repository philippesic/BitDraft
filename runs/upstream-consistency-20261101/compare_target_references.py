"""Exploratory original-versus-conservative target-only characterization."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_native_eagle import available_port, json_write, sha256, wait_ready

OLD = ROOT / "runs/spec-shortlist-20261030"
FIX = ROOT / "runs/spec-cuda-fix-20261031"
OUT = P / "target-reference-02"
CONTROL = Path.home() / ".config/binary-eagle-decoding/gpu-control.json"
MODELS = ("Qwen3.5-4B-Q4_K_M.gguf", "qwen35-4b-mtp-nvfp4.gguf",
          "Qwen3.5-9B-Q4_K_M.gguf", "qwen35-9b-mtp-nvfp4.gguf")
PORT = 18936
server = None


def check_pause():
    state = json.loads(CONTROL.read_text())["rtx5080"]
    if state.get("pause_requested") or state.get("stop_all_requested") or (OUT / "STOP").exists():
        raise InterruptedError("local pause or stop requested")


def request(body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=data,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        raw = response.read()
    return json.loads(raw), time.perf_counter() - start


def stop():
    global server
    if server is None:
        return
    if server.poll() is None:
        server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)
    server = None


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    started = time.time()
    OUT.mkdir(exist_ok=False)
    lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_pause()
    assert available_port("127.0.0.1", PORT), "port occupied"
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    assert not apps.strip(), "existing CUDA compute application"
    prompts = json.loads((FIX / "prompts.json").read_text())[:5]
    assert len(prompts) == 5
    for row in prompts:
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["sha256"]
    json_write(OUT / "prompts.json", prompts)
    variants = {
        "original": {"libraries": str(OLD / "build/bin"), "environment": {}},
        "conservative32": {"libraries": str(FIX / "patched-build/bin") + ":" + str(OLD / "build/bin"),
                           "environment": {"GGML_CUDA_DECODE_CONSISTENT": "32"}},
    }
    plan = {"scope": "exploratory development, target-only; no held-out/performance claim",
            "source": pin(__file__), "server": pin(OLD / "build/bin/llama-server"),
            "models": {name: pin(OLD / "models" / name) for name in MODELS},
            "variants": variants, "repetitions": 2, "prompt_count": 5,
            "order": "original,conservative then conservative,original for each model",
            "protocol": {"context": 4096, "batch": 512, "ubatch": 512, "kv": "f16",
                         "flash_attention": "on", "temperature": 0, "seed": 1234,
                         "output_limit": 256, "cache_prompt": False},
            "library_pins": {name: pin(path) for name, path in {
                "original_cuda": OLD / "build/bin/libggml-cuda.so.0.28.0",
                "conservative_cuda": FIX / "patched-build/bin/libggml-cuda.so.0.28.0"}.items()},
            "preparation_seconds": time.time() - started}
    json_write(OUT / "plan.json", plan)
    rows = []
    startup_seconds = 0.0
    warmup_seconds = 0.0
    global server
    try:
        for model in MODELS:
            for rep in range(2):
                order = list(variants) if rep == 0 else list(reversed(variants))
                for variant in order:
                    check_pause()
                    directory = OUT / model / f"{rep}-{variant}"
                    directory.mkdir(parents=True)
                    settings = variants[variant]
                    env = {k: v for k, v in os.environ.items()
                           if not k.startswith("GGML_") and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
                    env.update(settings["environment"], LD_LIBRARY_PATH=settings["libraries"],
                               CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4")
                    cmd = [str(OLD / "build/bin/llama-server"), "-m", str(OLD / "models" / model),
                           "-ngl", "99", "-c", "4096", "-np", "1", "-b", "512", "-ub", "512",
                           "-t", "4", "-fa", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
                           "--host", "127.0.0.1", "--port", str(PORT), "--spec-type", "none",
                           "--no-webui", "--no-warmup"]
                    json_write(directory / "command.json", {"command": cmd, "variant": settings})
                    with (directory / "server.log").open("w") as log:
                        try:
                            start = time.perf_counter()
                            server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                            wait_ready(server, f"http://127.0.0.1:{PORT}", 180)
                            startup_seconds += time.perf_counter() - start
                            maps = Path(f"/proc/{server.pid}/maps").read_text()
                            loaded = sorted({line.split()[-1] for line in maps.splitlines() if "libggml-cuda.so" in line})
                            expected = plan["library_pins"]["original_cuda" if variant == "original" else "conservative_cuda"]
                            assert loaded == [expected["path"]], (loaded, expected)
                            json_write(directory / "loaded-cuda.json", {"files": loaded, "sha256": sha256(Path(loaded[0]))})
                            warm, elapsed = request({"prompt": "Say hello.", "n_predict": 32, "temperature": 0,
                                                     "seed": 1234, "cache_prompt": False, "return_tokens": True})
                            warmup_seconds += elapsed
                            json_write(directory / "warmup.json", {"elapsed": elapsed, "result": warm})
                            for prompt in prompts if rep == 0 else reversed(prompts):
                                check_pause()
                                body = {"prompt": prompt["prompt"], "n_predict": 256, "temperature": 0,
                                        "seed": 1234, "cache_prompt": False, "return_tokens": True, "stream": False}
                                result, elapsed = request(body)
                                assert result["timings"]["cache_n"] == 0
                                assert isinstance(result["tokens"], list) and result["tokens"]
                                row = {"model": model, "variant": variant, "rep": rep,
                                       "prompt_id": prompt["id"], "elapsed": elapsed, "result": result}
                                rows.append(row)
                                with (OUT / "requests.jsonl").open("a") as stream:
                                    stream.write(json.dumps(row) + "\n")
                            print(json.dumps({"model": model, "variant": variant, "rep": rep,
                                              "requests_complete": len(rows)}), flush=True)
                        finally:
                            stop()
        by = {(r["model"], r["variant"], r["rep"], r["prompt_id"]): r for r in rows}
        comparisons = []
        for model in MODELS:
            pairs = []
            for rep in range(2):
                for p in prompts:
                    a = by[model, "original", rep, p["id"]]["result"]["tokens"]
                    b = by[model, "conservative32", rep, p["id"]]["result"]["tokens"]
                    first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
                    if first is None and len(a) != len(b):
                        first = min(len(a), len(b))
                    pairs.append({"rep": rep, "prompt_id": p["id"], "exact": a == b,
                                  "first_divergence": first, "lengths": [len(a), len(b)]})
            comparisons.append({"model": model, "exact": sum(p["exact"] for p in pairs),
                                "comparisons": len(pairs), "pairs": pairs})
        repeats = {model: {v: all(by[model, v, 0, p["id"]]["result"]["tokens"] ==
                                      by[model, v, 1, p["id"]]["result"]["tokens"] for p in prompts)
                           for v in variants} for model in MODELS}
        json_write(OUT / "summary.json", {"completed": True, "comparisons": comparisons,
                   "deterministic_repeats": repeats, "requests": len(rows),
                   "costs": {"training_seconds": 0, "preparation_seconds": plan["preparation_seconds"],
                             "server_startup_seconds": startup_seconds, "excluded_warmup_seconds": warmup_seconds,
                             "evaluation_full_request_seconds": sum(r["elapsed"] for r in rows),
                             "total_elapsed_seconds": time.time() - started},
                   "scope": "exploratory target-only reference characterization, no quality or speed acceptance"})
    finally:
        stop()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
