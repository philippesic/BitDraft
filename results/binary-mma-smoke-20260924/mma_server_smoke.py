import hashlib, json, os, pathlib, subprocess, time, urllib.request, urllib.error
root = pathlib.Path.cwd()
out = root / "results/binary-mma-smoke-20260924"
target = "models/gguf/Qwen3-4B-f16.gguf"
draft = "models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf"
port = 18089
cmd = [
    str(root / "build/llama-cuda-bmma-sm120-20260924/bin/llama-server"),
    "-m", str(root / target), "-md", str(root / draft),
    "--spec-type", "draft-eagle3", "--spec-draft-n-max", "5",
    "--spec-draft-ngl", "all", "--spec-draft-type-k", "f16",
    "--spec-draft-type-v", "f16", "--n-gpu-layers", "all", "--ctx-size", "2048",
    "--parallel", "1", "--fit", "off", "--cache-type-k", "f16",
    "--cache-type-v", "f16", "--jinja", "--metrics", "--perf", "-lv", "4",
    "--host", "127.0.0.1", "--port", str(port),
]
env = os.environ.copy()
env.update({"GGML_CUDA_W1A1_MMA": "1", "CUDA_VISIBLE_DEVICES": "0"})
def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
server_log = (out / "server.log").open("wb")
proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=server_log, stderr=subprocess.STDOUT)
try:
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before ready: {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError("server readiness timeout")
    req_path = out / "reference-request.json"
    req = json.loads(req_path.read_text())
    request_bytes = json.dumps(req, separators=(",", ":")).encode()
    request = urllib.request.Request(url + "/v1/chat/completions", data=request_bytes,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    (out / "mma-response.json").write_bytes(raw + b"\n")
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:400]!r}")
    got = json.loads(raw)
    ref = json.loads((out / "portable-production-response.json").read_text())
    a = (ref.get("__verbose") or {}).get("tokens")
    b = (got.get("__verbose") or {}).get("tokens")
    if not isinstance(a, list) or not isinstance(b, list):
        raise RuntimeError("missing raw token arrays")
    n = min(len(a), len(b))
    idx = next((i for i in range(n) if a[i] != b[i]), None)
    summary = {
        "variant": "packed_head_w1a1",
        "dispatch_env": env["GGML_CUDA_W1A1_MMA"],
        "server_command": cmd,
        "request_path": str(req_path),
        "request_sha256": sha(req_path),
        "reference_response_path": str(out / "portable-production-response.json"),
        "reference_response_sha256": sha(out / "portable-production-response.json"),
        "mma_response_path": str(out / "mma-response.json"),
        "mma_response_sha256": sha(out / "mma-response.json"),
        "portable_reference_code": "llama.cpp 92bc70602e13214d6db94c007894261b51f5f36c",
        "mma_code": subprocess.check_output(["git", "-C", "third_party/llama.cpp", "rev-parse", "HEAD"], text=True).strip(),
        "target_path": target,
        "draft_path": draft,
        "prompt_id": "prose-04",
        "status": status,
        "generated_tokens": len(b),
        "reference_tokens": len(a),
        "exact_token_match": a == b,
        "first_difference": None if idx is None else {"index": idx, "portable_id": a[idx], "mma_id": b[idx]},
        "finish_reason": (got.get("choices") or [{}])[0].get("finish_reason"),
        "gpu_snapshot": subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    server_log.close()
