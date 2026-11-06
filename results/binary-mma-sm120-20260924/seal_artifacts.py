import hashlib, json, pathlib, subprocess, time
root = pathlib.Path.cwd()
out = root / "results/binary-mma-sm120-20260924"
paths = [
 "results/binary-mma-sm120-20260924/final-process-gpu-state.txt",
 "results/binary-mma-sm120-20260924/restoration-cleanup.txt",
 "results/binary-mma-sm120-20260924/candidate-vs-production.patch",
 "results/binary-mma-sm120-20260924/toolchain.txt",
 "results/binary-mma-smoke-20260924/mma_server_smoke.py",
 "results/binary-mma-smoke-20260924/reference-request.json",
 "results/binary-mma-smoke-20260924/portable-production-response.json",
 "results/binary-mma-smoke-20260924/mma-response.json",
 "results/binary-mma-smoke-20260924/comparison.json",
 "results/binary-mma-smoke-20260924/server.log",
 "results/binary-mma-sm75-20260924/sm75-sass-full.txt",
 "results/binary-mma-sm75-20260924/binary-mma-sass-excerpt.txt",
 "results/native-eagle-5080-20260924/manifest.json",
 "configs/native_benchmark.toml",
 "configs/acceptance_prompts.jsonl",
 "third_party/llama.cpp/ggml/src/ggml-cuda/w1a1.cu",
 "third_party/llama.cpp/tests/test-backend-ops.cpp",
 "results/cuda-glibc-compat/include/crt/math_functions.h",
 "results/cuda-glibc-compat/include/cuda_fp4.hpp",
 "build/llama-cuda-bmma-sm120-20260924/CMakeCache.txt",
 "build/llama-cuda-bmma-sm75-20260924/CMakeCache.txt",
 "build/llama-cuda-bmma-sm120-20260924/bin/test-backend-ops",
 "build/llama-cuda-bmma-sm120-20260924/bin/llama-server",
 "build/llama-cuda-bmma-sm75-20260924/bin/libggml-cuda.so.0.25.1",
 "build/llama-cuda-bmma-sm75-20260924/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/w1a1.cu.o",
]
run_ids = [
 "binary-mma-sm120-build-20260924",
 "binary-mma-sm120-portable-20260924",
 "binary-mma-sm120-mma-20260924",
 "binary-mma-server-smoke-20260924",
 "binary-mma-sm75-build-20260924",
 "binary-mma-sm75-sass-20260924",
]
def file_info(rel):
    p = root / rel
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
files = {p: file_info(p) for p in paths}
runs = {}
for rid in run_ids:
    d = root / "runs" / rid
    state = json.loads((d / "state.json").read_text())
    runs[rid] = {
        "state": state,
        "stdout": {"path": str(d / "stdout.log"), **file_info(f"runs/{rid}/stdout.log")},
    }
manifest = {
 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "host": {"name": "DESKTOP-E78LJSI / WSL2", "address_from_shared_registry": "192.168.4.46", "ssh_user": "philip"},
 "hardware": {"gpu": "NVIDIA GeForce RTX 5080", "uuid": "GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e", "driver": "616.92", "cuda_umd": "13.4"},
 "code": {"llama_cpp_candidate": "9bb01a682ed4ba5e506870c8a38338589830b164", "candidate_branch": "feat/w1a1-sm75-mma", "production_base_and_restored_revision": "92bc70602e13214d6db94c007894261b51f5f36c", "parent_main_unchanged": True},
 "toolchain": {"nvcc": "CUDA 13.1.115", "cmake": "3.31.10", "ninja": "1.13.2.git.kitware.jobserver-pipe-1", "private_cuda_include_root": "results/cuda-glibc-compat/include"},
 "builds": {
   "sm120a": {"build_tree": "build/llama-cuda-bmma-sm120-20260924", "arch": "compute_120a/sm_120a", "result": "success"},
   "sm75_compile_only": {"build_tree": "build/llama-cuda-bmma-sm75-20260924", "arch": "compute_75/sm_75", "result": "success; binary never executed"}
 },
 "backend_correctness": {
   "operation": "W1A1_MUL_MAT",
   "portable_run": "binary-mma-sm120-portable-20260924",
   "portable_expected_and_passed_cases": 8,
   "portable_dispatch_log": "CUDA packed W1A1 portable XOR/POPCOUNT dispatch",
   "mma_run": "binary-mma-sm120-mma-20260924",
   "mma_expected_and_passed_cases": 8,
   "mma_dispatch_log": "CUDA packed W1A1 binary MMA dispatch (cc=1200)",
   "performance_claim": False
 },
 "packed_server_smoke": {
   "run": "binary-mma-server-smoke-20260924",
   "prompt": "prose-04",
   "request_settings": {"temperature": 0.0, "seed": 42, "max_tokens": 128, "cache_prompt": False, "enable_thinking": False, "target_weights_and_kv": "F16", "draft_head": "packed W1A1"},
   "result": "exact generated-token match to sealed production-portable response",
   "generated_token_ids_matched": 85,
   "finish_reason": "stop",
   "dispatch_log": "CUDA packed W1A1 binary MMA dispatch (K=2560, rows=32000, tokens=2, cc=1200)"
 },
 "sm75_sass": {
   "run": "binary-mma-sm75-sass-20260924",
   "instruction": "BMMA.88128.XOR.POPC",
   "present": True,
   "excerpt_path": "results/binary-mma-sm75-20260924/binary-mma-sass-excerpt.txt",
   "full_disassembly_path": "results/binary-mma-sm75-20260924/sm75-sass-full.txt",
   "full_disassembly_bytes": 1305594366,
   "binary_execution": "none"
 },
 "models": {
   "target": {"path": "models/gguf/Qwen3-4B-f16.gguf", "sha256": "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"},
   "packed_draft": {"path": "models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf", "sha256": "b2095130b5196574a9a08a88d2fb9a32ac1ea870ff3cf587ae7d6e7f64e7819f"},
   "hash_source": "existing completed native benchmark manifest"
 },
 "files": files,
 "runs": runs,
 "cleanup": {"supervisor_process_groups_terminal": True, "remote_submodule_restored": "92bc70602e13214d6db94c007894261b51f5f36c"}
}
(out / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"manifest": str(out / "artifact-manifest.json"), "file_count": len(files), "run_count": len(runs)}, sort_keys=True))
