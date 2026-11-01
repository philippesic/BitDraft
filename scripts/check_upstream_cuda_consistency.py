#!/usr/bin/env python3
"""Check the issue-25618 source candidates without CUDA execution or model loads."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


PIN = "5eb221ec53cb4317aef963e3bdec3cc69401eca4"
REPORT = Path(__file__).resolve().parents[1] / "experiments/upstream-cuda-consistency-2026-11-01"


def run(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args!r}\n{result.stdout}{result.stderr}")
    return result


def function(source, name):
    start = source.index(name + "(")
    start = source.rfind("\n", 0, start) + 1
    end = source.index("{", start)
    depth = 1
    while depth:
        end += 1
        depth += (source[end] == "{") - (source[end] == "}")
    return source[start:end + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    names = ["F16", "F32", "BF16", "Q4_0", "Q8_0", "I32", "Q4_K", "Q5_K", "Q6_K", "MXFP4", "NVFP4"]
    paths = ["ggml/src/ggml-cuda/" + name for name in
             ["mmvf.cu", "fattn.cu", "fattn-common.cuh", "fattn-vec.cuh", "ggml-cuda.cu", "mmq.cu"]]
    original = {p: run(["git", "-C", str(args.source_repo), "show", PIN + ":" + p]).stdout for p in paths}
    with tempfile.TemporaryDirectory(prefix="eagle-cuda-source-check-") as scratch:
        root = Path(scratch)
        for path, content in original.items():
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        patches = sorted((REPORT / "reference-ports").glob("*.patch")) + sorted((REPORT / "patches").glob("*.patch"))
        assert len(patches) == 3
        for patch in patches:
            run(["git", "apply", "--check", "--whitespace=error", str(patch)], cwd=root)
            run(["git", "apply", str(patch)], cwd=root)
        patched = {p: (root / p).read_text() for p in paths}
        old_select = function(original[paths[0]], "ggml_cuda_should_use_mmvf").replace(
            "ggml_cuda_should_use_mmvf", "original_select")
        new_select = function(patched[paths[0]], "ggml_cuda_should_use_mmvf").replace(
            "ggml_cuda_should_use_mmvf", "candidate_select")
        old_audit = function(original[paths[4]], "ggml_cuda_matmul_audit").replace(
            "ggml_cuda_matmul_audit", "original_audit")
        new_audit = function(patched[paths[4]], "ggml_cuda_matmul_audit").replace(
            "ggml_cuda_matmul_audit", "candidate_audit")
        mmq = patched[paths[5]]
        label_start = mmq.index("    const bool use_native_fp4 =")
        label_end = mmq.index("    const size_t y_block_size", label_start)
        labels = mmq[label_start:label_end].replace("ggml_cuda_matmul_audit", "record")
        partitions = []
        for prefix, contents in [("original", original[paths[2]]), ("candidate", patched[paths[2]])]:
            start = contents.index("        // parallel_blocks must" if prefix == "original" else
                                   "        // Keep the reduction order stable")
            end = contents.index("        blocks_num.x = ntiles_x;", start)
            partitions.append(f"int {prefix}_partition(int cc, int nq, int nkv, int heads, int nsm, "
                              "int max_blocks_per_sm, bool mixed) {\n"
                              "ggml_tensor query{GGML_TYPE_F32, nullptr, {256, nq, 16, 1}};\n"
                              "ggml_tensor key{GGML_TYPE_Q8_0, nullptr, {256, nkv, 4, 1}};\n"
                              "const ggml_tensor * Q = &query, * K = &key, * V = &key;\n"
                              "const bool need_f16_K = mixed, need_f16_V = mixed;\n"
                              "(void) Q; (void) K; (void) V; (void) need_f16_K; (void) need_f16_V; (void) cc;\n"
                              "int parallel_blocks = max_blocks_per_sm;\n"
                              "const int ntiles_KV = (nkv + 255)/256;\n"
                              "const int ntiles_dst = nq * heads;\n" + contents[start:end] +
                              "return parallel_blocks;\n}\n")
        header = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#define GGML_MAX_DIMS 4
#define GGML_CUDA_CC_TURING 750
#define GGML_CUDA_CC_ADA_LOVELACE 890
#define GGML_CUDA_CC_IS_NVIDIA(cc) ((cc) < 10000)
#define GGML_CUDA_CC_IS_AMD(cc) ((cc) >= 10000)
#define GGML_CUDA_CC_IS_RDNA3(cc) ((cc) == 11000)
#define GGML_CUDA_CC_IS_RDNA4(cc) ((cc) == 12000)
'''
        header += "enum ggml_type {" + ",".join("GGML_TYPE_" + n for n in names) + "};\n"
        header += "const char * type_names[] = {" + ",".join(json.dumps(n) for n in names) + "};\n"
        header += r'''
const char * ggml_type_name(ggml_type t) { return type_names[t]; }
size_t ggml_type_size(ggml_type t) { return t == GGML_TYPE_F32 ? 4 : 2; }
int compiled_arch;
bool ampere_mma_available(int cc) { return cc < 10000 && compiled_arch >= 800; }
bool fp16_mma_hardware_available(int cc) { return cc < 10000 && cc >= 700; }
bool bf16_mma_hardware_available(int cc) { return cc < 10000 && cc >= 800; }
bool fp32_mma_hardware_available(int) { return false; }
bool blackwell_mma_available(int cc) { return cc < 10000 && compiled_arch >= 1200 && compiled_arch < 1300; }
bool ggml_is_quantized(ggml_type t) { return t >= GGML_TYPE_Q4_0 && t != GGML_TYPE_I32; }
struct ggml_tensor { ggml_type type; const char * name; int64_t ne[4]; };
struct ggml_backend_cuda_context { int device; };
std::string operand, accumulation, conversion;
void record(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *,
            const char *, const char * a, const char * b, const char * c) {
    operand = a; accumulation = b; conversion = c;
}
void check_labels(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, int cc) {
'''
        source = header + labels + "}\n" + old_select + "\n" + new_select + "\n"
        # The actual call sites pass a default null fused gate in common.cuh.
        source += old_audit + "\n" + new_audit + "\n" + "\n".join(partitions) + r'''
int main() {
    long checks = 0, changes = 0;
    for (int cc : {610, 700, 750, 800, 860, 890, 1200, 1210, 11000, 12000}) {
        for (int arch : {750, 800, 890, 1200, 1210}) {
            compiled_arch = std::min(cc, arch);
            for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16}) {
                for (int width = 1; width <= 9; ++width) {
                    for (int layout = 0; layout < 5; ++layout) {
                        int64_t ne[4] = {2560, 32, 1, 1};
                        size_t ts = ggml_type_size(type);
                        size_t nb[4] = {ts, 2560*ts, 81920*ts, 81920*ts};
                        if (layout == 1) ne[0] = 2559;
                        if (layout == 2) nb[0] = 2*ts;
                        if (layout == 3) nb[2] += ts;
                        if (layout == 4) { ne[1] = 1024; ne[2] = 4; }
                        bool a = original_select(type, cc, ne, nb, width);
                        bool b = candidate_select(type, cc, ne, nb, width);
                        bool expected_change = type == GGML_TYPE_F16 && cc < 10000 && cc >= 890 &&
                            compiled_arch >= 800 && width >= 2 && width <= 4 && layout == 0;
                        assert((a != b) == expected_change);
                        if (expected_change) assert(!a && b);
                        changes += a != b;
                        ++checks;
                    }
                }
            }
        }
    }
    ggml_backend_cuda_context ctx{0};
    ggml_tensor x{GGML_TYPE_F32, "x", {2560, 4, 1, 1}};
    ggml_tensor w{GGML_TYPE_F32, "blk.0.ssm_alpha.weight", {2560, 32, 1, 1}};
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
        w.type = type;
        original_audit(ctx, &w, &x, "MMVQ", "Q8_1", "I32_dot_F32_scaled_sum", "separate_Q8_1_pack", nullptr);
        candidate_audit(ctx, &w, &x, "MMVQ", "Q8_1", "I32_dot_F32_scaled_sum", "separate_Q8_1_pack", nullptr);
        candidate_audit(ctx, &w, &x, "MMVQ", "Q8_1", "I32_dot_F32_scaled_sum", "separate_Q8_1_pack", nullptr);
    }
    for (int arch : {860, 1200, 1210, 1300}) {
        compiled_arch = arch;
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
            w.type = type;
            check_labels(ctx, &w, &x, arch);
            bool native = (arch == 1200 || arch == 1210) && type != GGML_TYPE_Q4_K;
            assert(operand == (native ? "FP4_E2M1" : "Q8_1"));
            assert(accumulation == (native ? "F32_MMA" : "I32_dot_F32_scaled_sum"));
            assert(conversion == (native ? (type == GGML_TYPE_NVFP4 ?
                "separate_MMQ_NVFP4_pack_F32_column_scale" : "separate_MMQ_MXFP4_pack") : "separate_MMQ_Q8_1_pack"));
        }
    }
    long partition_checks = 0;
    for (int cc : {860, 890, 1200, 1210}) {
        for (bool mixed : {false, true}) {
            for (int nq = 1; nq <= 8; ++nq) {
                for (int nkv : {256, 512, 2048, 7168, 16384, 65536}) {
                    for (int occupancy : {1, 2, 4, 8}) {
                        int a = original_partition(cc, nq, nkv, 4, 84, occupancy, mixed);
                        int b = candidate_partition(cc, nq, nkv, 4, 84, occupancy, mixed);
                        if (cc >= 890 && nq <= 4 && !mixed) assert(b == occupancy);
                        else assert(a == b);
                        ++partition_checks;
                    }
                }
            }
        }
    }
    printf("%ld %ld %ld\n", checks, changes, partition_checks);
}
'''
        cpp = root / "check.cpp"
        cpp.write_text(source)
        binary = root / "check"
        run([args.cxx, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)])
        result = run([str(binary)], env={"GGML_CUDA_MATMUL_AUDIT": "1"})
        records = [json.loads(line.split(" ", 1)[1]) for line in result.stderr.splitlines()]
        assert len(records) == 7, records  # One old record; six new records, each deduplicated.
        assert [r["weight_type"] for r in records] == ["F32", "F32", "Q4_K", "Q5_K", "Q6_K", "MXFP4", "NVFP4"]
        counts = result.stdout.strip().split()
        print(json.dumps({"source_commit": PIN, "patches_apply": len(patches),
                          "host_dispatch_cases": int(counts[0]), "expected_dispatch_changes": int(counts[1]),
                          "audit_records": len(records), "mmq_label_cases": 12,
                          "host_fa_partition_cases": int(counts[2]),
                          "cuda_compile": False, "gpu_execution": False}, sort_keys=True))


if __name__ == "__main__":
    main()
