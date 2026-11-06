#!/usr/bin/env python3
"""Generate a private DFlash decoded-binary graph overlay; no target-source writes."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("missing/nonunique binary graph anchor: " + old[:90])
    return source.replace(old, new, 1)

LOAD = '''    uint32_t binary_ref_version = 0;
    std::string binary_ref_kind, binary_ref_projection;
    int binary_ref_mode = 0;
    if (ml->get_key("dflash.binary.version", binary_ref_version, false)) {
        std::string arithmetic;
        ml->get_key("dflash.binary.kind", binary_ref_kind);
        ml->get_key("dflash.binary.projection", binary_ref_projection);
        ml->get_key("dflash.binary.arithmetic", arithmetic);
        if (binary_ref_version != 1 || !binary || binary_version != 4 ||
                dflash_w1ax_bits != 8 || weight_group || rotation || column_gains ||
                binary_ref_projection != "blk.4.ffn_up" ||
                arithmetic != "decoded_F32_A8_RNE_reference_v1")
            throw std::runtime_error("unsupported private binary reference contract");
        binary_ref_mode = binary_reference::parse_kind(binary_ref_kind);
    }
    for (const auto & item : ml->weights_map) {
        const auto at = item.first.find(".binary_");
        if (at != std::string::npos && (!binary_ref_mode || item.first.substr(0, at) != binary_ref_projection))
            throw std::runtime_error("undeclared binary reference tensor: " + item.first);
    }
'''

CUSTOM = '''        if (binary_ref_mode && base == binary_ref_projection) {
            if (ml->get_tensor_meta(name.c_str()) || ml->get_tensor_meta((base+".w1a1_packed").c_str()) ||
                    ml->get_tensor_meta((base+".w1a1_scale").c_str()))
                throw std::runtime_error("binary reference must exclusively replace packed/dense projection");
            auto load_ref = [&](const char * suffix, int64_t d0, int64_t d1) {
                const std::string key = base + "." + suffix;
                const auto * meta = ml->get_tensor_meta(key.c_str());
                if (!meta || meta->type != GGML_TYPE_F32 || meta->ne[0] != d0 || meta->ne[1] != d1 ||
                        meta->ne[2] != 1 || meta->ne[3] != 1)
                    throw std::runtime_error("binary reference F32 tensor mismatch: " + key);
                auto * value = create_tensor(LLM_TN_IMPL(LLM_ARCH_DFLASH, tensor, suffix, bid, -1), {d0, d1}, 0);
                eagle_w1ax_midpoints.emplace(key, value);
                return value;
            };
            const auto * signs_meta = ml->get_tensor_meta((base+".binary_signs").c_str());
            if (!signs_meta) throw std::runtime_error("missing binary decoded signs");
            const int64_t rank = signs_meta->ne[1];
            if (binary_ref_mode == binary_reference::factors && rank != 32 && rank != 1984)
                throw std::runtime_error("binary factor rank must be32 or1984");
            auto * signs = load_ref("binary_signs", k, binary_ref_mode == binary_reference::factors ? rank : rows);
            signs->op_params[0] = binary_ref_mode;
            load_ref("binary_scales", rows, 1);
            if (binary_ref_mode == binary_reference::factors) {
                load_ref("binary_factor_out", rank, rows);
                load_ref("binary_input_scale", k, 1);
            } else if (binary_ref_mode == binary_reference::mos || binary_ref_mode == binary_reference::mos_static) {
                load_ref("binary_router", k, 2);
                load_ref("binary_input_experts", 2, k);
                load_ref("binary_output_offsets", 2, rows);
            }
            dense = nullptr; packed = nullptr; scales = nullptr;
            LLAMA_LOG_INFO("%s: DECODED_BINARY_REFERENCE %s kind=%s F32; not packed-W1 performance\\n",
                           __func__, base.c_str(), binary_ref_kind.c_str());
            return;
        }
'''

def transform(source):
    source = once(source, '#include "models.h"', '#include "models.h"\n#include "../binary_reference.h"')
    anchor = '    auto load_linear = [&](llm_tensor tensor, int bid, int64_t k, int64_t rows,'
    source = once(source, anchor, LOAD + anchor)
    anchor = '        const auto * pm = ml->get_tensor_meta((base+".w1a1_packed").c_str());'
    source = once(source, anchor, CUSTOM + anchor)
    anchor = '            auto * up = layer.ffn_up_w1a1_packed\n'
    replacement = '''            const std::string binary_prefix = "blk." + std::to_string(il) + ".ffn_up";
            const auto binary_it = model.eagle_w1ax_midpoints.find(binary_prefix+".binary_signs");
            if (binary_it != model.eagle_w1ax_midpoints.end()) cb(cur, "binary_input", il);
            auto * up = binary_it != model.eagle_w1ax_midpoints.end()
                ? binary_reference::graph(ctx0, model.eagle_w1ax_midpoints, binary_prefix, cur, binary_it->second->op_params[0])
                : layer.ffn_up_w1a1_packed
'''
    return once(source, anchor, replacement)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rel = "src/models/dflash.cpp"
    original = (args.source/rel).read_text(); changed = transform(original)
    header = (HERE/"binary_reference.h").read_text()
    for name, text in [(rel, changed), ("src/binary_reference.h", header)]:
        dest=args.out/"overlay"/name;dest.parent.mkdir(parents=True, exist_ok=True);dest.write_text(text)
    patch = list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile="a/"+rel, tofile="b/"+rel))
    patch += list(difflib.unified_diff([], header.splitlines(True), fromfile="/dev/null", tofile="b/src/binary_reference.h"))
    (args.out/"binary-reference.patch").write_text("".join(patch))
    manifest={"base_source":str(args.source),"files":{rel:{"base_sha256":hashlib.sha256(original.encode()).hexdigest(),"output_sha256":hashlib.sha256(changed.encode()).hexdigest()},"src/binary_reference.h":{"output_sha256":hashlib.sha256(header.encode()).hexdigest()}},"native_built":False,"GPU_calls":0}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__ == "__main__": main()
