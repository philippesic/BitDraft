#!/usr/bin/env python3
"""Generate isolated native vocab graph/loader patch against Root's v2 source."""
import argparse,difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
NAMES=('VOCAB_ROUTER_DOWN','VOCAB_ROUTER_UP','VOCAB_STATIC_HEAD','VOCAB_STATIC_IDS')

def once(s,a,b):
    if s.count(a)!=1:raise ValueError('missing/nonunique anchor: '+a[:100])
    return s.replace(a,b,1)

def transform(rel,s):
    if rel=='src/llama-arch.h':
        anchor='    LLM_TENSOR_OUTPUT_FACTOR_UP,\n'
        return once(s,anchor,anchor+''.join('    LLM_TENSOR_'+n+',\n' for n in NAMES))
    if rel=='src/llama-arch.cpp':
        anchor='    { LLM_TENSOR_OUTPUT_FACTOR_UP,                       "output.factor_up" },'
        s=once(s,anchor,anchor+'\n'+''.join('    { LLM_TENSOR_'+n+', "'+n.lower()+'" },\n' for n in NAMES).rstrip())
        anchor='    {LLM_TENSOR_OUTPUT_FACTOR_UP,           {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},'
        return once(s,anchor,anchor+'\n'+''.join('    {LLM_TENSOR_'+n+', {LLM_TENSOR_LAYER_OUTPUT, '+('GGML_OP_GET_ROWS' if n.endswith('_IDS') else 'GGML_OP_MUL_MAT')+'}},\n' for n in NAMES).rstrip())
    if rel=='src/llama-model.h':
        anchor='    struct ggml_tensor * output_factor_up   = nullptr;'
        return once(s,anchor,anchor+'\n'+''.join('    struct ggml_tensor * '+n.lower()+' = nullptr;\n' for n in NAMES).rstrip())
    if rel=='src/llama-context.cpp':
        anchor='(model.arch == LLM_ARCH_DFLASH && model.output_factor_down && model.output_factor_up)'
        return once(s,anchor,anchor+' || (model.arch == LLM_ARCH_DFLASH && model.vocab_static_head && model.vocab_static_ids)')
    if rel=='common/speculative.cpp':
        anchor='(draft->output || (draft->output_factor_down && draft->output_factor_up))'
        return once(s,anchor,'(draft->output || (draft->output_factor_down && draft->output_factor_up) || (draft->vocab_static_head && draft->vocab_static_ids))')
    if rel=='src/models/dflash.cpp':
        s=once(s,'#include "llama-kv-cache-iswa.h"','#include "llama-kv-cache-iswa.h"\n#include "vocab_head.h"')
        anchor='    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);'
        loader='''    const auto * vocab_down=ml->get_tensor_meta("vocab_router_down.weight");
    const auto * vocab_up=ml->get_tensor_meta("vocab_router_up.weight");
    const auto * vocab_head=ml->get_tensor_meta("vocab_static_head.weight");
    const auto * vocab_ids=ml->get_tensor_meta("vocab_static_ids");
    const bool vocab_dynamic=vocab_down || vocab_up;
    const bool vocab_static=vocab_head || vocab_ids;
    if(vocab_dynamic || vocab_static) {
        if(vocab_dynamic==vocab_static || !dspark_markov_w1 || d2t || output_factor_down ||
           hparams.dsv4_hc_mult>0 || dflash_selector_hidden || n_vocab_draft!=n_vocab ||
           ml->get_tensor_meta("output.scale"))throw std::runtime_error("exclusive standard DSpark vocab head required");
        auto shape=[](const ggml_tensor *t,int64_t a,int64_t b,ggml_type type) {
            return t && t->type==type && t->ne[0]==a && t->ne[1]==b && t->ne[2]==1 && t->ne[3]==1;
        };
        if(vocab_dynamic) {
            if(!shape(vocab_down,n_embd,64,GGML_TYPE_F16) || !shape(vocab_up,64,n_vocab,GGML_TYPE_F16) ||
               !ml->get_tensor_meta("output.weight"))throw std::runtime_error("vocab dynamic r64 requires original head and both F16 router factors");
            vocab_router_down=create_tensor(tn(LLM_TENSOR_VOCAB_ROUTER_DOWN,"weight"),{n_embd,64},0);
            vocab_router_up=create_tensor(tn(LLM_TENSOR_VOCAB_ROUTER_UP,"weight"),{64,n_vocab},0);
        } else {
            if(!shape(vocab_head,n_embd,4096,GGML_TYPE_F16) || !shape(vocab_ids,4096,1,GGML_TYPE_I32) ||
               ml->get_tensor_meta("output.weight"))throw std::runtime_error("vocab static K4096 requires packed head/IDs and absent dense head");
            vocab_static_head=create_tensor(tn(LLM_TENSOR_VOCAB_STATIC_HEAD,"weight"),{n_embd,4096},0);
            vocab_static_ids=create_tensor(tn(LLM_TENSOR_VOCAB_STATIC_IDS),{4096},0);
        }
        LLAMA_LOG_INFO("DSPARK_VOCAB mode=%s full_markov=1 selected_rows=F16 body_rows=7\\n",vocab_dynamic?"dynamic-r64-k1024":"static-k4096");
    }
'''
        s=once(s,anchor,loader+anchor)
        s=once(s,'(!output && !output_factor_down)','(!output && !output_factor_down && !vocab_static_head)')
        s=once(s,'tok_embd != nullptr && !output_factor_down)','tok_embd != nullptr && !output_factor_down && !vocab_static_head)')
        anchor='    if (model.output_factor_down) {'
        early='''    if (model.vocab_router_down || model.vocab_static_head) {
        if(model.vocab_router_down) {
            cur=build_lora_mm(model.vocab_router_down,cur,nullptr);
            cur=build_lora_mm(model.vocab_router_up,cur,nullptr);
        } else cur=build_lora_mm(model.vocab_static_head,cur,nullptr);
        cb(cur,"vocab_base_projection",-1);res->t_logits=cur;
        build_dspark_vocab_head(*this,model,inp_tokens);
        return;
    }
'''
        return once(s,anchor,early+anchor)
    raise ValueError(rel)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('fresh output required')
    a.out.mkdir(parents=True);patch=[];pins={}
    for rel in ('src/llama-arch.h','src/llama-arch.cpp','src/llama-model.h','src/llama-context.cpp','common/speculative.cpp','src/models/dflash.cpp'):
        old=(a.source/rel).read_text();new=transform(rel,old);dest=a.out/'overlay'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(new)
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel));pins[rel]={'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'patched_sha256':hashlib.sha256(new.encode()).hexdigest()}
    rel='src/vocab_head.h';new=(HERE/'vocab_head.h').read_text();dest=a.out/'overlay'/rel;dest.write_text(new)
    patch.extend(difflib.unified_diff([],new.splitlines(True),fromfile='/dev/null',tofile='b/'+rel));pins[rel]={'base_sha256':None,'patched_sha256':hashlib.sha256(new.encode()).hexdigest()}
    (a.out/'vocabulary.patch').write_text(''.join(patch));(a.out/'manifest.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps({'patch':str(a.out/'vocabulary.patch'),'new_CUDA_code':False}))
if __name__=='__main__':main()
