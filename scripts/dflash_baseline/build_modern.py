"""Isolated DFlash relink against the matched modern packed runtime."""
from pathlib import Path
import difflib, hashlib, json, shlex, subprocess
P=Path('/home/philip/binary-eagle-decoding')
OLD=P/'runs/w1a8-modern-packed-runtime-20261103-01'
B=OLD/'build02'
R=P/'runs/dflash-modern-native-20261104-03'
R.mkdir(exist_ok=True)
(R/'bin').mkdir(exist_ok=True)
f=OLD/'source/src/models/dflash.cpp'
s=f.read_text(); original=s
s=s.replace('#include "models.h"','#include "models.h"\n#include <set>')
guard=r'''
    uint32_t version = 0;
    const bool binary = ml->get_key("dflash.w1ax.version", version, false);
    std::set<std::string> declared;
    if (binary) {
        uint32_t bits=0;
        std::string profile, order, sign, scale;
        std::vector<std::string> names;
        ml->get_key("dflash.w1ax.activation_bits", bits);
        ml->get_key("dflash.w1ax.profile", profile);
        ml->get_key("dflash.w1ax.bit_order", order);
        ml->get_key("dflash.w1ax.sign_rule", sign);
        ml->get_key("dflash.w1ax.scale_rule", scale);
        ml->get_arr("dflash.w1ax.tensors", names);
        declared.insert(names.begin(), names.end());
        std::set<std::string> expected;
        for(int i=0;i<6;++i) for(auto p : {"gate","up","down"}) expected.insert("blk."+std::to_string(i)+".ffn_"+p+".weight");
        const std::vector<int32_t> taps={2,6,10,14,18,22,26,30};
        bool ok=version==1 && bits==8 && profile=="ffn18" && order=="little" &&
            sign=="nonnegative_is_one" && scale=="f32_learned_nonnegative" &&
            declared==expected && names.size()==18 && n_layer==6 && n_embd==2560 &&
            target_layer_ids==taps && hparams.dflash_block_size==16 && hparams.n_swa==4096 &&
            hparams.rope_freq_base_train==10000000.0f && hparams.llm_ffn_op==LLM_FFN_SILU &&
            !hparams.dsv4_hc_mult && !hparams.dflash_conv_kernel_size && !hparams.lilicorr_layers;
        for(int i=0;i<n_layer;++i) ok=ok && hparams.n_ff(i)==9216 && hparams.n_head(i)==32 &&
            hparams.n_head_kv(i)==8 && hparams.is_swa_impl[i]==(i<5);
        for(auto name : {"markov_w1.weight","selector_hidden.weight","d2t","token_embd.weight","output.weight"}) ok=ok && !ml->get_tensor_meta(name);
        const auto anchor_it=gguf_kv.find("dflash.sample_from_anchor");
        const bool anchor=anchor_it!=gguf_kv.end() && anchor_it->second=="true";
        if(!ok || anchor || vocab.token_mask()<0 || vocab.token_mask()>=n_vocab) throw std::runtime_error("unsupported modern DFlash W1A8 contract");
        LLAMA_LOG_INFO("DFlash W1A8: admitted modern ffn18 packed profile\n");
    }
    for(const auto & item : ml->weights_map) {
        const std::string name=item.first;
        for(auto suffix : {".w1a1_packed",".w1a1_scale",".w1ax_midpoint"}) {
            const std::string end=suffix;
            if(name.size()>=end.size() && name.compare(name.size()-end.size(),end.size(),end)==0 &&
                (!binary || end==".w1ax_midpoint" || !declared.count(name.substr(0,name.size()-end.size())+".weight")))
                throw std::runtime_error("undeclared DFlash packed tensor: "+name);
        }
    }
    auto load_binary = [&](llm_tensor tensor,int bid,int64_t k,int64_t rows,ggml_tensor *& dense,llama_w1a8_linear & packed) {
        if(!binary) {dense=create_tensor(tn(tensor,"weight",bid),{k,rows},0); return;}
        const std::string name=tn(tensor,"weight",bid).str(), base=name.substr(0,name.size()-7);
        const auto * pm=ml->get_tensor_meta((base+".w1a1_packed").c_str());
        const auto * sm=ml->get_tensor_meta((base+".w1a1_scale").c_str());
        if(k%32 || ml->get_tensor_meta(name.c_str()) || !pm || !sm || pm->type!=GGML_TYPE_I32 || sm->type!=GGML_TYPE_F32 ||
            pm->ne[0]!=k/32 || pm->ne[1]!=rows || pm->ne[2]!=1 || pm->ne[3]!=1 || sm->ne[0]!=rows || sm->ne[1]!=1 || sm->ne[2]!=1 || sm->ne[3]!=1)
            throw std::runtime_error("DFlash packed shape/type mismatch: "+name);
        packed.packed=create_tensor(LLM_TN_IMPL(LLM_ARCH_DFLASH,tensor,"w1a1_packed",bid,-1),{k/32,rows},0);
        packed.scales=create_tensor(LLM_TN_IMPL(LLM_ARCH_DFLASH,tensor,"w1a1_scale",bid,-1),{rows},0);
        packed.logical_k=k;
        dense=nullptr;
    };
'''
needle='    const int64_t n_embd_inp = hparams.n_embd_inp_enc();'
assert s.count(needle)>=1
s=s.replace(needle,needle+guard,1)
for part,shape in [('gate','n_embd, n_ff'),('up','n_embd, n_ff'),('down','n_ff, n_embd')]:
 import re
 pattern=rf'layer.ffn_{part}\s*= create_tensor\(tn\(LLM_TENSOR_FFN_{part.upper()},\s*"weight", i\), \{{ {shape} \}}, 0\);'
 s,n=re.subn(pattern,f'load_binary(LLM_TENSOR_FFN_{part.upper()}, i, {shape}, layer.ffn_{part}, layer.ffn_{part}_w1a8);',s)
 assert n>=1
needle='''        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);'''
assert s.count(needle)==1
s=s.replace(needle,'''        if (layer.ffn_up_w1a8.packed) {
            auto packed = ggml_w1ax_pack(ctx0, ggml_cont(ctx0, cur), 8);
            auto up = ggml_w1ax_mul_mat_shared(ctx0, layer.ffn_up_w1a8.packed, layer.ffn_up_w1a8.scales, packed, layer.ffn_up_w1a8.logical_k, 8);
            auto gate = ggml_w1ax_mul_mat_shared(ctx0, layer.ffn_gate_w1a8.packed, layer.ffn_gate_w1a8.scales, packed, layer.ffn_gate_w1a8.logical_k, 8);
            cur = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
            cur = ggml_w1ax_mul_mat(ctx0, layer.ffn_down_w1a8.packed, layer.ffn_down_w1a8.scales, ggml_cont(ctx0, cur), layer.ffn_down_w1a8.logical_k, 8);
        } else {
'''+needle+'\n        }')
(R/'dflash.cpp').write_text(s)
patch=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/models/dflash.cpp',tofile='b/src/models/dflash.cpp'))
(Path(__file__).resolve().parents[2]/'experiments/dflash-w1a8-baseline-20261104/runtime/matched-modern.patch').write_text(patch)
unity=B/'src/CMakeFiles/llama.dir/Unity/unity_1_cxx.cxx'
(R/'unity.cxx').write_text(unity.read_text().replace(str(f),str(R/'dflash.cpp')))
def commands(target): return subprocess.check_output(['ninja','-C',str(B),'-t','commands',target],text=True).splitlines()
cmd=shlex.split(commands('src/CMakeFiles/llama.dir/Unity/unity_1_cxx.cxx.o')[-1])
cmd.append('-I'+str(f.parent))
cmd[cmd.index('-o')+1]=str(R/'unity.o')
cmd[cmd.index('-c')+1]=str(R/'unity.cxx')
if '-MF' in cmd: cmd[cmd.index('-MF')+1]=str(R/'unity.d')
log=[]
with (R/'build.log').open('w') as out:
 subprocess.run(cmd,cwd=B,stdout=out,stderr=subprocess.STDOUT,check=True); log.append(cmd)
 link=shlex.split(commands('bin/libllama.so.0.8.0')[-1]); link=link[2:-2]
 link=[str(R/'unity.o') if x=='src/CMakeFiles/llama.dir/Unity/unity_1_cxx.cxx.o' else x for x in link if not x.startswith('-Wl,--dependency-file=')]
 link[link.index('-o')+1]=str(R/'bin/libllama.so.0.8.0')
 subprocess.run(link,cwd=B,stdout=out,stderr=subprocess.STDOUT,check=True); log.append(link)
for p in (B/'bin').iterdir():
 dest=R/'bin'/p.name
 if p.is_file() and not dest.exists(): dest.symlink_to('libllama.so.0.8.0' if p.name in ['libllama.so','libllama.so.0'] else p.resolve())
def pin(p):
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=digest,bytes=p.stat().st_size)
(R/'build.json').write_text(json.dumps(dict(commands=log,source=pin(f),modified=pin(R/'dflash.cpp'),library=pin(R/'bin/libllama.so.0.8.0')),indent=2)+'\n')
print(R)
