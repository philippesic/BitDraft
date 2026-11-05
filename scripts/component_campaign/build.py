"""Small model-only overlay on the qualified immutable optimized CUDA runtime."""
import json,subprocess,difflib,shutil
from pathlib import Path
from .common import P,R,OPT,ROOT,pin,save

def build(r=None):
 r=Path(r) if r is not None else R/'runtime';r.mkdir(exist_ok=False,parents=True);(r/'bin').mkdir()
 frozen=P/'runs/precision-native-20261104-04';base=P/'runs/w1a8-modern-packed-runtime-20261103-01/build02'
 manifest=json.load(open(frozen/'build.json'))
 original=(OPT/'src/models/dflash.cpp').read_text();cpp=original
 # Private full vocabulary embedding; optional tied private output aliases this very tensor.
 cpp=cpp.replace('"d2t","token_embd.weight"','"d2t"')
 needle='    if (hparams.n_lora_kv > 0) {'
 position=cpp.index(needle,cpp.index('void llama_model_dflash::load_arch_tensors'))
 cpp=cpp[:position]+'''    const auto component_tied_it = gguf_kv.find("dflash.component.tied");
    const bool component_tied = component_tied_it != gguf_kv.end() && component_tied_it->second == "true";
    if (component_tied) {
        if (!output || tok_embd) throw std::runtime_error("invalid component tied allocation");
        tok_embd = output;
        LLAMA_LOG_INFO("COMPONENT tied head/embedding: same tensor=%p data=%p bytes=%zu type=%s\\n", (void*)output, output->data, ggml_nbytes(output), ggml_type_name(output->type));
    }
''' + cpp[position:]
 # F16 metadata storage is converted explicitly to F32 graph operands. Kernels remain exact.
 cpp=cpp.replace('sm->type!=GGML_TYPE_F32','(sm->type!=GGML_TYPE_F32 && sm->type!=GGML_TYPE_F16)')
 cpp=cpp.replace('auto gain=tensor(base+".w1ax_gain");','auto gain=tensor(base+".w1ax_gain"); if(gain && gain->type!=GGML_TYPE_F32) gain=ggml_cast(ctx0,gain,GGML_TYPE_F32);')
 cpp=cpp.replace('ggml_w1ax_mul_mat_shared(ctx0,p.packed,p.scales,packed,p.logical_k,bits)','ggml_w1ax_mul_mat_shared(ctx0,p.packed,p.scales->type==GGML_TYPE_F32?p.scales:ggml_cast(ctx0,p.scales,GGML_TYPE_F32),packed,p.logical_k,bits)')
 cpp=cpp.replace('result->src[4]=tensor(base+".w1ax_midpoint");','result->src[4]=tensor(base+".w1ax_midpoint"); if(result->src[4] && result->src[4]->type!=GGML_TYPE_F32) result->src[4]=ggml_cast(ctx0,result->src[4],GGML_TYPE_F32);')
 # Norm gain storage conversion; reductions and multiplication stay F32.
 import re
 cpp=re.sub(r'build_norm\(([^,]+), (layer\.(?:attn_norm|attn_q_norm|attn_k_norm|ffn_norm)|model\.output_norm(?:_enc)?),',r'build_norm(\1, \2->type==GGML_TYPE_F32?\2:ggml_cast(ctx0,\2,GGML_TYPE_F32),',cpp)
 dest=r/'src/models/dflash.cpp';dest.parent.mkdir(parents=True);dest.write_text(cpp)
 (r/'component.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),cpp.splitlines(True),fromfile='a/src/models/dflash.cpp',tofile='b/src/models/dflash.cpp')))
 (r/'ggml-impl.h').symlink_to(frozen/'ggml/src/ggml-impl.h')
 unity=r/'unity.cxx';unity.write_text((frozen/'unity.cxx').read_text().replace(str(frozen/'src/models/dflash.cpp'),str(dest)))
 obj=r/'dflash.cpp.o';cmd=next(c.copy() for c in manifest['commands'] if '-c' in c and str(frozen/'dflash.cpp.o') in c)
 cmd=['-I'+str(r) if x=='-I'+str(frozen) else x for x in cmd];cmd[cmd.index('-c')+1]=str(unity);cmd[cmd.index('-o')+1]=str(obj)
 if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(obj)+'.d'
 commands=[cmd]
 with (r/'build.log').open('w') as log:
  subprocess.run(cmd,cwd=base,stdout=log,stderr=subprocess.STDOUT,check=True)
  cmd=next(c.copy() for c in manifest['commands'] if '-shared' in c and str(frozen/'bin/libllama.so.0.8.0') in c)
  cmd=[str(obj) if x==str(frozen/'dflash.cpp.o') else x for x in cmd];cmd[cmd.index('-o')+1]=str(r/'bin/libllama.so.0.8.0');commands.append(cmd)
  subprocess.run(cmd,cwd=base,stdout=log,stderr=subprocess.STDOUT,check=True)
 for src in (OPT/'bin').iterdir():
  dest=r/'bin'/src.name
  if dest.exists():continue
  own=r/'bin'/src.resolve().name;dest.symlink_to(own if own.exists() else src.resolve())
 save(r/'build.json',dict(base=pin(OPT/'build.json'),qualified_cuda=pin(OPT/'bin/libggml-cuda.so.0.28.0'),patch=pin(r/'component.patch'),source=pin(r/'src/models/dflash.cpp'),commands=commands,libraries=[pin(r/'bin/libllama.so.0.8.0')],semantics='new declared component precision; qualified execution kernels retained; target/verifier consistent32'))
 print('Component runtime built',flush=True)

if __name__=='__main__':build()
