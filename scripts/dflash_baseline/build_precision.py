"""Relink isolated matched runtime objects; never modify the frozen source/build."""
import argparse,difflib,hashlib,json,shlex,subprocess
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');BASE=P/'runs/w1a8-modern-packed-runtime-20261103-01';B=BASE/'build02';S=BASE/'source'
def replace(s,a,b,count=1):
 if s.count(a)!=count:raise ValueError(f'expected {count} occurrences, got {s.count(a)}: {a[:100]}')
 return s.replace(a,b)
def prepare(r):
 originals={};changed={}
 def add(name,s,origin=None):
  originals[name]=(origin or S/name).read_text();changed[name]=s
 impl=(S/'ggml/src/ggml-impl.h').read_text()
 impl=impl.replace('bits == 8 &&','(bits == 1 || bits == 2 || bits == 4 || bits == 8) &&')
 impl=replace(impl,'layout.total_words = layout.scale_offset + n;','layout.total_words = layout.scale_offset + 2*n;')
 impl=replace(impl,'ggml_get_op_params_i32(op, 0) != 8','(ggml_get_op_params_i32(op, 0) != 1 && ggml_get_op_params_i32(op, 0) != 2 && ggml_get_op_params_i32(op, 0) != 4 && ggml_get_op_params_i32(op, 0) != 8)')
 impl=replace(impl,'for (uint32_t i = 1; i < GGML_MAX_OP_PARAMS/sizeof(int32_t); ++i)', 'for (uint32_t i = 3; i < GGML_MAX_OP_PARAMS/sizeof(int32_t); ++i)')
 impl=replace(impl,'ggml_w1ax_pack_layout(a->ne[0], a->ne[1], 8)','ggml_w1ax_pack_layout(a->ne[0], a->ne[1], ggml_get_op_params_i32(op, 0))')
 impl=replace(impl,'ggml_get_op_params_i32(op, 2) != 8','(ggml_get_op_params_i32(op, 2) != 1 && ggml_get_op_params_i32(op, 2) != 2 && ggml_get_op_params_i32(op, 2) != 4 && ggml_get_op_params_i32(op, 2) != 8)')
 impl=replace(impl,'for (int i = 4; i < GGML_MAX_SRC; ++i)', 'if (op->src[4] && (op->src[4]->type != GGML_TYPE_F32 || op->src[4]->ne[0] != w->ne[1] || ggml_nelements(op->src[4]) != w->ne[1])) return false;\n    for (int i = 5; i < GGML_MAX_SRC; ++i)')
 impl=replace(impl,'ggml_w1ax_pack_layout(k, a->ne[1], 8)','ggml_w1ax_pack_layout(k, a->ne[1], ggml_get_op_params_i32(op, 2))')
 impl=replace(impl,'const struct ggml_w1ax_pack_layout layout = ggml_w1ax_pack_layout(a->ne[0]', 'if (ggml_get_op_params_i32(op, 1) < 0 || ggml_get_op_params_i32(op, 1) > 2) return false;\n        const struct ggml_w1ax_pack_layout layout = ggml_w1ax_pack_layout(a->ne[0]')
 impl=replace(impl,'for (int i = 1; i < GGML_MAX_SRC; ++i) {\n            if (op->src[i] != NULL) return false;','if (op->src[1] && (op->src[1]->type != GGML_TYPE_F32 || op->src[1]->ne[0] != a->ne[0] || ggml_nelements(op->src[1]) != a->ne[0] || !ggml_is_contiguous(op->src[1]))) return false;\n        for (int i = 2; i < GGML_MAX_SRC; ++i) {\n            if (op->src[i] != NULL) return false;')
 add(Path('ggml/src/ggml-impl.h'),impl)
 c=(S/'ggml/src/ggml.c').read_text().replace('GGML_ASSERT(bits == 8 &&','GGML_ASSERT((bits == 1 || bits == 2 || bits == 4 || bits == 8) &&')
 add(Path('ggml/src/ggml.c'),c)
 cu=(S/'ggml/src/ggml-cuda/w1a1.cu').read_text()
 cu=replace(cu,'float * scales) {','float * scales, float * sums, int bits, int mode, float clip) {')
 cu=replace(cu,'__shared__ float token_absmax;','__shared__ float token_absmax;\n    __shared__ int totals[256];\n    const int qmax = bits <= 2 ? 1 : (1 << (bits-1))-1;')
 cu=replace(cu,'token_absmax = __uint_as_float(maxima[0]);','token_absmax = mode == 0 ? __uint_as_float(maxima[0]) : clip;\n        if (!isfinite(token_absmax) || token_absmax < 0 || (mode != 0 && token_absmax == 0)) __trap();')
 cu=cu.replace('127.0f','(float) qmax').replace('*127.0','*(double) qmax')
 cu=replace(cu,'const int code = __float2int_rn(normalized);\n        codes[token*k + i] = (int8_t) max(-127, min(127, code));','const int code = bits == 1 ? (row[i] >= 0 ? 1 : -1) : max(-qmax, min(qmax, __float2int_rn(fminf((float)qmax,fmaxf(-(float)qmax,normalized)))));\n        codes[token*k + i] = (int8_t) code;\n        total += code;')
 cu=replace(cu,'for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {\n        // If','int total=0;\n    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {\n        // If')
 cu=replace(cu,'        total += code;\n    }\n}', '''        total += code;
    }
    totals[threadIdx.x]=total; __syncthreads();
    for(int stride=128;stride>0;stride/=2) { if(threadIdx.x<stride) totals[threadIdx.x]+=totals[threadIdx.x+stride]; __syncthreads(); }
    if(threadIdx.x==0) sums[token]=__fmul_rn((float)totals[0],scales[token]);
}''')
 cu=replace(cu,'int64_t m, int64_t k, int64_t words, float * output)', 'int64_t m, int64_t k, int64_t words, float * output, const float * midpoints, const float * sums)')
 cu=replace(cu,'output[token*m + row] = __fmul_rn(weighted, beta);','const float dot = __fmul_rn(weighted, beta);\n        output[token*m + row] = midpoints ? __fadd_rn(dot,__fmul_rn(midpoints[row],sums[token])) : dot;')
 cu=replace(cu,'(const float *) acts->data, k, codes, scales);','(const float *) acts->data, k, codes, scales, scales+n, ggml_get_op_params_i32(dst,0), ggml_get_op_params_i32(dst,1), ggml_get_op_params_f32(dst,2));')
 cu=replace(cu,'m, k, (k+31)/32, (float *) dst->data);','m, k, (k+31)/32, (float *) dst->data, dst->src[4] ? (const float *)dst->src[4]->data : nullptr, (const float *)(packed+layout.scale_offset+n));')
 cu=replace(cu,'float clip) {','float clip, const float * gains) {')
 cu=replace(cu,'static __global__ void w1ax_quantize(','''static __device__ float gained_value(const float * row,const float * gains,int64_t j) {
    if(!gains) return row[j];
    const float gain=gains[j];
    if(!isfinite(gain) || gain<=0) __trap();
    return __fmul_rn(row[j],gain);
}
static __global__ void w1ax_quantize(''')
 cu=replace(cu,'const uint32_t magnitude = __float_as_uint(row[i])','const float value=gained_value(row,gains,i);\n        const uint32_t magnitude = __float_as_uint(value)')
 cu=replace(cu,'const float normalized = zero ?','const float value=gained_value(row,gains,i);\n        const float normalized = zero ?')
 cu=cu.replace('__fmul_rn(row[i], inv)','__fmul_rn(value, inv)').replace('(double) row[i]/','(double) value/').replace('(row[i] >= 0 ? 1 : -1)','(value >= 0 ? 1 : -1)')
 cu=replace(cu,'ggml_get_op_params_f32(dst,2));','ggml_get_op_params_f32(dst,2), dst->src[1] ? (const float *)dst->src[1]->data : nullptr);')
 add(Path('ggml/src/ggml-cuda/w1a1.cu'),cu)
 # CPU implementation remains the original contract; fail closed in this runtime.
 cpu=(S/'ggml/src/ggml-cpu/ggml-cpu.cpp').read_text()
 cpu=replace(cpu,'case GGML_OP_W1AX_PACK:\n        case GGML_OP_W1A1_MUL_MAT:\n            return ggml_w1ax_is_supported(op);','case GGML_OP_W1AX_PACK:\n        case GGML_OP_W1A1_MUL_MAT:\n            return false; // affine precision campaign requires CUDA')
 add(Path('ggml/src/ggml-cpu/ggml-cpu.cpp'),cpu)
 # Recompile backend admission against the changed header too.
 add(Path('ggml/src/ggml-cuda/ggml-cuda.cu'),(S/'ggml/src/ggml-cuda/ggml-cuda.cu').read_text())
 f=P/'runs/dflash-modern-native-20261104-03/dflash.cpp';cpp=f.read_text()
 cpp=replace(cpp,'for(auto name : {\"markov_w1.weight\",\"selector_hidden.weight\",\"d2t\",\"token_embd.weight\",\"output.weight\"}) ok=ok && !ml->get_tensor_meta(name);','for(auto name : {\"markov_w1.weight\",\"selector_hidden.weight\",\"d2t\",\"token_embd.weight\"}) ok=ok && !ml->get_tensor_meta(name); if(version==1) ok=ok && !ml->get_tensor_meta(\"output.weight\");')
 cpp=replace(cpp,'bool ok=version==1 && bits==8','bool ok=(version==1 || version==2) && bits==8')
 cpp=replace(cpp,'(!binary || end==".w1ax_midpoint" || !declared.count','(!binary || (end==".w1ax_midpoint" && version!=2) || !declared.count')
 needle='''        packed.logical_k=k;
        dense=nullptr;'''
 cpp=replace(cpp,needle,'''        packed.logical_k=k;
        if(version==2) {
            uint32_t abits=0, mode=0; float clip=0;
            ml->get_key(base+".w1ax.bits",abits); ml->get_key(base+".w1ax.mode",mode); ml->get_key(base+".w1ax.clip",clip);
            if((abits!=1 && abits!=2 && abits!=4 && abits!=8) || mode>2 || !std::isfinite(clip) || (mode && clip<=0)) throw std::runtime_error("invalid quantizer metadata");
            memcpy(packed.scales->op_params,&abits,4); memcpy((char*)packed.scales->op_params+4,&mode,4); memcpy((char*)packed.scales->op_params+8,&clip,4);
            create_tensor(tn(tensor,"w1ax_midpoint",bid),{rows},0);
            if(tensor!=LLM_TENSOR_FFN_UP) create_tensor(tn(tensor,"w1ax_gain",bid),{k},0);
        }
        dense=nullptr;''')
 start=cpp.index('            auto packed = ggml_w1ax_pack(ctx0, ggml_cont(ctx0, cur), 8);')
 end=cpp.index('\n        } else {',start)
 cpp=cpp[:start]+'''            auto tensor=[&](const std::string &n) { return const_cast<ggml_tensor *>(model.get_tensor(n.c_str())); };
            auto pack=[&](ggml_tensor * input, const llama_w1a8_linear &p, const std::string &base) {
                auto gain=tensor(base+".w1ax_gain");
                int bits=p.scales->op_params[0]; if(!bits) bits=8;
                auto result=ggml_w1ax_pack(ctx0,ggml_cont(ctx0,input),bits); result->src[1]=gain;
                if(p.scales->op_params[0]) memcpy((char*)result->op_params+4,(char*)p.scales->op_params+4,8);
                return result;
            };
            auto project=[&](const llama_w1a8_linear &p,ggml_tensor *packed,const std::string &base) {
                int bits=p.scales->op_params[0]; if(!bits) bits=8;
                auto result=ggml_w1ax_mul_mat_shared(ctx0,p.packed,p.scales,packed,p.logical_k,bits);
                result->src[4]=tensor(base+".w1ax_midpoint"); return result;
            };
            const std::string base="blk."+std::to_string(il)+".ffn_";
            if(memcmp(layer.ffn_gate_w1a8.scales->op_params,layer.ffn_up_w1a8.scales->op_params,12)) throw std::runtime_error("gate/up quantizers must match");
            auto packed=pack(cur,layer.ffn_gate_w1a8,base+"gate");
            auto up=project(layer.ffn_up_w1a8,packed,base+"up");
            auto gate=project(layer.ffn_gate_w1a8,packed,base+"gate");
            cur=ggml_mul(ctx0,ggml_silu(ctx0,gate),up);
            auto down=pack(cur,layer.ffn_down_w1a8,base+"down");
            cur=project(layer.ffn_down_w1a8,down,base+"down");'''+cpp[end:]
 add(Path('src/models/dflash.cpp'),cpp,f)
 for name,content in changed.items():
  dest=r/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content)
  patch=''.join(difflib.unified_diff(originals[name].splitlines(True),content.splitlines(True),fromfile='a/'+str(name),tofile='b/'+str(name)))
  (r/(name.name+'.patch')).write_text(patch)
 # Local headers override originals without changing any original object/source.
 (r/'ggml-impl.h').symlink_to(r/'ggml/src/ggml-impl.h')
 return changed

def build(r):
 changed=prepare(r);(r/'bin').mkdir();commands=[]
 def command(target):return shlex.split(subprocess.check_output(['ninja','-C',str(B),'-t','commands',target],text=True).splitlines()[-1])
 objects={}
 with (r/'build.log').open('w') as log:
  for name in changed:
   if name.suffix=='.h':continue
   if name==Path('src/models/dflash.cpp'):
    target='src/CMakeFiles/llama.dir/Unity/unity_1_cxx.cxx.o';unity=B/'src/CMakeFiles/llama.dir/Unity/unity_1_cxx.cxx'
    content=unity.read_text().replace(str(S/name),str(r/name));src=r/'unity.cxx';src.write_text(content)
   else:
    target=str(name.parent/'CMakeFiles'/('ggml-cuda.dir' if name.suffix=='.cu' else 'ggml-cpu.dir' if 'ggml-cpu' in str(name) else 'ggml-base.dir')/(name.name+'.o'));src=r/name
   if 'ggml-cpu' in str(name):target='ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o'
   obj=r/(name.name+'.o');cmd=command(target);cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(obj)
   if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(obj)+'.d'
   cmd[1:1]=['-I'+str(r),'-I'+str((S/name).parent)]
   commands.append(cmd);subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True);objects[target]=str(obj)
  for lib in ('libggml-base.so.0.28.0','libggml-cpu.so.0.28.0','libggml-cuda.so.0.28.0','libllama.so.0.8.0'):
   cmd=command('bin/'+lib)[2:-2];cmd=[objects.get(x,x) for x in cmd if not x.startswith('-Wl,--dependency-file=')];cmd[cmd.index('-o')+1]=str(r/'bin'/lib)
   commands.append(cmd);subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 for src in (B/'bin').iterdir():
  dest=r/'bin'/src.name
  if src.is_file() and not dest.exists():
   own=r/'bin'/src.resolve().name;dest.symlink_to(own if own.exists() else src.resolve())
 def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
 (r/'build.json').write_text(json.dumps(dict(commands=commands,files=[pin(r/n) for n in changed],libraries=[pin(p) for p in (r/'bin').iterdir() if not p.is_symlink()]),indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);a=ap.parse_args();a.out.mkdir(exist_ok=False);build(a.out)
