from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P,F
from scripts.execution_campaign.extend import relink
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--silu',action='store_true');ap.add_argument('--residual',action='store_true');ap.add_argument('--generic',action='store_true');ap.add_argument('--joint',action='store_true');ap.add_argument('--prenorm',action='store_true');ap.add_argument('--debug',action='store_true');a=ap.parse_args()
r=build(a.out,copy=True,bound=True,threads=256)
cu=(r/'w1a1.cu').read_text();backend=(F/'ggml/src/ggml-cuda/ggml-cuda.cu').read_text();header=(F/'ggml/src/ggml-impl.h').read_text();cpp=(r/'dflash.cpp').read_text();pre=''
if a.silu:
 cpp=cpp.replace('ggml_mul(ctx0,ggml_silu(ctx0,gate),up)','ggml_mul(ctx0,up,ggml_silu(ctx0,gate))')
 header=header.replace('for (int i = 2; i < GGML_MAX_SRC; ++i)', 'if (op->src[2] && (op->src[2]->type!=GGML_TYPE_F32 || !ggml_are_same_shape(op->src[2],a) || !ggml_is_contiguous(op->src[2]))) return false;\n        for (int i = 3; i < GGML_MAX_SRC; ++i)',1)
 cu=cu.replace('// Four output rows per block',(ROOT/'scripts/execution_campaign/fusion.cuh').read_text()+'\n// Four output rows per block')
 cu=cu.replace('    if (ggml_get_op_params_i32(dst,1)) {', '''    if (dst->src[2]) {
        exec_silu_pack<256><<<n,256,0,ctx.stream()>>>((const float*)acts->data,(const float*)dst->src[2]->data,dst->src[1]?(const float*)dst->src[1]->data:nullptr,k,ggml_get_op_params_i32(dst,0),ggml_get_op_params_f32(dst,2),codes,scales,scales+n);
    } else if (ggml_get_op_params_i32(dst,1)) {''')
 pre+='''    if (ggml_can_fuse_subgraph(cgraph,i,{GGML_OP_UNARY,GGML_OP_MUL,GGML_OP_W1AX_PACK},{i+1,i+2}) && ggml_get_unary_op(cgraph->nodes[i])==GGML_UNARY_OP_SILU) {
        auto* silu=cgraph->nodes[i];auto* mul=cgraph->nodes[i+1];auto* pack=cgraph->nodes[i+2];
        if (pack->src[0]==mul && mul->src[1]==silu && ggml_get_op_params_i32(pack,1)!=0 && ggml_is_contiguous(silu->src[0]) && ggml_is_contiguous(mul->src[0])) {
            bool uses_ok=true;for(int j=i+3;j<cgraph->n_nodes;++j)for(int slot=0;slot<GGML_MAX_SRC;++slot)if(cgraph->nodes[j]->src[slot]==mul && !(cgraph->nodes[j]->op==GGML_OP_W1A1_MUL_MAT && slot==2))uses_ok=false;
            int outputs[]={i+2};if(!uses_ok || !ggml_cuda_check_fusion_memory_ranges(cgraph,i,3,outputs,1))return 0;
            GGML_LOG_INFO("EXEC_SILU_FUSION\\n");ggml_tensor fused=*pack;fused.src[0]=silu->src[0];fused.src[2]=mul->src[0];ggml_cuda_w1ax_pack(*cuda_ctx,&fused);return 2;
        }
    }
'''
if a.residual:
 header=header.replace('for (int i = 5; i < GGML_MAX_SRC; ++i)', 'if (op->src[5] && (op->src[5]->type!=GGML_TYPE_F32 || !ggml_are_same_shape(op->src[5],op) || !ggml_is_contiguous(op->src[5]))) return false;\n    for (int i = 6; i < GGML_MAX_SRC; ++i)',1)
 start=cu.index('static __global__ void w1ax_integer_dot(');end=cu.index('void ggml_cuda_w1ax_pack',start)
 part=cu[start:end].replace('const float * sums) {','const float * sums, const float * residual=nullptr) {')
 part=part.replace('output[token*m + row] = midpoints ?', 'const float value = midpoints ?').replace(': dot;\n', ': dot;\n        output[token*m+row]=residual?__fadd_rn(value,residual[token*m+row]):value;\n')
 cu=cu[:start]+part+cu[end:]
 cu=cu.replace('layout.scale_offset+n));','layout.scale_offset+n),dst->src[5]?(const float*)dst->src[5]->data:nullptr);')
 pre+='''    if (ggml_can_fuse(cgraph,i,{GGML_OP_W1A1_MUL_MAT,GGML_OP_ADD})) {
        auto* dot=cgraph->nodes[i];auto* add=cgraph->nodes[i+1];
        if (add->src[0]==dot && ggml_are_same_shape(dot,add->src[1]) && ggml_is_contiguous(add->src[1])) {
            int outputs[]={i+1};if(!ggml_cuda_check_fusion_memory_ranges(cgraph,i,2,outputs,1))return 0;
            GGML_LOG_INFO("EXEC_RESIDUAL_FUSION\\n");ggml_tensor fused=*dot;fused.data=add->data;fused.src[5]=add->src[1];ggml_cuda_w1a1_mul_mat(*cuda_ctx,&fused);return 1;
        }
    }
'''
if a.joint:
 cpp=cpp.replace('ggml_mul(ctx0,ggml_silu(ctx0,gate),up)','ggml_mul(ctx0,up,ggml_silu(ctx0,gate))')
 cu=cu.replace('// Four output rows per block',(ROOT/'scripts/execution_campaign/joint.cuh').read_text()+'\n// Four output rows per block')
 cu+='\n'+(ROOT/'scripts/execution_campaign/joint_host.cuh').read_text()
 backend=backend.replace('static int ggml_cuda_try_fuse','void ggml_cuda_exec_joint(ggml_backend_cuda_context&,ggml_tensor*,ggml_tensor*);\nstatic int ggml_cuda_try_fuse',1)
 pre+='''    if (ggml_can_fuse_subgraph(cgraph,i,{GGML_OP_W1A1_MUL_MAT,GGML_OP_W1A1_MUL_MAT},{i,i+1})) {
        auto*a=cgraph->nodes[i];auto*b=cgraph->nodes[i+1];int outputs[]={i,i+1};
        if(a->src[3]==b->src[3] && ggml_are_same_shape(a,b) && ggml_cuda_check_fusion_memory_ranges(cgraph,i,2,outputs,2)){
            GGML_LOG_INFO("EXEC_JOINT_FUSION\\n");ggml_cuda_exec_joint(*cuda_ctx,a,b);return 1;
        }
    }
'''
if a.prenorm:
 cu=cu.replace('// Four output rows per block',(ROOT/'scripts/execution_campaign/prenorm.cuh').read_text()+'\n// Four output rows per block')
 cu+='\n'+(ROOT/'scripts/execution_campaign/prenorm_host.cuh').read_text()
 backend=backend.replace('static int ggml_cuda_try_fuse','void ggml_cuda_exec_prenorm(ggml_backend_cuda_context&,ggml_tensor*,ggml_tensor*,ggml_tensor*);\nstatic int ggml_cuda_try_fuse',1)
 pre+='''    if (ggml_can_fuse_subgraph(cgraph,i,{GGML_OP_RMS_NORM,GGML_OP_MUL,GGML_OP_W1AX_PACK},{i+1,i+2})) {
        auto*rms=cgraph->nodes[i];auto*mul=cgraph->nodes[i+1];auto*pack=cgraph->nodes[i+2];int outputs[]={i+1,i+2};bool ok=true;
        for(int j=i+3;j<cgraph->n_nodes;++j)for(int slot=0;slot<GGML_MAX_SRC;++slot)if(cgraph->nodes[j]->src[slot]==mul && !(cgraph->nodes[j]->op==GGML_OP_W1A1_MUL_MAT && slot==2))ok=false;
        if(ok && rms->ne[0]==2560 && pack->src[0]==mul && mul->src[0]==rms && ggml_is_contiguous(rms->src[0]) && ggml_is_contiguous(mul->src[1]) && ggml_nelements(mul->src[1])==rms->ne[0] && ggml_get_op_params_i32(pack,1)!=0 && ggml_cuda_check_fusion_memory_ranges(cgraph,i,3,outputs,2)){
            GGML_LOG_INFO("EXEC_PRENORM_FUSION\\n");ggml_cuda_exec_prenorm(*cuda_ctx,pack,rms,mul);return 2;
        }
    }
'''
needle='static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {\n'
if a.debug:pre='    if (cgraph->nodes[i]->op==GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[i])==GGML_UNARY_OP_SILU && i+3<cgraph->n_nodes) GGML_LOG_INFO("EXEC_ORDER %s %s %s %s\\n",ggml_op_name(cgraph->nodes[i]->op),ggml_op_name(cgraph->nodes[i+1]->op),ggml_op_name(cgraph->nodes[i+2]->op),ggml_op_name(cgraph->nodes[i+3]->op));\n'+pre
backend=backend.replace(needle,needle+pre)
if a.generic:
 backend=backend.replace('if (ggml_cuda_fusion_disabled()) {\n        return 0;', 'if (ggml_cuda_fusion_disabled() && strncmp(cgraph->nodes[i]->name,"exec_draft_",11)!=0) {\n        return 0;',1)
 # Mark only DFlash1 decoder computation nodes, without touching target/shared weight leaves.
 marker='    // DSpark: bias the draft logits with the Markov head\n'
 code='''    if (!model.dspark_markov_w1 && !model.dflash_selector_hidden && !model.lilicorr_pass_hidden_proj) {
        std::set<ggml_tensor*> visited;
        auto mark=[&](auto&&self,ggml_tensor*t)->void{if(!t||t->op==GGML_OP_NONE||!visited.insert(t).second)return;for(auto*s:t->src)self(self,s);std::string n="exec_draft_"+std::string(t->name);ggml_set_name(t,n.c_str());};mark(mark,res->t_logits);
    }
'''
 cpp=cpp.replace(marker,code+marker,1)
changes={'ggml-impl.h':header,'w1a1.cu':cu,'ggml/src/ggml-cuda/ggml-cuda.cu':backend}
if a.generic or a.silu or a.joint:changes['src/models/dflash.cpp']=cpp
relink(r,changes)
