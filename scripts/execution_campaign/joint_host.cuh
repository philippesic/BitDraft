void ggml_cuda_exec_joint(ggml_backend_cuda_context&ctx,ggml_tensor*a,ggml_tensor*b){
 int k=a->src[2]->ne[0],n=a->src[2]->ne[1],m=a->ne[0];auto layout=ggml_w1ax_pack_layout(k,n,8);auto*p=(uint32_t*)a->src[3]->data;
 exec_joint_dot<<<dim3((m+3)/4,n),dim3(32,4),0,ctx.stream()>>>((const uint32_t*)a->src[0]->data,(const float*)a->src[1]->data,a->src[4]?(const float*)a->src[4]->data:nullptr,(float*)a->data,(const uint32_t*)b->src[0]->data,(const float*)b->src[1]->data,b->src[4]?(const float*)b->src[4]->data:nullptr,(float*)b->data,(const int8_t*)p,(const float*)(p+layout.scale_offset),(const float*)(p+layout.scale_offset+n),m,k);CUDA_CHECK(cudaGetLastError());
}
