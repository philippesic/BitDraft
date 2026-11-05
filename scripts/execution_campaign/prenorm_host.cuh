void ggml_cuda_exec_prenorm(ggml_backend_cuda_context&ctx,ggml_tensor*pack,ggml_tensor*rms,ggml_tensor*mul){
 int k=rms->ne[0],n=rms->ne[1];auto layout=ggml_w1ax_pack_layout(k,n,8);auto*p=(uint32_t*)pack->data;
 exec_prenorm_kernel<<<n,1024,0,ctx.stream()>>>((const float*)rms->src[0]->data,(const float*)mul->src[1]->data,pack->src[1]?(const float*)pack->src[1]->data:nullptr,k,ggml_get_op_params_i32(pack,0),ggml_get_op_params_f32(rms,0),ggml_get_op_params_f32(pack,2),(int8_t*)p,(float*)(p+layout.scale_offset),(float*)(p+layout.scale_offset+n));CUDA_CHECK(cudaGetLastError());
}
