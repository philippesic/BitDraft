#include "llama-model.h"
#include "ggml-alloc.h"
static void teacher_head(llama_model * model, ggml_backend_dev_t dev, const fs::path & base_path,
 const std::vector<int> & labels, int last, const fs::path & out) {
 auto values=read_floats(base_path,7*151936);
 auto * backend=ggml_backend_dev_init(dev,nullptr);need(backend,"teacher CUDA backend");
 ggml_init_params params{16*1024*1024,nullptr,true};auto * ctx=ggml_init(params);need(ctx,"teacher context");
 auto * base=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,151936,7);
 auto * ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,7);ggml_set_input(base);ggml_set_input(ids);
 ggml_tensor * result=nullptr;
 for(int i=0;i<7;++i){
  auto * prev=ggml_view_1d(ctx,ids,1,i*sizeof(int32_t));
  auto * lookup=ggml_get_rows(ctx,model->dspark_markov_w1,prev);
  auto * bias=ggml_mul_mat(ctx,model->dspark_markov_w2,lookup);
  if(model->dspark_markov_w2_s)bias=ggml_mul(ctx,bias,model->dspark_markov_w2_s);
  auto * col=ggml_add(ctx,ggml_view_2d(ctx,base,151936,1,base->nb[1],i*base->nb[1]),bias);
  result=result?ggml_concat(ctx,result,col,1):col;
 }
 ggml_set_output(result);auto * graph=ggml_new_graph_custom(ctx,2048,false);ggml_build_forward_expand(graph,result);
 auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);need(buffer,"teacher graph allocation");
 int32_t predecessors[7];predecessors[0]=last;for(int i=1;i<7;++i)predecessors[i]=labels[i-1];
 ggml_backend_tensor_set(base,values.data(),0,values.size()*4);ggml_backend_tensor_set(ids,predecessors,0,sizeof(predecessors));
 need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"teacher graph execution");
 std::vector<float> raw(7*151936);ggml_backend_tensor_get(result,raw.data(),0,raw.size()*4);
 need(std::all_of(raw.begin(),raw.end(),[](float x){return std::isfinite(x);}),"teacher finite logits");
 std::ofstream f(out,std::ios::binary);f.write(reinterpret_cast<char*>(raw.data()),raw.size()*4);need(bool(f),"teacher logits write");
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
