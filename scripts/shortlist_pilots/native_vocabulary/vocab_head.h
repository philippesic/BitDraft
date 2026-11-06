#pragma once
// Native C1 DSpark vocabulary reference; all selected/scatter work is in the graph.
// Requires existing dflash/model/graph declarations before this header.
static ggml_tensor * vocab_scatter(ggml_context * ctx, ggml_tensor * scores,
                                  ggml_tensor * ids, int64_t vocab) {
    const int64_t k=ids->ne[0];
    auto * full=ggml_fill(ctx,ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,vocab,1),-INFINITY);
    auto * out=ggml_set_rows(ctx,full,ggml_reshape_3d(ctx,ggml_cont(ctx,scores),1,k,1),
                            ggml_reshape_3d(ctx,ggml_cont(ctx,ids),k,1,1));
    return ggml_reshape_2d(ctx,out,vocab,1);
}

static void build_dspark_vocab_head(llm_graph_context & g,const llama_model & model,ggml_tensor * tokens) {
    auto * ctx=g.ctx0;auto & res=g.res;auto * base=res->t_logits;auto * hidden=res->t_embd;
    const bool dynamic=model.vocab_router_down!=nullptr;
    const int64_t vocab=model.vocab.n_tokens(),ntok=hidden->ne[1];
    const int64_t k=dynamic?1024:4096;
    const auto anchor=model.gguf_kv.find("dflash.sample_from_anchor");
    if(g.ubatch.n_seqs_unq!=1 || ntok<1 || ntok>7 || g.cparams.causal_attn ||
       anchor==model.gguf_kv.end() || anchor->second!="true" || model.d2t ||
       !model.dspark_markov_w1 || !model.dspark_markov_w2 ||
       (g.cparams.dspark_markov_rows && ntok!=7))
        throw std::runtime_error("vocab reference requires C1 noncausal anchor-first DSpark block<=7");
    const int64_t markov_end=g.cparams.dspark_markov_rows?5:ntok;
    auto * prev=ggml_cont_1d(ctx,ggml_view_1d(ctx,tokens,1,0),1);
    ggml_tensor * cat=nullptr;ggml_tensor * confidence=nullptr;
    const bool has_conf=model.dspark_conf_proj!=nullptr;
    for(int64_t i=0;i<ntok;++i) {
        ggml_tensor * col=nullptr;ggml_tensor * conf=nullptr;
        const int64_t rows=dynamic?vocab:k;
        auto * position=ggml_view_2d(ctx,base,rows,1,base->nb[1],i*base->nb[1]);
        if(i<markov_end) {
            auto * latent=ggml_get_rows(ctx,model.dspark_markov_w1,prev);
            g.cb(latent,"dspark_markov_lookup",i);
            auto * bias=g.build_lora_mm(model.dspark_markov_w2,latent,model.dspark_markov_w2_s);
            g.cb(bias,"dspark_markov_projection",i); // original FULL-vocabulary Markov
            ggml_tensor * ids=model.vocab_static_ids;ggml_tensor * selected=nullptr;
            if(dynamic) {
                auto * combined=ggml_add(ctx,position,bias);
                ids=ggml_cont_1d(ctx,ggml_top_k(ctx,combined,k),k);
                g.cb(ids,"vocab_combined_topk",i);
                auto * gathered=ggml_get_rows(ctx,model.output,ids);
                g.cb(gathered,"vocab_original_row_gather",i);
                // Explicit floating exception: decoded original rows rounded F16.
                // This is not native Q8_1/MMVQ selected-row arithmetic.
                auto * packed=ggml_cast(ctx,gathered,GGML_TYPE_F16);
                g.cb(packed,"vocab_selected_rows_f16",i);
                auto * h=ggml_cont(ctx,ggml_view_2d(ctx,hidden,hidden->ne[0],1,hidden->nb[1],i*hidden->nb[1]));
                selected=ggml_mul_mat(ctx,packed,h);
                g.cb(selected,"vocab_selected_projection",i);
            } else selected=position;
            auto * selected_bias=ggml_reshape_2d(ctx,
                ggml_get_rows(ctx,ggml_reshape_2d(ctx,bias,1,vocab),ids),k,1);
            auto * scores=ggml_add(ctx,selected,selected_bias);
            col=vocab_scatter(ctx,scores,ids,vocab);
            g.cb(col,"dspark_markov_add",i);
            if(has_conf) {
                auto * h=ggml_cont(ctx,ggml_view_2d(ctx,hidden,hidden->ne[0],1,hidden->nb[1],i*hidden->nb[1]));
                auto * features=ggml_concat(ctx,h,latent,0);
                conf=ggml_mul_mat(ctx,model.dspark_conf_proj,features);
                if(model.dspark_conf_proj_b)conf=ggml_add(ctx,conf,model.dspark_conf_proj_b);
                conf=ggml_sigmoid(ctx,conf);
            }
            if(i+1<markov_end) {
                // Full-ID argmax after exact selected-row reranking. Every
                // dependent Markov lookup/topK suffix uses this actual parent.
                prev=ggml_argmax(ctx,col);g.cb(prev,"dspark_markov_argmax",i);
            }
        } else {
            // Unsubmitted D5 tail: preserve seven output/body rows, without
            // computing unused Markov projections or a hidden dense full head.
            col=dynamic?ggml_cont(ctx,position):vocab_scatter(ctx,position,model.vocab_static_ids,vocab);
            if(has_conf)conf=ggml_fill(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1,1),0.f);
        }
        cat=cat?ggml_concat(ctx,cat,col,1):col;
        if(has_conf)confidence=confidence?ggml_concat(ctx,confidence,conf,1):conf;
    }
    if(has_conf) {
        res->t_h_nextn=ggml_repeat(ctx,confidence,hidden);
        ggml_build_forward_expand(g.gf,res->t_h_nextn);
    }
    g.cb(cat,"dspark_markov_output",-1);res->t_logits=cat;
    ggml_build_forward_expand(g.gf,cat);
}
