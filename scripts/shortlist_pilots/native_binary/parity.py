#!/usr/bin/env python3
"""Run one native operator graph and compare trained PackedOperator reference.

Root invokes this under its admitted GPU guard; this file does not allocate a
lane or resume training. First7rows are actual captured inputs; two additional
rows exercise zero and ±.5/±1.5 RNE ties with row max127.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src"),str(ROOT/"third_party/llama.cpp/gguf-py")]
from shortlist_pilots.native_binary.export import sha
from shortlist_pilots.objectives import runtime
runtime.configure_imports()

def configure_reference_precision():
    def observed():
        return {"float32_matmul_precision":torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32":bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32":bool(torch.backends.cudnn.allow_tf32)}
    before=observed()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return {"before":before,"after":observed(),"matches_original_training_configure_cuda":True,
            "environment":{k:os.environ.get(k) for k in ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE","NVIDIA_TF32_OVERRIDE"]}}

def reference_stages(operator,inputs):
    from torch.nn import functional as F
    from w1a1_eagle.recurrent_qat import fixed_activation_codes
    x=inputs.float();mode=operator.description["type"];result={}
    outscale=operator.scales.expand(*x.shape[:-1],operator.scales.numel())
    if mode.startswith("mixture-scales"):
        logits=(operator.router.mean(0).expand(*x.shape[:-1],2)
                if mode=="mixture-scales-static" else x@operator.router)
        gate=logits.softmax(-1);inscale=gate@operator.input_experts
        result.update(router_logits=logits,gates=gate,input_scale=inscale)
        offsets=gate@operator.output_expert_offsets;result["output_offsets"]=offsets
        outscale=outscale*(1+offsets)
        x=x*inscale
    elif mode=="double-binary":
        result["input_scale"]=operator.input_scale.expand_as(x)
        x=x*operator.input_scale
    codes,beta,_=fixed_activation_codes(x,8)
    dot=F.linear(codes,operator.signs)
    result.update(scaled_input=x,actual_codes=codes,actual_beta=beta,output_scale=outscale,dot=dot)
    if mode=="double-binary":
        intermediate=dot*beta;result["intermediate"]=intermediate
        dot2=F.linear(intermediate,operator.factor_out);result["dot2"]=dot2
        result["final"]=dot2*outscale
    else:result["final"]=(dot*outscale)*beta
    return result

def append_fixtures(inputs):
    if inputs.shape[0]!=7 or inputs.shape[1]<8:raise ValueError("actual7row operator capture required")
    extra=np.zeros((2,inputs.shape[1]),np.float32)
    extra[1,:8]=[.5,1.5,-.5,-1.5,2.5,-2.5,127.,-127.]
    if inputs.shape[1]>=20:
        ties=np.array([.5,1.5,-.5,-1.5,2.5,-2.5],dtype=np.float32)
        extra[1,8:14]=np.nextafter(ties,np.float32(-np.inf))
        extra[1,14:20]=np.nextafter(ties,np.float32(np.inf))
    return np.concatenate([inputs,extra])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--helper",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
    p.add_argument("--model-sha256",required=True);p.add_argument("--input",type=Path,required=True)
    p.add_argument("--input-sha256",required=True);p.add_argument("--width",type=int,default=2560)
    p.add_argument("--operators",type=Path);p.add_argument("--operators-sha256")
    p.add_argument("--decoded-parent",action="store_true");p.add_argument("--out",type=Path,required=True)
    p.add_argument("--atol",type=float,default=2e-4);p.add_argument("--rtol",type=float,default=2e-4)
    p.add_argument("--trace-stages",action="store_true",help="diagnostic graph outputs; may alter fusion/planning")
    a=p.parse_args()
    precision_policy=configure_reference_precision()
    from source_gate import verify_training_source
    source_validation=verify_training_source(runtime.CAMPAIGN)
    from shortlist_pilots.binary.packed import PackedOperator, load_operator
    from w1a1_eagle.recurrent_qat import fixed_activation_codes
    if sha(a.model)!=a.model_sha256 or sha(a.input)!=a.input_sha256:raise ValueError("model/capture pin differs")
    import gguf
    reader=gguf.GGUFReader(a.model);mode=reader.fields["dflash.binary.kind"].contents();prefix="blk.4.ffn_up"
    a.out.mkdir(parents=True,exist_ok=False)
    inputs=append_fixtures(np.fromfile(a.input,dtype="<f4").reshape(7,a.width))
    if not np.isfinite(inputs).all():raise ValueError("finite captured inputs required")
    input_path=a.out/"input-plus-ties.f32";inputs.tofile(input_path);raw=a.out/"native.f32"
    command=[str(a.helper),str(a.model),mode,str(input_path),str(raw),str(len(inputs))]
    if a.trace_stages:command.append("trace")
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    if a.decoded_parent:
        native={t.name:t.data for t in reader.tensors}
        operator=PackedOperator({"type":"native-row-binary","input":a.width},
            {"signs":native[prefix+".binary_signs"],"scales":native[prefix+".binary_scales"]})
    else:
        if not a.operators or not a.operators_sha256:raise ValueError("trained operator source pin required")
        operator=load_operator(a.operators,prefix,a.operators_sha256)
    operator=operator.to("cuda:0").eval()
    with torch.no_grad():
        gpu_inputs=torch.from_numpy(inputs).to("cuda:0")
        reference=torch.cat([operator(gpu_inputs[:7]),operator(gpu_inputs[7:])]).cpu().numpy()
        expected_codes=fixed_activation_codes(gpu_inputs,8)[0].cpu().numpy()
        stage_references=None
        if a.trace_stages:
            chunks=[reference_stages(operator,gpu_inputs[:7]),reference_stages(operator,gpu_inputs[7:])]
            stage_references={name:torch.cat([c[name] for c in chunks]).cpu().numpy() for name in chunks[0]}
    actual=np.fromfile(raw,dtype="<f4").reshape(reference.shape)
    actual_codes=np.fromfile(str(raw)+".codes.f32",dtype="<f4").reshape(inputs.shape)
    codes_exact=bool(np.array_equal(actual_codes,expected_codes))
    error=np.abs(actual-reference);passed=codes_exact and bool(np.allclose(actual,reference,atol=a.atol,rtol=a.rtol,equal_nan=False))
    report={"schema":"native_binary_operator_parity_v1","passed":passed,"mode":mode,
        "actual_captured_rows":7,"fixture_rows":2,"shape":list(reference.shape),"atol":a.atol,"rtol":a.rtol,
        "actual_graph_rows":7,"fixture_graph_rows":2,"A8_codes_exact":codes_exact,
        "reference_source_validation":source_validation,
        "reference_precision_policy":precision_policy,
        "A8_code_mismatches":int(np.count_nonzero(actual_codes!=expected_codes)),
        "fixture_tie_codes":actual_codes[8,:8].tolist(),
        "fixture_neighbor_codes":actual_codes[8,8:20].tolist(),
        "max_absolute_error":float(error.max()),"rms_error":float(np.sqrt(np.mean(error.astype(np.float64)**2))),
        "actual7row_max_absolute_error":float(error[:7].max()),"reference":"trained PackedOperator CUDA F32",
        "source":{"model":{"path":str(a.model),"sha256":a.model_sha256},"input":{"path":str(a.input),"sha256":a.input_sha256},"helper":{"path":str(a.helper),"sha256":sha(a.helper)}},
        "command":command,"helper_stdout":result.stdout,"helper_stderr":result.stderr,
        "scope":"operator semantic gate only; subnormal reference exception remains explicit; no throughput claim"}
    if a.trace_stages:
        comparisons={}
        for name,expected in stage_references.items():
            path=Path(str(raw)+".stage-"+name+".f32")
            observed=np.fromfile(path,dtype="<f4").reshape(expected.shape)
            diff=np.abs(observed-expected)
            comparisons[name]={"shape":list(expected.shape),"max_absolute_error":float(diff.max()),
                "rms_error":float(np.sqrt(np.mean(diff.astype(np.float64)**2))),
                "exact":bool(np.array_equal(observed,expected)),"mismatched_values":int(np.count_nonzero(observed!=expected)),
                "within_output_tolerance":bool(np.allclose(observed,expected,atol=a.atol,rtol=a.rtol)),
                "actual7row_max_absolute_error":float(diff[:7].max()),"native_path":str(path)}
        report["diagnostic_stages"]=comparisons
        report["actual_scaled_A8_codes_exact"]=comparisons["actual_codes"]["exact"]
        report["diagnostic_graph_warning"]="Additional outputs may alter graph fusion/planning; reconfirm repaired plain graph separately."
    (a.out/"parity.json").write_text(json.dumps(report,indent=2)+"\n")
    if not passed:raise SystemExit(2)

if __name__=="__main__":main()
