#!/usr/bin/env python3
"""Exclusive decoded-F32 GGUF export of trained portable binary operators.

Every unchanged native payload is copied byte-for-byte. All original13 sign/scale
pairs and all22 trained norms come from the authenticated portable model state;
the selected projection is replaced by separate F32 operators, never collapsed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src"), str(ROOT/"third_party/llama.cpp/gguf-py")]

def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()

def norm_name(name):
    if name == "output_norm_enc":return "enc.output_norm.weight"
    if name == "output_norm":return "output_norm.weight"
    parts=name.split(".")
    if len(parts)==3 and parts[0]=="layers" and parts[2] in {"attn_norm","attn_q_norm","attn_k_norm","ffn_norm"}:
        return "blk."+parts[1]+"."+parts[2]+".weight"
    raise ValueError("unexpected trained nonbinary parameter: "+name)

def unpack(bits, width):
    return np.unpackbits(bits,axis=-1,bitorder="little")[:,:width].astype(np.float32)*2-1

def native_words(bits, width):
    raw=np.unpackbits(bits,axis=-1,bitorder="little")[:,:width]
    raw=np.pad(raw,((0,0),(0,(-width)%32)))
    return np.packbits(raw,axis=-1,bitorder="little").copy().view("<i4")

def validate_checkpoint_payload(payload, descriptions, state):
    import torch
    def exact(left,right,message):
        left,right=np.asarray(left),np.asarray(right)
        if left.shape!=right.shape or left.dtype!=right.dtype or left.tobytes()!=right.tobytes():
            raise ValueError("operator/checkpoint bytes differ: "+message)
    for name,desc in descriptions.items():
        saved=state["linears"][name];key=name.replace(".","_")+"__"
        latent=saved["latent_sign"].detach().numpy()
        if not np.isfinite(latent).all():raise ValueError("nonfinite checkpoint latent")
        raw=saved["initial_scale"]+saved["scale_offset"]
        scales=torch.where(raw>=0,raw,torch.zeros_like(raw)).numpy()
        exact(payload[key+"scales"],scales,name+" scales")
        if desc["type"]=="ternary-a8":
            threshold=saved["threshold"].numpy()
            ternary=np.where(np.abs(latent)<threshold,0,np.where(latent<0,-1,1))
            packed=payload[key+"ternary2"]
            codes=np.stack([(packed>>(2*i))&3 for i in range(4)],axis=-1).reshape(desc["output"],-1)[:,:desc["input"]]
            if not np.array_equal(np.where(codes==2,-1,codes.astype(np.int8)),ternary):raise ValueError("ternary checkpoint codes differ")
        else:
            expected=np.packbits(latent>=0,axis=-1,bitorder="little")
            exact(payload[key+"signs"],expected,name+" signs")
        if desc["type"].startswith("mixture-scales"):
            for field in ("router","input_experts","output_expert_offsets"):
                exact(payload[key+field],saved[field].numpy(),name+" "+field)
        elif desc["type"]=="double-binary":
            exact(payload[key+"factor_out"],np.packbits(saved["factor_out"].numpy()>=0,axis=-1,bitorder="little"),name+" factor signs")
            exact(payload[key+"input_scale"],saved["input_scale"].numpy(),name+" input scales")

def operator_arrays(payload, description, projection):
    key=projection.replace(".","_")+"__";m,n=description["output"],description["input"]
    kind=description["type"]
    values={"binary_scales":np.asarray(payload[key+"scales"],dtype=np.float32)}
    if kind=="ternary-a8":
        p=payload[key+"ternary2"]
        codes=np.stack([(p>>(2*i))&3 for i in range(4)],axis=-1).reshape(m,-1)[:,:n].astype(np.int8)
        if (codes==3).any():raise ValueError("invalid ternary code")
        values["binary_signs"]=np.where(codes==2,-1,codes).astype(np.float32)
    else:
        values["binary_signs"]=unpack(payload[key+"signs"],n)
    if kind.startswith("mixture-scales"):
        values.update(binary_router=payload[key+"router"].T.copy(),
                      binary_input_experts=payload[key+"input_experts"].T.copy(),
                      binary_output_offsets=payload[key+"output_expert_offsets"].T.copy())
        mode="mos-static" if kind.endswith("static") else "mos"
    elif kind=="double-binary":
        rank=description["factor_out_shape"][1]
        if rank not in (32,1984):raise ValueError("native factor rank must be32 or1984")
        values.update(binary_factor_out=unpack(payload[key+"factor_out"],rank),
                      binary_input_scale=payload[key+"input_scale"].copy())
        mode="double-binary"
    elif kind=="ternary-a8":mode=kind
    else:raise ValueError("only trained MoS/factor/ternary portable representations accepted")
    for name,array in values.items():
        if array.dtype!=np.float32 or not np.isfinite(array).all():raise ValueError("finite F32 operator tensor required: "+name)
    return mode,{projection+"."+k:np.ascontiguousarray(v) for k,v in values.items()}

def export(parent,parent_sha,output,*,operators=None,operators_sha=None,checkpoint=None,checkpoint_sha=None,
           decoded_parent=False,projection="blk.4.ffn_up"):
    import gguf
    if projection!="blk.4.ffn_up":raise ValueError("bounded native screen supports only original blk.4.ffn_up")
    parent,output=Path(parent),Path(output)
    if output.exists() or output.with_suffix(".json").exists():raise ValueError("exclusive fresh export required")
    if sha(parent)!=parent_sha:raise ValueError("parent donor digest differs")
    reader=gguf.GGUFReader(parent);native={t.name:t for t in reader.tensors}
    if len(native)!=len(reader.tensors):raise ValueError("duplicate donor tensor")
    if reader.fields.get("dflash.binary.version") is not None:raise ValueError("donor already has a binary reference override")
    replacements={};removed={projection+".w1a1_packed",projection+".w1a1_scale"};source={"parent":{"path":str(parent),"sha256":parent_sha}}
    if decoded_parent:
        if any(v is not None for v in (operators,checkpoint)):raise ValueError("decoded parent must use its original donor only")
        packed=native[projection+".w1a1_packed"].data
        width=int(native[projection+".w1a1_packed"].shape[0])*32
        signs=unpack(np.ascontiguousarray(packed).view(np.uint8).reshape(packed.shape[0],-1),width)
        replacements={projection+".binary_signs":signs,projection+".binary_scales":native[projection+".w1a1_scale"].data.copy()}
        mode="decoded-parent"
    else:
        import torch
        if not all(v is not None for v in (operators,operators_sha,checkpoint,checkpoint_sha)):
            raise ValueError("authenticated operator payload and checkpoint required")
        if sha(operators)!=operators_sha or sha(checkpoint)!=checkpoint_sha:raise ValueError("trained payload/checkpoint digest differs")
        description=json.loads(Path(operators).with_suffix(".json").read_text())
        if description["sha256"]!=operators_sha or description["native_gguf"] is not False:raise ValueError("portable operator receipt differs")
        state=torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=False)
        if state["schema"]!="shortlist_binary_fullmodel_v1":raise ValueError("portable checkpoint schema differs")
        checkpoint_parent=state["source"].get("parent")
        if checkpoint_parent and checkpoint_parent["model"]["sha256"]!=parent_sha:raise ValueError("checkpoint's original native donor differs")
        if set(description["operators"])!=set(state["linears"]):raise ValueError("operator/checkpoint inventories differ")
        with np.load(operators,allow_pickle=False) as archive:
            validate_checkpoint_payload(archive,description["operators"],state)
            mode,custom=operator_arrays(archive,description["operators"][projection],projection)
            for name,desc in description["operators"].items():
                if name==projection:continue
                if desc["type"]!="native-row-binary":raise ValueError("only one transformed projection is qualified")
                key=name.replace(".","_")+"__"
                replacements[name+".w1a1_packed"]=native_words(archive[key+"signs"],desc["input"])
                replacements[name+".w1a1_scale"]=archive[key+"scales"].copy()
            replacements.update(custom)
        norms=state["nonlinear_parameters"]
        if len(norms)!=22:raise ValueError("full trained48 model requires22 explicit trained norm tensors")
        for name,value in norms.items():replacements[norm_name(name)]=value.detach().numpy().copy()
        source.update(operators={"path":str(operators),"sha256":operators_sha},checkpoint={"path":str(checkpoint),"sha256":checkpoint_sha})
        del state
    if not removed<=set(native):raise ValueError("donor lacks exclusive selected binary pair")
    if projection+".weight" in native:raise ValueError("unexpected dense shadow in donor")
    for name,value in replacements.items():
        if not np.isfinite(value).all():raise ValueError("nonfinite export tensor: "+name)
        if name in native and name not in removed:
            if value.shape!=native[name].data.shape or value.dtype!=native[name].data.dtype:
                raise ValueError("unchanged codec tensor shape/type differs: "+name)
    output.parent.mkdir(parents=True,exist_ok=True)
    metadata={"dflash.binary.version":1,"dflash.binary.kind":mode,"dflash.binary.projection":projection,
              "dflash.binary.arithmetic":"decoded_F32_A8_RNE_reference_v1",
              "dflash.binary.a8_exception":"subnormal rows clamp denominator to FLT_MIN; normal rows reciprocal RNE",
              "dflash.binary.storage":"decoded F32 signs/factors; not packed-W1 BPW or native-W1 performance"}
    with tempfile.TemporaryDirectory(dir=output.parent,prefix=".binary-native-") as directory:
        stage=Path(directory)/"model.gguf";writer=gguf.GGUFWriter(stage,"dflash")
        for key,v in reader.fields.items():
            if key.startswith("GGUF.") or key=="general.architecture":continue
            writer.add_key_value(key,v.contents(),v.types[0],v.types[-1] if len(v.types)>1 else None)
        for key,value in metadata.items():
            if isinstance(value,int):writer.add_uint32(key,value)
            else:writer.add_string(key,value)
        for name,tensor in native.items():
            if name in removed:continue
            writer.add_tensor(name,replacements.get(name,tensor.data),raw_dtype=tensor.tensor_type)
        for name,value in replacements.items():
            if name not in native:writer.add_tensor(name,value.astype(np.float32,copy=False))
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        check=gguf.GGUFReader(stage);actual={t.name:t for t in check.tensors}
        expected=(set(native)-removed)|(set(replacements)-removed)
        if set(actual)!=expected:raise ValueError("exclusive output tensor inventory differs")
        for name,tensor in actual.items():
            source_array=replacements[name] if name in replacements else native[name].data
            if tensor.data.tobytes()!=source_array.tobytes():raise ValueError("exact payload export differs: "+name)
        for key,v in reader.fields.items():
            if key.startswith("GGUF."):continue
            if key not in check.fields or check.fields[key].contents()!=v.contents():raise ValueError("donor metadata changed: "+key)
        digest=sha(stage);os.link(stage,output)
    report={"schema":"decoded_binary_native_export_v1","source":source,"output":{"path":str(output),"sha256":digest},
            "mode":mode,"projection":projection,"arithmetic":metadata,"removed_original_pair":sorted(removed),
            "replaced_tensors":sorted(replacements),"decoded_operator_bytes":sum(v.nbytes for k,v in replacements.items() if ".binary_" in k),
            "trained_payloads_exact_F32":True,"all_other_tensor_bytes_preserved":True,"native_GPU_qualified":False}
    with output.with_suffix(".json").open("x") as stream:json.dump(report,stream,indent=2);stream.write("\n")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--parent-sha256",required=True)
    p.add_argument("--operators",type=Path);p.add_argument("--operators-sha256")
    p.add_argument("--checkpoint",type=Path);p.add_argument("--checkpoint-sha256")
    p.add_argument("--decoded-parent",action="store_true")
    a=p.parse_args();print(json.dumps(export(a.parent,a.parent_sha256,a.output,operators=a.operators,operators_sha=a.operators_sha256,
        checkpoint=a.checkpoint,checkpoint_sha=a.checkpoint_sha256,decoded_parent=a.decoded_parent)))

if __name__=="__main__":main()
