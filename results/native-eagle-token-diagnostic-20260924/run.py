import json, os, pathlib, subprocess, time, urllib.request, urllib.error, hashlib, signal, tomllib
root=pathlib.Path(__file__).resolve().parents[2]
out=pathlib.Path(__file__).resolve().parent
cfg=tomllib.loads((root/"configs/native_benchmark.toml").read_text())
prompts={x["id"]:x for x in [json.loads(s) for s in (root/"configs/acceptance_prompts.jsonl").read_text().splitlines() if s.strip()]}
gpu_cmd=["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader"]
def gpu():
    p=subprocess.run(gpu_cmd,capture_output=True,text=True)
    return {"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"command":gpu_cmd}
def request(url,body,timeout):
    data=json.dumps(body,separators=(",",":")).encode()
    req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r: return r.status,r.read().decode()
    except urllib.error.HTTPError as e: return e.code,e.read().decode()
def wait_ready(url,proc,logpath):
    deadline=time.time()+300
    while time.time()<deadline:
        if proc.poll() is not None: raise RuntimeError("server exited before ready; see "+str(logpath))
        try:
            with urllib.request.urlopen(url+"/health",timeout=2) as r:
                if r.status==200:return
        except Exception: time.sleep(1)
    raise TimeoutError("server did not become ready")
def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=15)
        else:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
summary={"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"submodule":subprocess.check_output(["git","-C","third_party/llama.cpp","rev-parse","HEAD"],cwd=root,text=True).strip(),"config_sha256":hashlib.sha256((root/"configs/native_benchmark.toml").read_bytes()).hexdigest(),"prompts_sha256":hashlib.sha256((root/"configs/acceptance_prompts.jsonl").read_bytes()).hexdigest(),"prompts":["prose-04","reasoning-04"],"max_tokens":128,"temperature":0.0,"seed":42,"variants":{},"gpu_before":gpu()}
port=18082
for variant,draft_key in [("target_only",None),("ordinary_eagle","ordinary_draft"),("packed_head_w1a1","packed_head_draft")]:
    server=cfg["server"]; models=cfg["models"]; cmd=[str(root/server["binary"]),"-m",str(root/models["target"])]
    if draft_key is None: cmd += ["--spec-type","none"]
    else: cmd += ["-md",str(root/models[draft_key]),"--spec-type","draft-eagle3","--spec-draft-n-max",str(cfg["evaluation"]["max_draft_tokens"]),"--spec-draft-ngl","all","--spec-draft-type-k","f16","--spec-draft-type-v","f16"]
    cmd += list(server["common_args"])+["--host","127.0.0.1","--port",str(port)]
    d=out/variant; d.mkdir(parents=True,exist_ok=True); logpath=d/"server.log"
    with logpath.open("wb") as log:
        proc=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
        try:
            url="http://127.0.0.1:"+str(port); wait_ready(url,proc,logpath)
            entry={"command":cmd,"gpu_loaded":gpu(),"responses":{}}
            for pid in ("prose-04","reasoning-04"):
                prompt=prompts[pid]; body={"messages":prompt["messages"],"max_tokens":128,"temperature":0.0,"seed":42,"stream":False,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False},"reasoning_format":"none","return_tokens":True,"verbose":True}
                (d/(pid+"-request.json")).write_text(json.dumps(body,indent=2)+"\n")
                status,raw=request(url+"/v1/chat/completions",body,180)
                rp=d/(pid+"-response.json"); rp.write_text(raw+"\n")
                if status!=200: raise RuntimeError(variant+" "+pid+" HTTP "+str(status)+": "+raw[:500])
                obj=json.loads(raw); choice=(obj.get("choices") or [{}])[0]; verbose=obj.get("__verbose") or {}
                toks=verbose.get("tokens")
                if toks is None:toks=obj.get("tokens")
                entry["responses"][pid]={"status":status,"finish_reason":choice.get("finish_reason"),"completion_tokens":(obj.get("usage") or {}).get("completion_tokens"),"text":(choice.get("message") or {}).get("content"),"token_ids":toks,"token_array_field":"__verbose.tokens" if "__verbose" in obj and "tokens" in obj["__verbose"] else ("tokens" if "tokens" in obj else None),"response_sha256":hashlib.sha256(raw.encode()).hexdigest()}
            summary["variants"][variant]=entry
        finally:
            stop(proc); summary["variants"].setdefault(variant,{}).update({"server_exit_after_stop":proc.poll(),"gpu_after":gpu()})
    port+=1
ref={pid:summary["variants"]["target_only"]["responses"][pid].get("token_ids") for pid in ("prose-04","reasoning-04")}
for variant in ("ordinary_eagle","packed_head_w1a1"):
    for pid in ("prose-04","reasoning-04"):
        toks=summary["variants"][variant]["responses"][pid].get("token_ids"); target=ref[pid]
        if isinstance(toks,list) and isinstance(target,list):
            n=min(len(toks),len(target)); idx=next((i for i in range(n) if toks[i]!=target[i]),None)
            summary["variants"][variant]["responses"][pid]["token_id_match_target"]=toks==target
            summary["variants"][variant]["responses"][pid]["first_diff_target"]={"index":idx,"target":target[idx] if idx is not None else None,"variant":toks[idx] if idx is not None else None,"target_len":len(target),"variant_len":len(toks)}
        else:summary["variants"][variant]["responses"][pid]["token_id_match_target"]=None
for pid in ("prose-04","reasoning-04"):
    a=summary["variants"]["ordinary_eagle"]["responses"][pid].get("token_ids"); b=summary["variants"]["packed_head_w1a1"]["responses"][pid].get("token_ids")
    summary.setdefault("ordinary_packed_token_match",{})[pid]=a==b if isinstance(a,list) and isinstance(b,list) else None
(out/"token-diagnostic-summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
