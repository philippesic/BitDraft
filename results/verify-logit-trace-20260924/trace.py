import hashlib,json,os,pathlib,re,signal,subprocess,time,tomllib,urllib.request,urllib.error
root=pathlib.Path("/home/philip/binary-eagle-decoding")
out=root/"results/verify-logit-trace-20260924";out.mkdir(parents=True,exist_ok=True)
cfg=tomllib.loads((root/"configs/native_benchmark.toml").read_text())
prompts={x["id"]:x for x in [json.loads(s) for s in (root/"configs/acceptance_prompts.jsonl").read_text().splitlines() if s.strip()]}
previous=json.load(open(root/"results/native-eagle-token-diagnostic-20260924/token-diagnostic-summary.json"))
expected={"prose-04":33,"reasoning-04":123}
def gpu():
 return subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.used,memory.free,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout
def request(url,body):
 d=json.dumps(body,separators=(",",":")).encode()
 q=urllib.request.Request(url,data=d,headers={"Content-Type":"application/json"},method="POST")
 try:
  with urllib.request.urlopen(q,timeout=180) as r:return r.status,r.read().decode()
 except urllib.error.HTTPError as e:return e.code,e.read().decode()
def ready(url,p):
 end=time.time()+300
 while time.time()<end:
  if p.poll() is not None:raise RuntimeError("trace server exited before ready")
  try:
   with urllib.request.urlopen(url+"/health",timeout=2) as r:
    if r.status==200:return
  except Exception:time.sleep(1)
 raise TimeoutError("trace server startup timed out")
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=15)
  else:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
server=cfg["server"];models=cfg["models"];port=18088
cmd=[str(root/"build/llama-cuda-verifier-trace-20260924/bin/llama-server"),"-m",str(root/models["target"]),"-md",str(root/models["ordinary_draft"]),"--spec-type","draft-eagle3","--spec-draft-n-max","5","--spec-draft-ngl","all","--spec-draft-type-k","f16","--spec-draft-type-v","f16"]+list(server["common_args"])+["--host","127.0.0.1","--port",str(port)]
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="0";env["W1A1_TRACE_VERIFY_LOGITS"]="1"
summary={"code_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),"llama_cpp_commit":subprocess.check_output(["git","-C",str(root/"third_party/llama.cpp"),"rev-parse","HEAD"],text=True).strip(),"config_sha256":hashlib.sha256((root/"configs/native_benchmark.toml").read_bytes()).hexdigest(),"prompts_sha256":hashlib.sha256((root/"configs/acceptance_prompts.jsonl").read_bytes()).hexdigest(),"server_env":{"CUDA_VISIBLE_DEVICES":"0","W1A1_TRACE_VERIFY_LOGITS":"1"},"server_command":cmd,"request_settings":{"temperature":0.0,"seed":42,"max_tokens":128,"cache_prompt":False,"enable_thinking":False,"return_tokens":True,"verbose":True},"gpu_before":gpu(),"prompts":{},"stopped_early":False}
logpath=out/"server.log";line_count=0
with logpath.open("wb") as log:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  base="http://127.0.0.1:"+str(port);ready(base,p)
  for pid in ("prose-04","reasoning-04"):
   body={"messages":prompts[pid]["messages"],"max_tokens":128,"temperature":0.0,"seed":42,"stream":False,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False},"reasoning_format":"none","return_tokens":True,"verbose":True}
   d=out/pid;d.mkdir(exist_ok=True)
   (d/"request.json").write_text(json.dumps(body,indent=2)+"\n")
   status,raw=request(base+"/v1/chat/completions",body);(d/"response.json").write_text(raw+"\n")
   meta={"http_status":status,"response_sha256":hashlib.sha256(raw.encode()).hexdigest()}
   if status!=200:
    meta["error"]=raw;summary["prompts"][pid]=meta;summary["stopped_early"]=True;break
   obj=json.loads(raw);choice=(obj.get("choices") or [{}])[0];tokens=(obj.get("__verbose") or {}).get("tokens")
   old=previous["variants"]["ordinary_eagle"]["responses"][pid]["token_ids"];target=previous["variants"]["target_only"]["responses"][pid]["token_ids"];diff=next((i for i,(a,b) in enumerate(zip(tokens,target)) if a!=b),None)
   trace_rows=[line for line in logpath.read_text(errors="replace").splitlines()[line_count:] if "verify_logits pos=" in line];line_count=len(logpath.read_text(errors="replace").splitlines())
   pos=expected[pid];selected=[line for line in trace_rows if "verify_logits pos="+str(pos)+" " in line]
   null_rows=[line for line in selected if "logits=null" in line]
   meta.update({"finish_reason":choice.get("finish_reason"),"completion_tokens":(obj.get("usage") or {}).get("completion_tokens"),"token_ids":tokens,"matches_sealed_ordinary_ids":tokens==old,"first_diff_vs_target":diff,"expected_first_diff":pos,"trace_rows":selected,"trace_rows_for_expected_position":len(selected),"null_logit_rows":null_rows,"response_keys":list(obj.keys())})
   summary["prompts"][pid]=meta
   if tokens!=old or diff!=pos or null_rows or not selected:
    summary["stopped_early"]=True;break
 finally:
  stop(p);summary["server_exit_code"]=p.poll();summary["gpu_after_server"]=gpu()
text=(out/"server.log").read_text(errors="replace")
summary["all_verify_logits_rows"]=[line for line in text.splitlines() if "verify_logits pos=" in line]
summary["gpu_after"]=gpu()
(out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
