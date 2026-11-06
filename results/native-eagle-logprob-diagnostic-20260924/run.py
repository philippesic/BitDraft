import json,os,pathlib,subprocess,time,urllib.request,urllib.error,hashlib,signal,tomllib
root=pathlib.Path("/home/philip/binary-eagle-decoding"); out=root/"results/native-eagle-logprob-diagnostic-20260924"
cfg=tomllib.loads((root/"configs/native_benchmark.toml").read_text())
prompts={x["id"]:x for x in [json.loads(s) for s in (root/"configs/acceptance_prompts.jsonl").read_text().splitlines() if s.strip()]}
old=json.load(open(root/"results/native-eagle-token-diagnostic-20260924/token-diagnostic-summary.json"))
def gpu():
 return subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.free,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout
def post(url,b):
 d=json.dumps(b,separators=(",",":")).encode();q=urllib.request.Request(url,data=d,headers={"Content-Type":"application/json"},method="POST")
 try:
  with urllib.request.urlopen(q,timeout=180) as r:return r.status,r.read().decode()
 except urllib.error.HTTPError as e:return e.code,e.read().decode()
def ready(url,p):
 end=time.time()+300
 while time.time()<end:
  if p.poll() is not None:raise RuntimeError("server exited before ready")
  try:
   if urllib.request.urlopen(url+"/health",timeout=2).status==200:return
  except Exception:time.sleep(1)
 raise TimeoutError("server timeout")
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=15)
s={"code_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),"submodule":subprocess.check_output(["git","-C",str(root/"third_party/llama.cpp"),"rev-parse","HEAD"],text=True).strip(),"seed":42,"temperature":0.0,"max_tokens":128,"n_probs":5,"gpu_before":gpu(),"sequence_unchanged":True,"variants":{}}
for variant,dk,port in (("target_only",None,18084),("ordinary_eagle","ordinary_draft",18085)):
 sc=cfg["server"];m=cfg["models"];cmd=[str(root/sc["binary"]),"-m",str(root/m["target"])]
 if dk is None:cmd+=["--spec-type","none"]
 else:cmd+=["-md",str(root/m[dk]),"--spec-type","draft-eagle3","--spec-draft-n-max","5","--spec-draft-ngl","all","--spec-draft-type-k","f16","--spec-draft-type-v","f16"]
 cmd+=list(sc["common_args"])+["--host","127.0.0.1","--port",str(port)]
 d=out/variant;d.mkdir(parents=True,exist_ok=True);entry={"command":cmd,"gpu_loaded":gpu(),"responses":{}}
 with (d/"server.log").open("wb") as log:
  p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
  try:
   url="http://127.0.0.1:"+str(port);ready(url,p)
   for pid in ("prose-04","reasoning-04"):
    b={"messages":prompts[pid]["messages"],"max_tokens":128,"temperature":0.0,"seed":42,"stream":False,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False},"reasoning_format":"none","return_tokens":True,"verbose":True,"n_probs":5}
    (d/(pid+"-request.json")).write_text(json.dumps(b,indent=2)+"\n");status,raw=post(url+"/v1/chat/completions",b);(d/(pid+"-response.json")).write_text(raw+"\n")
    item={"http_status":status,"response_sha256":hashlib.sha256(raw.encode()).hexdigest()}
    if status==200:
     obj=json.loads(raw);ch=(obj.get("choices") or [{}])[0];vb=obj.get("__verbose") or {};ids=vb.get("tokens",obj.get("tokens"));before=old["variants"][variant]["responses"][pid]["token_ids"]
     item.update({"finish_reason":ch.get("finish_reason"),"completion_tokens":(obj.get("usage") or {}).get("completion_tokens"),"token_ids":ids,"sequence_matches_previous":ids==before,"response_keys":list(obj.keys())})
     if ids==before:item["_obj"]=obj
     else:s["sequence_unchanged"]=False
    else:item["error_body"]=raw;s["sequence_unchanged"]=False
    entry["responses"][pid]=item
    if not s["sequence_unchanged"]:break
  finally:stop(p);entry["server_exit_after_stop"]=p.poll();entry["gpu_after"]=gpu()
 s["variants"][variant]=entry
 if not s["sequence_unchanged"]:break
if s["sequence_unchanged"]:
 for pid,ix in (("prose-04",33),("reasoning-04",123)):
  row={"index":ix};ids={}
  for v in ("target_only","ordinary_eagle"):
   x=s["variants"][v]["responses"][pid];obj=x.pop("_obj");tok=x["token_ids"];ids[v]=tok[ix];lp=(obj["choices"][0].get("logprobs") or {}).get("content") or [];e=lp[ix] if ix<len(lp) else {};top=e.get("top_logprobs") or []
   row[v]={"generated_id":tok[ix],"generated_logprob":e.get("logprob"),"top5":[{"id":z.get("id"),"logprob":z.get("logprob"),"token":z.get("token")} for z in top],"margin_top1_top2":top[0]["logprob"]-top[1]["logprob"] if len(top)>1 else None}
  cand=[ids["target_only"],ids["ordinary_eagle"]]
  for v in ("target_only","ordinary_eagle"):row[v]["both_ids_in_top5"]={str(i):i in [z["id"] for z in row[v]["top5"]] for i in cand}
  row["target_id"]=cand[0];row["ordinary_id"]=cand[1];s.setdefault("indices",{})[pid]=row
(out/"summary.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
