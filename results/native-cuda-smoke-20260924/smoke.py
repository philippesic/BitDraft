import hashlib,json,os,signal,subprocess,sys,time,urllib.error,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/native-cuda-smoke-20260924"
OUT.mkdir(parents=True,exist_ok=True)
TARGET=ROOT/"models/gguf/Qwen3-4B-f16.gguf"
PACKED=ROOT/"models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf"
BINARY=ROOT/"build/llama-cuda/bin/llama-server"
PORT=18081
BASE=f"http://127.0.0.1:{PORT}"
MARKER="CUDA packed W1A1 XOR/POPCOUNT dispatch"
def gpu():
    r=subprocess.run(["nvidia-smi","--query-gpu=timestamp,name,driver_version,memory.used,memory.free,utilization.gpu","--format=csv"],capture_output=True,text=True)
    return {"exit_code":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
def req(url,payload=None,timeout=30):
    data=None if payload is None else json.dumps(payload).encode()
    q=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(q,timeout=timeout) as resp:return resp.status,resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:return e.code,e.read().decode(errors="replace")
def metric_values(body):
    wanted={"llamacpp:spec_decode_num_draft_tokens_total":"proposed","llamacpp:spec_decode_num_accepted_tokens_total":"accepted","llamacpp:spec_decode_num_drafts_total":"rounds"}
    d={}
    for line in body.splitlines():
        s=line.strip().split()
        if len(s)>=2 and s[0] in wanted:
            try:d[wanted[s[0]]]=int(float(s[1]))
            except ValueError:pass
    return d
prompt=json.loads((ROOT/"configs/acceptance_prompts.jsonl").read_text().splitlines()[0])
(OUT/"prompt.jsonl").write_text(json.dumps(prompt,ensure_ascii=False,separators=(",",":"))+"\n")
cmd=[str(BINARY),"-m",str(TARGET),"-md",str(PACKED),"--spec-type","draft-eagle3","--spec-draft-n-max","5","--spec-draft-ngl","all","--spec-draft-type-k","f16","--spec-draft-type-v","f16","--n-gpu-layers","all","--ctx-size","2048","--parallel","1","--fit","off","--cache-type-k","f16","--cache-type-v","f16","--jinja","--metrics","--perf","-lv","4","--host","127.0.0.1","--port",str(PORT)]
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="0"
(OUT/"command.json").write_text(json.dumps({"argv":cmd,"env":{"CUDA_VISIBLE_DEVICES":"0"},"prompt_id":prompt["id"],"max_tokens":16},indent=2)+"\n")
(OUT/"gpu-before.csv").write_text(gpu()["stdout"])
with (OUT/"server.log").open("wb") as log:
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            if p.poll() is not None:raise RuntimeError(f"server exited early: {p.returncode}")
            try:
                status,body=req(BASE+"/health",timeout=3)
                if status==200 and json.loads(body).get("status")=="ok":break
            except Exception:pass
            time.sleep(.5)
        else:raise TimeoutError("server health timeout")
        (OUT/"gpu-loaded.csv").write_text(gpu()["stdout"])
        status,before=req(BASE+"/metrics")
        (OUT/"metrics-before.prom").write_text(before)
        if status!=200:raise RuntimeError(f"metrics-before HTTP {status}")
        body={"messages":prompt["messages"],"max_tokens":16,"temperature":0.0,"seed":42,"stream":False,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False},"reasoning_format":"none"}
        (OUT/"request.json").write_text(json.dumps(body,indent=2)+"\n")
        status,raw=req(BASE+"/v1/chat/completions",body,timeout=180)
        (OUT/"response.json").write_text(raw)
        if status!=200:raise RuntimeError(f"completion HTTP {status}")
        response=json.loads(raw)
        status,after=req(BASE+"/metrics")
        (OUT/"metrics-after.prom").write_text(after)
        if status!=200:raise RuntimeError(f"metrics-after HTTP {status}")
        b=metric_values(before);a=metric_values(after)
        delta={k:(a[k]-b[k]) if k in a and k in b else None for k in ("proposed","accepted","rounds")}
        if any(v is None for v in delta.values()):raise RuntimeError(f"missing speculative counters: {delta}")
        choice=(response.get("choices") or [{}])[0]
        result={"status":"ok","prompt_id":prompt["id"],"http_status":status,"finish_reason":choice.get("finish_reason"),"completion_tokens":(response.get("usage") or {}).get("completion_tokens"),"completion_sha256":hashlib.sha256(str((choice.get("message") or {}).get("content","")).encode()).hexdigest(),"metrics_before":b,"metrics_after":a,"counter_delta":delta,"accepted_per_round":delta["accepted"]/delta["rounds"] if delta["rounds"] else None,"acceptance_rate":delta["accepted"]/delta["proposed"] if delta["proposed"] else None,"dispatch_marker":MARKER,"server_argv":cmd,"gpu_loaded":(OUT/"gpu-loaded.csv").read_text()}
        (OUT/"smoke-result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill();p.wait(timeout=15)
        log.flush()
(OUT/"server-exit.json").write_text(json.dumps({"returncode":p.returncode,"supervisor_child_pid":p.pid},indent=2)+"\n")
(OUT/"gpu-after.csv").write_text(gpu()["stdout"])
text=(OUT/"server.log").read_text(errors="replace")
dispatch=MARKER in text
evidence={"explicit_cuda_dispatch_marker_present":dispatch,"marker":MARKER,"server_loaded_packed_head":"EAGLE3 using packed W1A1 draft head" in text,"cuda_backend_log":"ggml_cuda_init" in text,"server_exit_code":p.returncode}
(OUT/"dispatch-evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
print(json.dumps(evidence,indent=2))
if not dispatch:raise RuntimeError("server did not emit explicit CUDA packed W1A1 dispatch marker")

