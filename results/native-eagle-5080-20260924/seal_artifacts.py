import json,hashlib,pathlib
R=pathlib.Path.cwd();D=R/"results/native-eagle-5080-20260924";S=R/"runs/native-eagle-5080-20260924"
m=json.load(open(D/"manifest.json"));q=json.load(open(D/"report.json"));rows=json.load(open(D/"records.json"));state=json.load(open(S/"state.json"))
assert q["status"]=="complete" and len(rows)==180 and state["status"]=="finished" and state["exit_code"]==0
checks=[]
for rep in range(5):
    for variant in ("target_only","ordinary_eagle","packed_head_w1a1"):
        if variant=="packed_head_w1a1":
            p=D/f"rep-{rep:02d}"/variant/"dispatch-evidence.json";d=json.load(open(p));checks.append((rep,d["packed_head_loader_log"],d["explicit_cuda_w1a1_op_log"],d["cuda_w1a1_dispatch_confirmed"]))
assert len(checks)==5 and all(x[1:]==(True,True,True) for x in checks)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
artifacts={}
for p in sorted(D.rglob("*")):
    if p.is_file() and p.name!="artifact-seal.json" and p.name!="seal_artifacts.py":
        artifacts[str(p.relative_to(R))]={"bytes":p.stat().st_size,"sha256":sha(p)}
for p in (S/"state.json",S/"stdout.log",R/"results/native-eagle-5080-dryrun-20260924/manifest.json"):
    artifacts[str(p.relative_to(R))]={"bytes":p.stat().st_size,"sha256":sha(p)}
seal={"run_id":"native-eagle-5080-20260924","project_commit":m["project_commit"],"llama_gitlink":m["llama_gitlink"],"supervisor_state":{"status":state["status"],"exit_code":state["exit_code"],"started":state["started_at_utc"],"ended":state["ended_at_utc"]},"record_count":len(rows),"report":q,"packed_dispatch_confirmed_repetitions":[x[0] for x in checks],"final_gpu_cleanup":{"windows_samples":"results/native-eagle-5080-20260924/final-windows-gpu.txt","baseline":"3,050 MiB used / 12,928 MiB free; 0-1% across 3 samples","wsl_project_processes":[],"llama_servers":[]},"native_cuda_validation_manifest_sha256":sha("results/native-ggml-cuda-validation-20260924/run-manifest.json"),"dry_run_manifest_sha256":sha("results/native-eagle-5080-dryrun-20260924/manifest.json"),"artifacts":artifacts}
out=D/"artifact-seal.json";out.write_text(json.dumps(seal,indent=2,sort_keys=True)+"\n")
print(json.dumps({"seal_path":str(out),"seal_sha256":sha(out),"record_count":len(rows),"aggregate":q["aggregation"],"speedups":q["packed_speedup_vs"],"text_match":q["greedy_text_match_vs_target_only"],"draft_timing":q["draft_timing"],"all_packed_dispatch_confirmed":True,"raw_file_count":len(artifacts)},indent=2))

