import json,pathlib,hashlib,subprocess,datetime,platform
root=pathlib.Path("/home/philip/binary-eagle-decoding"); out=root/"results/sm75-mma-probe-20260924"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source=root/"kernels/sm75_mma_probe.cu"; header=root/"results/cuda-glibc-compat/include/crt/math_functions.h"; binary=out/"sm75_mma_probe"
sass=out/"sass.txt"; ptx=out/"ptx.txt"
lines=[]
for kind,p in (("SASS",sass),("PTX",ptx)):
    lines.append("=== "+kind+" excerpt ===")
    src=p.read_text().splitlines()
    for i,line in enumerate(src):
        if "BMMA.88128.XOR.POPC" in line or "mma.sync.aligned.m8n8k128" in line: lines.extend(src[max(0,i-1):min(len(src),i+2)])
(out/"disassembly-excerpt.txt").write_text("\n".join(lines)+"\n")
nvcc=subprocess.run(["nvcc","--version"],capture_output=True,text=True)
cudump=subprocess.run(["cuobjdump","--version"],capture_output=True,text=True)
gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True)
states={name:json.loads((root/"runs"/name/"state.json").read_text()) for name in ("sm75-mma-compile-20260924","sm75-mma-inspect-20260924")}
files={}
for p in sorted(out.iterdir()):
    if p.is_file() and p.name!="artifact-manifest.json": files[str(p.relative_to(root))]=sha(p)
for name in states:
    for p in sorted((root/"runs"/name).glob("*")):
        if p.is_file(): files[str(p.relative_to(root))]=sha(p)
files[str(header.relative_to(root))]=sha(header)
sys_header=pathlib.Path("/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h")
obj={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"main_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),"probe_source_commit":subprocess.check_output(["git","-C",str(root),"log","-1","--format=%H","--","kernels/sm75_mma_probe.cu"],text=True).strip(),"submodule_commit":subprocess.check_output(["git","-C",str(root/"third_party/llama.cpp"),"rev-parse","HEAD"],text=True).strip(),"host_os":platform.platform(),"probe_source_sha256":sha(source),"private_header_sha256":sha(header),"system_cuda_header_sha256":sha(sys_header),"compile_command":["nvcc","-std=c++17","-O2","-arch=sm_75","-I/home/philip/binary-eagle-decoding/results/cuda-glibc-compat/include","kernels/sm75_mma_probe.cu","-o","results/sm75-mma-probe-20260924/sm75_mma_probe"],"nvcc_version":nvcc.stdout,"cuobjdump_version":cudump.stdout,"compile_supervisor":states["sm75-mma-compile-20260924"],"inspection_supervisor":states["sm75-mma-inspect-20260924"],"binary_sha256":sha(binary),"sass_sha256":sha(sass),"ptx_sha256":sha(ptx),"compile_only":True,"binary_executed":False,"gpu_post_run":gpu.stdout,"files_sha256":files}
(out/"artifact-manifest.json").write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
print("manifest_sha256",sha(out/"artifact-manifest.json"),"files",len(files),"gpu",gpu.stdout.strip())
