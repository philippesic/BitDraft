import json,subprocess,hashlib,time
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');S=R/'runs/checkouts/native-seven-5eb221-20261006';B=R/'runs/build/native-seven-5eb221-sm120-20261006';O=R/'results/dspark-launch-20261006/native-seven-build-provenance.json'
assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()=='5eb221ec53cb4317aef963e3bdec3cc69401eca4'
assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
assert not B.exists() and not O.exists()
C=R/'.venv/bin/cmake'
configure=[str(C),'-S',str(S),'-B',str(B),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_MAKE_PROGRAM='+str(R/'.venv/bin/ninja'),'-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc','-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++','-DCMAKE_C_COMPILER=/usr/bin/gcc','-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_CUDA_FLAGS=-I'+str(R/'results/cuda-glibc-compat/include'),'-DLLAMA_BUILD_TESTS=ON','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON','-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_APP=OFF','-DLLAMA_BUILD_UI=OFF','-DLLAMA_OPENSSL=OFF','-DGGML_METAL=OFF','-DGGML_CUDA_FORCE_CUBLAS=OFF','-DGGML_CUDA_FORCE_MMQ=OFF','-DGGML_CUDA_FA=ON','-DGGML_CUDA_GRAPHS=ON']
build=[str(C),'--build',str(B),'--parallel','2','--target','llama-server','test-block-binary','test-arg-parser','test-backend-ops']
subprocess.run(configure,check=True,timeout=180);subprocess.run(build,check=True,timeout=2700)
def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
subprocess.run([str(B/'bin/test-arg-parser')],check=True,timeout=120)
v={'schema':'dspark_seven_reserve_sm120_build_v1','status':'PASS','native_commit':'5eb221ec53cb4317aef963e3bdec3cc69401eca4','capture_native_commit_preserved':'ecff6d4e74814c631801df2e74f562d4ed6bd0eb','configure':configure,'build':build,'cuda_architecture':120,'compile_jobs':2,'memory_max_bytes':12884901888,'files':[pin(p) for p in [B/'CMakeCache.txt',B/'compile_commands.json',*sorted((B/'bin').glob('*.so*')),B/'bin/llama-server',B/'bin/test-block-binary',B/'bin/test-arg-parser',B/'bin/test-backend-ops']],'cpu_arg_parser':'PASS','native_cuda_runtime':'PENDING_root_owned_execution'}
O.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'status':'PASS','provenance':pin(O)}))
