from pathlib import Path
import subprocess,hashlib,json
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/binary-methods-20261104-01';C=Path(__file__).resolve().parent
s=(R/'lutgemm.hpp').read_text();s=s[:s.index('template<int k_tile_size>')]+ '\n}\n#endif\n';(R/'lutgemm_kernel.hpp').write_text(s)
q='''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "lutgemm_kernel.hpp"
extern "C" int lut_launch(uint32_t*w,half*a,half*x,half*y,int m,int k,int n,int kt,int mt,void*stream){
 auto s=(cudaStream_t)stream;cudaMemsetAsync(y,0,n*m*sizeof(half),s);
 for(int t=0;t<n;t++){
  dim3 g((m+mt-1)/mt,k/kt);
  if(kt==64)kernel::_nqmv<64><<<g,256,0,s>>>(w,a,x+t*k,y+t*m,m,k,1,mt,k);
  if(kt==128)kernel::_nqmv<128><<<g,256,0,s>>>(w,a,x+t*k,y+t*m,m,k,1,mt,k);
  if(kt==256)kernel::_nqmv<256><<<g,256,0,s>>>(w,a,x+t*k,y+t*m,m,k,1,mt,k);
 }
 return cudaGetLastError();
}
''';(R/'lutgemm.cu').write_text(q)
cmd=['/usr/local/cuda-13.1/bin/nvcc','-O3','-std=c++17','-arch=sm_120','-I'+str(P/'results/cuda-glibc-compat/include'),'-Xcompiler','-fPIC','-shared',str(R/'lutgemm.cu'),'-o',str(R/'lutgemm.so')]
subprocess.run(cmd,check=True);(R/'lutgemm-source.json').write_text(json.dumps(dict(url='https://raw.githubusercontent.com/naver-aics/lut-gemm/master/lutGEMM/src/cuda/kernels/mv_fp16.hpp',sha256=hashlib.sha256((R/'lutgemm.hpp').read_bytes()).hexdigest(),adaptation='Removed unused host wrappers; kernel unchanged; launches64/128/256 K tiles,512/2048 M tiles; transposed binary weights prepared once; output memset included',command=cmd),indent=2))
