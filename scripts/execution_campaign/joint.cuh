static __global__ void exec_joint_dot(const uint32_t*w0,const float*a0,const float*m0,float*y0,const uint32_t*w1,const float*a1,const float*m1,float*y1,const int8_t*q,const float*b,const float*s,int m,int k){
 int row=blockIdx.x*4+threadIdx.y,t=blockIdx.y,lane=threadIdx.x,acc0=0,acc1=0;
 if(row<m)for(int j=lane*4;j<k;j+=128){int x=*(const int*)(q+t*k+j);acc0=__dp4a(x,int(exec_sign4(w0[row*(k/32)+j/32]>>(j%32))),acc0);acc1=__dp4a(x,int(exec_sign4(w1[row*(k/32)+j/32]>>(j%32))),acc1);}
 for(int d=16;d;d/=2){acc0+=__shfl_down_sync(0xffffffff,acc0,d);acc1+=__shfl_down_sync(0xffffffff,acc1,d);}
 if(lane==0&&row<m){y0[t*m+row]=exec_affine(acc0,a0[row],b[t],m0?m0[row]:0,s[t]);y1[t*m+row]=exec_affine(acc1,a1[row],b[t],m1?m1[row]:0,s[t]);}
}
