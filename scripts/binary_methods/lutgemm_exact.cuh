/* LUT-GEMM
 * Copyright (c) 2024-present NAVER Cloud Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCREEN_LUT_EXACT_HPP
#define SCREEN_LUT_EXACT_HPP

namespace kernel{
inline int div_roundup(int x , int y){return (x + y - 1)/ y;}



template<int K_TILE_SIZE>
__global__ void _nqmv(uint32_t *W, float *alpha, const int8_t *input, float *output, int M, int K, int NUM_BITS, int M_TILE_SIZE, int group_size){

    input += blockIdx.z*K; output += blockIdx.z*M;
    __shared__ float lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
 
    int lut_y = threadIdx.x/lut_x_size;
    int lut_x = threadIdx.x%lut_x_size;

    const int8_t *_inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];
    
    float base =    + float((2 * ((lut_x>>0) & 1) - 1)) * _inp[0]
                     + float((2 * ((lut_x>>1) & 1) - 1)) * _inp[1]
                     + float((2 * ((lut_x>>2) & 1) - 1)) * _inp[2]
                     + float((2 * ((lut_x>>3) & 1) - 1)) * _inp[3]
                     + float((2 * ((lut_x>>4) & 1) - 1)) * _inp[4]
                     + float((2 * ((lut_x>>5) & 1) - 1)) * _inp[5]
                     + float((2 * ((lut_x>>6) & 1) - 1)) * _inp[6]
                     + float((2 * ((lut_x>>7) & 1) - 1)) * _inp[7] ;
             
    lut[lut_y][lut_x] = base;

    int s = (lut_x_size==1)  ?0:
            (lut_x_size==2)  ?1:
            (lut_x_size==4)  ?2:
            (lut_x_size==8)  ?3:
            (lut_x_size==16) ?4:
            (lut_x_size==32) ?5:
            (lut_x_size==64) ?6: 
            (lut_x_size==128)?7: 8;  

    for(;s<8;s++){
        float iValue =  float(2)*_inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] =  lut[lut_y][i +  lut_x - (1 << s)] + iValue;
        }
    }
    __syncthreads();

    int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x*2;
    int m_end = (blockIdx.x + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = blockDim.x * 2;

    uint32_t *bW = &W[blockIdx.y * K_TILE_SIZE/32 * NUM_BITS * M];
    int group_idx = (blockIdx.y * K_TILE_SIZE)/group_size;
    for(int m = m_start;m < m_end;m += m_step){
        float reg_o0 = 0;
        float reg_o1 = 0;
        for(int b=0;b < NUM_BITS;b++){
            float   reg_a0 = 1.0f;
            float   reg_a1 = 1.0f;
            float   reg_t_o0 = 0;
            float   reg_t_o1 = 0;
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 
                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + lut[kt*4 + 3][reg_w3]; 
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
        }
        atomicAdd(&output[m],reg_o0); atomicAdd(&output[m+1],reg_o1);
    }
}

}
#endif

// Screening adapter: repacking weights is deliberately included on every call.
// A production exporter could store the transposed layout once.
__global__ void lut_transpose(const uint32_t*w,uint32_t*t,int m,int words){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m*words)t[(i%words)*m+i/words]=w[i];}
__global__ void lut_scale(float*y,const float*a,const float*b,int m,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m*n)y[i]=(y[i]*a[i%m])*b[i/m];}
