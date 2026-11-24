// SM70 structure probes only. Compiled locally; not GPU-timed.
// Prepacked inputs, all shared scales = 1. The output scale 1/4 maps q=2*x.
// Every launch must use exactly the block size documented by the kernel name.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;
using AFrag = wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major>;
using BFrag = wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>;
using CFrag = wmma::fragment<wmma::accumulator,16,16,16,float>;

// Two independent ordinary products, each K=32, exactly the target outputs
// of a packed two-sided call. One warp per CTA; same WMMA implementation.
template<int Warps>
__device__ __forceinline__ void baseline_two_K32_body(const half* a0,const half* a1,
 const half* b0,const half* b1,float* c0,float* c1,int m) {
  int row=(blockIdx.x*Warps+threadIdx.x/32)*16,col=blockIdx.y*16;
  AFrag af; BFrag bf; CFrag lo,hi;
  wmma::fill_fragment(lo,0); wmma::fill_fragment(hi,0);
  #pragma unroll
  for(int kk=0;kk<32;kk+=16) {
    wmma::load_matrix_sync(af,a0+row+kk*m,m);
    wmma::load_matrix_sync(bf,b0+kk+col*32,32);
    wmma::mma_sync(lo,af,bf,lo);
    wmma::load_matrix_sync(af,a1+row+kk*m,m);
    wmma::load_matrix_sync(bf,b1+kk+col*32,32);
    wmma::mma_sync(hi,af,bf,hi);
  }
  #pragma unroll
  for(int i=0;i<lo.num_elements;++i) {lo.x[i]*=0.25f;hi.x[i]*=0.25f;}
  wmma::store_matrix_sync(c0+row+col*m,lo,m,wmma::mem_col_major);
  wmma::store_matrix_sync(c1+row+col*m,hi,m,wmma::mem_col_major);
}
extern "C" __global__ void baseline_two_K32(const half* a0,const half* a1,
 const half* b0,const half* b1,float* c0,float* c1,int m) {
  baseline_two_K32_body<1>(a0,a1,b0,b1,c0,c1,m);
}
extern "C" __global__ void baseline_two_K32_w4(const half* a0,const half* a1,
 const half* b0,const half* b1,float* c0,float* c1,int m) {
  baseline_two_K32_body<4>(a0,a1,b0,b1,c0,c1,m);
}

template<int DecodeMode,int Warps>
__device__ __forceinline__ void two_sided_K32_body(const half* a,const half* b,
 float* c0,float* c1,int m) {
  int row=(blockIdx.x*Warps+threadIdx.x/32)*16,col=blockIdx.y*16;
  AFrag af; BFrag bf; CFrag z,lo,hi;
  wmma::fill_fragment(z,0);
  #pragma unroll
  for(int kk=0;kk<32;kk+=16) {
    wmma::load_matrix_sync(af,a+row+kk*m,m);
    wmma::load_matrix_sync(bf,b+kk+col*32,32);
    wmma::mma_sync(z,af,bf,z);
  }
  #pragma unroll
  for(int i=0;i<z.num_elements;++i) {
    if(DecodeMode==1) {
      int iz=__float2int_rn(z.x[i]);
      int h=(iz+32768)>>16;
      int l=((iz+128)&255)-128;
      hi.x[i]=float(h)*0.25f; lo.x[i]=float(l)*0.25f;
    } else if(DecodeMode==2) {
      // For K=32 ternary |z|<=2,113,568<2^22. Addition stays in the
      // [2^23,2^24) binade where float bits encode consecutive integers.
      // No general float-to-int/int-to-float instructions are needed.
      int u=__float_as_int(z.x[i]+12582912.0f);
      int hb=((u+32768)>>16)+(0x4b400000-0x4b40);
      int lb=((u+128)&255)+(0x4b400000-128);
      hi.x[i]=fmaf(__int_as_float(hb),0.25f,-3145728.0f);
      lo.x[i]=fmaf(__int_as_float(lb),0.25f,-3145728.0f);
    } else {
      float h=nearbyintf(z.x[i]*(1.0f/65536.0f));
      float rem=z.x[i]-65536.0f*h;
      float mid=nearbyintf(rem*(1.0f/256.0f));
      float l=rem-256.0f*mid;
      hi.x[i]=h*0.25f;lo.x[i]=l*0.25f;
    }
  }
  wmma::store_matrix_sync(c0+row+col*m,lo,m,wmma::mem_col_major);
  wmma::store_matrix_sync(c1+row+col*m,hi,m,wmma::mem_col_major);
}

extern "C" __global__ void two_sided_float_K32_w1(const half* a,const half* b,
 float* c0,float* c1,int m) {
  two_sided_K32_body<0,1>(a,b,c0,c1,m);
}
extern "C" __global__ void two_sided_float_K32_w4(const half* a,const half* b,
 float* c0,float* c1,int m) {
  two_sided_K32_body<0,4>(a,b,c0,c1,m);
}
extern "C" __global__ void two_sided_int_K32_w1(const half* a,const half* b,
 float* c0,float* c1,int m) {
  two_sided_K32_body<1,1>(a,b,c0,c1,m);
}
extern "C" __global__ void two_sided_int_K32_w4(const half* a,const half* b,
 float* c0,float* c1,int m) {
  two_sided_K32_body<1,4>(a,b,c0,c1,m);
}
extern "C" __global__ void two_sided_magic_K32_w4(const half* a,const half* b,
 float* c0,float* c1,int m) {
  two_sided_K32_body<2,4>(a,b,c0,c1,m);
}

template<bool IntegerDecode>
__device__ __forceinline__ void shared_B_K16_body(const half* a,const half* b,
 float* c0,float* c1,int m) {
  int row=blockIdx.x*16,col=blockIdx.y*16;
  AFrag af; BFrag bf; CFrag z,lo,hi;
  wmma::fill_fragment(z,0);
  wmma::load_matrix_sync(af,a+row,m);
  wmma::load_matrix_sync(bf,b+col*16,16);
  wmma::mma_sync(z,af,bf,z);
  #pragma unroll
  for(int i=0;i<z.num_elements;++i) {
    if(IntegerDecode) {
      int iz=__float2int_rn(z.x[i]);
      int h=(iz+256)>>9;
      int l=((iz+256)&511)-256;
      hi.x[i]=float(h)*0.25f;lo.x[i]=float(l)*0.25f;
    } else {
      float h=nearbyintf(z.x[i]*(1.0f/512.0f));
      hi.x[i]=h*0.25f;lo.x[i]=(z.x[i]-512.0f*h)*0.25f;
    }
  }
  wmma::store_matrix_sync(c0+row+col*m,lo,m,wmma::mem_col_major);
  wmma::store_matrix_sync(c1+row+col*m,hi,m,wmma::mem_col_major);
}
extern "C" __global__ void shared_B_float_K16(const half* a,const half* b,
 float* c0,float* c1,int m) { shared_B_K16_body<false>(a,b,c0,c1,m); }
extern "C" __global__ void shared_B_int_K16(const half* a,const half* b,
 float* c0,float* c1,int m) { shared_B_K16_body<true>(a,b,c0,c1,m); }
