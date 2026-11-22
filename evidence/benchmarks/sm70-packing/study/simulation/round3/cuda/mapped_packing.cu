// Real sm70 hardware harness; compiling this file is NOT hardware execution.
// Inputs are E2M1 nibbles with independent E8M0 scales for each of two GEMMs.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "fragment_mapping.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(x) do { auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)
#define BL(x) do {auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(#x)+": "+std::to_string(int(e)));}while(0)
template<class T> struct Buf {T* p=nullptr;explicit Buf(size_t n){CU(cudaMalloc((void**)&p,n*sizeof(T)));}~Buf(){if(p)cudaFree(p);}Buf(const Buf&)=delete;};

__host__ __device__ int q_from_code(unsigned code) {
  unsigned mag=code&7u;
  int q=mag<2 ? int(mag):int(2+(mag&1))<<(int(mag/2)-1);
  return (code&8u) ? -q:q;
}
__device__ int read_q(const unsigned char* p,size_t index) {
  unsigned byte=p[index>>1];return q_from_code((byte>>((index&1)*4))&15);
}

// codes [stream=2][rows=M+N][K/2]; scales [stream=2][rows][ceil(K/32)].
// rows < M hold row-major A; remaining rows hold columns of column-major B.
__global__ void prepare_half(const unsigned char* codes,const unsigned char* scales,
                            half* dst,int rows,int k,int sb) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<size_t(2)*rows*k) {
    int row=int(i/k),kk=int(i%k);
    float scale=__int_as_float(int(scales[size_t(row)*sb+kk/32])<<23);
    dst[i]=__float2half_rn(float(read_q(codes,i))*0.5f*scale);
  }
}

__global__ void prepare_packed(const unsigned char* codes,half* dst,int m,int rows,int k,int ra,int rb) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<size_t(rows)*k) {
    int r=(int(i/k)<m)?ra:rb;
    int q0=read_q(codes,i),q1=read_q(codes,i+size_t(rows)*k);
    dst[i]=__float2half_rn(float(q0+r*q1));
  }
}

__device__ __forceinline__ void mma4(float (&d)[8],uint2 a,uint2 b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
    : "r"(a.x),"r"(a.y),"r"(b.x),"r"(b.y));
}

template<int Mode,int LowBits>
__device__ __forceinline__ void decode(float z,float& l,float& h) {
  constexpr int r=1<<LowBits;
  if(Mode==0) {
    float high=nearbyintf(z*(1.0f/65536.0f));
    float rem=z-65536.0f*high;
    float mid=nearbyintf(rem*(1.0f/r));
    l=(rem-r*mid)*0.25f;h=high*0.25f;
  } else if(Mode==1) {
    int iz=__float2int_rn(z);
    l=float(((iz+r/2)&(r-1))-r/2)*0.25f;
    h=float((iz+32768)>>16)*0.25f;
  } else {
    int u=__float_as_int(z+12582912.0f);
    int hb=((u+32768)>>16)+(0x4b400000-0x4b40);
    int lb=((u+r/2)&(r-1))+(0x4b400000-r/2);
    h=fmaf(__int_as_float(hb),0.25f,-3145728.0f);
    l=fmaf(__int_as_float(lb),0.25f,-3145728.0f);
  }
}

// One warp computes one 16x16 output pair, using the four m8n8 groups.
// DecodeChunk=4 or 32. Independent MX scales are applied once per 32 terms,
// after adding the decoded subchunks, never to the mixed packed integer.
template<int DecodeChunk,int LowBits,int Mode,bool Scaled,bool VectorStore>
__global__ __launch_bounds__(128) void mapped_kernel(const half* packed,const unsigned char* scales,
                                                   float* out,int m,int n,int k,int sb) {
  int lane=threadIdx.x&31;
  int tile_row=(int(blockIdx.x)*4+(int(threadIdx.x)>>5))*16;
  int tile_col=int(blockIdx.y)*16;
  if(tile_row>=m)return; // Uniform for a whole warp; every active warp is full.
  int rows=m+n;
  int input_a=tile_row+a_row(lane),input_b=tile_col+b_col(lane);
  const half* a=packed+size_t(input_a)*k;
  const half* b=packed+size_t(m+input_b)*k;
  float c0[8]={},c1[8]={};
  for(int block_k=0;block_k<k;block_k+=32) {
    float lblock[8]={},hblock[8]={};
    int stop=min(block_k+32,k);
    for(int start=block_k;start<stop;start+=DecodeChunk) {
      float z[8]={};
      #pragma unroll
      for(int offset=0;offset<DecodeChunk;offset+=4) {
        uint2 av=*reinterpret_cast<const uint2*>(a+start+offset);
        uint2 bv=*reinterpret_cast<const uint2*>(b+start+offset);
        mma4(z,av,bv);
      }
      #pragma unroll
      for(int i=0;i<8;++i) {
        float lo,hi;decode<Mode,LowBits>(z[i],lo,hi);
        lblock[i]+=lo;hblock[i]+=hi;
      }
    }
    if(Scaled) {
      // Output fragment has two distinct rows and four distinct columns per lane.
      // Reuse 12 scale bytes instead of loading 32 copies for 8 output pairs.
      int ae0[2],ae1[2],be0[4],be1[4];
      #pragma unroll
      for(int r=0;r<2;++r) {
        int row=tile_row+c_row(lane,r*2);
        ae0[r]=int(scales[size_t(row)*sb+block_k/32]);
        ae1[r]=int(scales[size_t(rows+row)*sb+block_k/32]);
      }
      #pragma unroll
      for(int c=0;c<4;++c) {
        int i=(c&1)+((c&2)<<1),col=tile_col+c_col(lane,i);
        be0[c]=int(scales[size_t(m+col)*sb+block_k/32]);
        be1[c]=int(scales[size_t(rows+m+col)*sb+block_k/32]);
      }
      #pragma unroll
      for(int i=0;i<8;++i) {
        int r=(i>>1)&1,c=(i&1)|((i>>1)&2);
        float s0=__int_as_float((ae0[r]+be0[c]-127)<<23);
        float s1=__int_as_float((ae1[r]+be1[c]-127)<<23);
        c0[i]=fmaf(lblock[i],s0,c0[i]);c1[i]=fmaf(hblock[i],s1,c1[i]);
      }
    } else {
      #pragma unroll
      for(int i=0;i<8;++i) {c0[i]+=lblock[i];c1[i]+=hblock[i];}
    }
  }
  if(VectorStore) {
    #pragma unroll
    for(int i=0;i<8;i+=2) {
      size_t index=size_t(tile_row+c_row(lane,i))*n+tile_col+c_col(lane,i);
      *reinterpret_cast<float2*>(out+index)=make_float2(c0[i],c0[i+1]);
      *reinterpret_cast<float2*>(out+size_t(m)*n+index)=make_float2(c1[i],c1[i+1]);
    }
  } else {
    #pragma unroll
    for(int i=0;i<8;++i) {
      size_t index=size_t(tile_row+c_row(lane,i))*n+tile_col+c_col(lane,i);
      out[index]=c0[i];out[size_t(m)*n+index]=c1[i];
    }
  }
}

template<int Chunk,int Bits,bool Scaled>
void launch_mode(int mode,bool vector_store,const half* p,const unsigned char* s,float* o,int m,int n,int k,int sb) {
  dim3 grid((m+63)/64,n/16);constexpr int threads=128;
  if(!vector_store) mapped_kernel<Chunk,Bits,2,Scaled,false><<<grid,threads>>>(p,s,o,m,n,k,sb);
  else if(mode==0) mapped_kernel<Chunk,Bits,0,Scaled,true><<<grid,threads>>>(p,s,o,m,n,k,sb);
  else if(mode==1) mapped_kernel<Chunk,Bits,1,Scaled,true><<<grid,threads>>>(p,s,o,m,n,k,sb);
  else mapped_kernel<Chunk,Bits,2,Scaled,true><<<grid,threads>>>(p,s,o,m,n,k,sb);
  CU(cudaGetLastError());
}
void launch_case(int kind,int mode,bool scaled,bool vec,const half* p,const unsigned char* s,float* o,int m,int n,int k,int sb) {
  if(kind==0) {
    if(scaled)launch_mode<32,8,true>(mode,vec,p,s,o,m,n,k,sb);
    else launch_mode<32,8,false>(mode,vec,p,s,o,m,n,k,sb);
  } else if(kind==1) {
    if(scaled)launch_mode<4,8,true>(mode,vec,p,s,o,m,n,k,sb);
    else launch_mode<4,8,false>(mode,vec,p,s,o,m,n,k,sb);
  } else {
    if(scaled)launch_mode<4,7,true>(mode,vec,p,s,o,m,n,k,sb);
    else launch_mode<4,7,false>(mode,vec,p,s,o,m,n,k,sb);
  }
}

template<class F> float timed(F f,int warmup,int iters) {
  for(int i=0;i<warmup;++i)f();CU(cudaDeviceSynchronize());
  cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));CU(cudaEventRecord(a));
  for(int i=0;i<iters;++i)f();CU(cudaEventRecord(b));CU(cudaEventSynchronize(b));
  float ms;CU(cudaEventElapsedTime(&ms,a,b));CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));return ms/iters;
}

int encode_q(int q) {for(int c=0;c<16;++c)if(q_from_code(c)==q)return c;throw std::runtime_error("Not E2M1.");}
bool supported_q(int kind,bool is_a,int q) {
  if(kind==0)return q>=-1&&q<=1;
  if(kind==1)return q>=-3&&q<=3;
  if(kind==2 && is_a)return q>=-1&&q<=1;
  if(kind==2){for(int code=0;code<16;++code)if(q_from_code(code)==q)return true;}
  return false;
}
const char* names[]={"ternary_k32","seven_k4","ternary_full_k4"};
void run(cublasHandle_t h,int kind,int m,int n,int k,bool variable,bool extreme,bool bench,int warmup,int iters) {
  int chunk=kind==0?32:4,ra=kind==2?512:256,rb=kind==2?128:256;
  if(m%16||n%16||k%chunk)throw std::runtime_error("Invalid shape.");
  int rows=m+n,sb=(k+31)/32;size_t elems=size_t(2)*rows*k;
  std::mt19937 rng(20261122+kind*97+k);
  std::vector<int> qa={-1,0,1},qb=qa;
  if(kind==1)qa=qb={-3,-2,-1,0,1,2,3};
  if(kind==2)qb={-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12};
  std::vector<int> q(elems);std::vector<unsigned char> code(elems/2,0),sc(size_t(2)*rows*sb);
  std::uniform_int_distribution<int> exponent(-2,2);
  for(size_t i=0;i<sc.size();++i)sc[i]=127+(variable?exponent(rng):0);
  for(int s=0;s<2;++s)for(int row=0;row<rows;++row)for(int kk=0;kk<k;++kk) {
    size_t i=(size_t(s)*rows+row)*k+kk;
    const auto& choices=row<m?qa:qb;
    q[i]=extreme ? choices.back() : choices[rng()%choices.size()];
    if(extreme && s==1 && row%2==0)q[i]=-q[i];
    if(!supported_q(kind,row<m,q[i]))throw std::runtime_error("Input outside declared packing domain; no clipping or silent fallback.");
    code[i/2]|=unsigned(encode_q(q[i]))<<((i&1)*4);
    float physical=std::ldexp(float(q[i])*.5f,int(sc[(size_t(s)*rows+row)*sb+kk/32])-127);
    if(!std::isfinite(physical)||__half2float(__float2half_rn(physical))!=physical)
      throw std::runtime_error("Baseline half dequantization not exact.");
  }
  for(int row=0;row<rows;++row)for(int kk=0;kk<k;++kk) {
    size_t i=size_t(row)*k+kk;
    int exact=q[i]+(row<m?ra:rb)*q[i+size_t(rows)*k];
    if(__half2float(__float2half_rn(float(exact)))!=float(exact))
      throw std::runtime_error("Packing would round an input; refusing packed path.");
  }
  std::vector<float> reference(size_t(2)*m*n,0);
  for(int s=0;s<2;++s)for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
    double value=0;
    for(int kk=0;kk<k;++kk) {
      int aa=q[(size_t(s)*rows+r)*k+kk],bb=q[(size_t(s)*rows+m+c)*k+kk];
      int ex=int(sc[(size_t(s)*rows+r)*sb+kk/32])+int(sc[(size_t(s)*rows+m+c)*sb+kk/32])-254;
      value+=std::ldexp(double(aa)*bb*.25,ex);
    }
    reference[(size_t(s)*m+r)*n+c]=float(value);
    if(double(float(value))!=value)throw std::runtime_error("Reference outside chosen exact FP32 test domain.");
  }
  Buf<unsigned char> dc(code.size()),ds(sc.size());Buf<half> dh(elems),dp(size_t(rows)*k);
  Buf<float> base(reference.size()),out(reference.size());
  CU(cudaMemcpy(dc.p,code.data(),code.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(ds.p,sc.data(),sc.size(),cudaMemcpyHostToDevice));
  auto dequant=[&](){prepare_half<<<unsigned((elems+255)/256),256>>>(dc.p,ds.p,dh.p,rows,k,sb);CU(cudaGetLastError());};
  auto pack=[&](){prepare_packed<<<unsigned((size_t(rows)*k+255)/256),256>>>(dc.p,dp.p,m,rows,k,ra,rb);CU(cudaGetLastError());};
  auto blas=[&](){float alpha=1,beta=0;long long stride=static_cast<long long>(rows)*k;
    BL(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
      dh.p+size_t(m)*k,CUDA_R_16F,k,stride,dh.p,CUDA_R_16F,k,stride,&beta,
      base.p,CUDA_R_32F,n,static_cast<long long>(m)*n,2,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));};
  dequant();pack();blas();CU(cudaDeviceSynchronize());
  std::vector<float> got(reference.size());
  CU(cudaMemcpy(got.data(),base.p,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
  size_t base_bad=0;for(size_t i=0;i<got.size();++i)if(got[i]!=reference[i])++base_bad;
  if(base_bad)throw std::runtime_error("cuBLAS reference mismatch: "+std::to_string(base_bad));
  float bms=0,btotal=0,dms=0,pms=0;
  if(bench){bms=timed(blas,warmup,iters);btotal=timed([&](){dequant();blas();},warmup,iters);dms=timed(dequant,warmup,iters);pms=timed(pack,warmup,iters);}
  for(int variant=0;variant<4;++variant) {
    int mode=std::min(variant,2);bool vec=variant!=3;
    auto compute=[&](){launch_case(kind,mode,variable,vec,dp.p,ds.p,out.p,m,n,k,sb);};
    compute();CU(cudaDeviceSynchronize());CU(cudaMemcpy(got.data(),out.p,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
    size_t bad=0;float maxerr=0;
    for(size_t i=0;i<got.size();++i){if(got[i]!=reference[i])++bad;maxerr=std::max(maxerr,std::abs(got[i]-reference[i]));}
    std::cout<<"{\"type\":\""<<(bench?"benchmark":"correctness")<<"\",\"candidate\":\""<<names[kind]
      <<"\",\"M\":"<<m<<",\"N\":"<<n<<",\"K\":"<<k<<",\"decode_chunk\":"<<chunk
      <<",\"decode_mode\":"<<mode<<",\"vector_store\":"<<(vec?"true":"false")
      <<",\"variable_independent_MX_scales\":"<<(variable?"true":"false")<<",\"scale_exponent_min\":"<<(variable?-2:0)<<",\"scale_exponent_max\":"<<(variable?2:0)
      <<",\"extreme\":"<<(extreme?"true":"false")<<",\"output_values\":"<<reference.size()
      <<",\"candidate_mismatches\":"<<bad<<",\"max_abs_error\":"<<maxerr<<",\"baseline_mismatches\":"<<base_bad;
    if(bench && !bad) {
      float cms=timed(compute,warmup,iters),total=timed([&](){pack();compute();},warmup,iters);
      std::cout<<",\"baseline_cublas_ms\":"<<bms<<",\"baseline_dequant_ms\":"<<dms
        <<",\"baseline_end_to_end_ms\":"<<btotal<<",\"pack_ms\":"<<pms<<",\"candidate_compute_ms\":"<<cms
        <<",\"candidate_end_to_end_ms\":"<<total<<",\"compute_speedup\":"<<bms/cms
        <<",\"end_to_end_speedup\":"<<btotal/total<<",\"iterations\":"<<iters;
    }
    std::cout<<"}\n";
    if(bad)throw std::runtime_error("Mapped packing correctness failed; no valid performance claim.");
  }
}

int main(int argc,char**argv) {
 try {
  int device=0,iters=20,warmup=5;bool bench=true,allow=false;
  for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(++i>=argc)throw std::runtime_error("Missing option value");return std::stoi(argv[i]);};
   if(a=="--device")device=val();else if(a=="--iters")iters=val();else if(a=="--warmup")warmup=val();
   else if(a=="--correctness-only")bench=false;else if(a=="--allow-other-gpu")allow=true;
   else if(a=="--help"){std::cout<<"mapped_packing [--device N] [--iters N] [--warmup N] [--correctness-only] [--allow-other-gpu]\n";return 0;}
   else throw std::runtime_error("Unknown option: "+a);}
  if(iters<1||iters>10000||warmup<0||warmup>1000)throw std::runtime_error("Invalid iteration count");
  for(int kind=0;kind<3;++kind)
    if(supported_q(kind,true,12)&&supported_q(kind,false,12))
      throw std::runtime_error("Domain gate incorrectly accepts full E2M1 by full E2M1.");
  CU(cudaSetDevice(device));cudaDeviceProp prop{};CU(cudaGetDeviceProperties(&prop,device));
  bool v100=prop.major==7&&prop.minor==0&&std::string(prop.name).find("V100")!=std::string::npos;
  int dv,rv;CU(cudaDriverGetVersion(&dv));CU(cudaRuntimeGetVersion(&rv));
  std::cout<<std::setprecision(10)<<"{\"type\":\"environment\",\"gpu\":\""<<prop.name
    <<"\",\"actual_V100\":"<<(v100?"true":"false")<<",\"driver_version\":"<<dv<<",\"runtime_version\":"<<rv
    <<",\"SMs\":"<<prop.multiProcessorCount<<",\"global_memory_bytes\":"<<prop.totalGlobalMem<<"}\n";
  if(!v100&&!allow)throw std::runtime_error("Not an actual V100; explicit comparison flag required.");
  if(prop.major<7)throw std::runtime_error("Compute capability below 7.0.");
  cublasHandle_t h;BL(cublasCreate(&h));BL(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
  for(int kind=0;kind<3;++kind)for(bool variable:{false,true})for(bool extreme:{false,true}) {
    run(h,kind,16,16,kind==0?32:4,variable,extreme,false,warmup,iters);
    run(h,kind,32,32,64,variable,extreme,false,warmup,iters);
  }
  if(bench)for(int kind=0;kind<3;++kind)for(bool variable:{false,true})for(int k:{32,256})
    run(h,kind,128,128,k,variable,false,true,warmup,iters);
  BL(cublasDestroy(h));CU(cudaDeviceSynchronize());
  std::cout<<"{\"type\":\"completed\",\"success\":true}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"Hardware harness failed: "<<e.what()<<"\n";return 2;}
}
