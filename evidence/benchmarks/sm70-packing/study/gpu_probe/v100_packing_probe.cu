// Experimental radix packing probes, not a production MXFP4 implementation.
// Compile with CUDA 12.x. See README.md for scopes and limitations.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
  throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)
#define BLAS_OK(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) \
  throw std::runtime_error(std::string(#x) + ": cublas status " + std::to_string(int(e_))); } while (0)

template<class T> struct Device {
  T* p = nullptr;
  explicit Device(size_t n) { CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
  ~Device() { if (p) cudaFree(p); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

std::string quote(const std::string& s) {
  std::ostringstream o; o << '"';
  for (char c : s) {
    if (c == '"' || c == '\\') o << '\\' << c;
    else if (c == '\n') o << "\\n";
    else o << c;
  }
  o << '"'; return o.str();
}

// Column-major [2*M,K] input; the two logical A matrices occupy row blocks.
__global__ void pack_rows(const half* a, half* packed, int m, int k, float radix) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size_t(m) * k) {
    int row = int(idx % m), col = int(idx / m);
    float a0 = __half2float(a[row + size_t(col) * (2*m)]);
    float a1 = __half2float(a[row + m + size_t(col) * (2*m)]);
    packed[idx] = __float2half_rn(a0 + radix * a1);
  }
}

__global__ void unpack_rows(const float* packed, float* out, int m, int n,
                            float radix, float physical_scale) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size_t(m) * n) {
    float z = packed[idx];
    float hi = nearbyintf(z / radix);
    float lo = z - radix * hi;
    int row = int(idx % m), col = int(idx / m);
    out[row + size_t(col) * (2*m)] = physical_scale * lo;
    out[row + m + size_t(col) * (2*m)] = physical_scale * hi;
  }
}

// One warp computes one 16x16 output tile. This intentionally simple kernel
// is for arithmetic probes. It is NOT an optimized GEMM performance baseline.
__global__ void wmma_tile(const half* a, const half* b, float* c, int k) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
  wmma::fragment<wmma::accumulator,16,16,16,float> cf;
  wmma::fill_fragment(cf,0.0f);
  for (int kk = 0; kk < k; kk += 16) {
    wmma::load_matrix_sync(af,a + size_t(kk)*16,16);
    wmma::load_matrix_sync(bf,b + kk,k);
    wmma::mma_sync(cf,af,bf,cf);
  }
  wmma::store_matrix_sync(c,cf,16,wmma::mem_col_major);
}

// Decode every 16 products before adding the two logical results. This avoids
// radix carry in the narrow alphabet even for long K, at extra instruction cost.
__global__ void wmma_decode_chunks(const half* a, const half* b, float* out,
                                    int m, int n, int k, float radix,
                                    float physical_scale) {
  using namespace nvcuda;
  const int row = int(blockIdx.x)*16, col = int(blockIdx.y)*16;
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
  wmma::fragment<wmma::accumulator,16,16,16,float> cf,lo,hi;
  wmma::fill_fragment(lo,0.0f); wmma::fill_fragment(hi,0.0f);
  for (int kk = 0; kk < k; kk += 16) {
    wmma::load_matrix_sync(af,a + row + size_t(kk)*m,m);
    wmma::load_matrix_sync(bf,b + kk + size_t(col)*k,k);
    wmma::fill_fragment(cf,0.0f);
    wmma::mma_sync(cf,af,bf,cf);
    for (int i = 0; i < cf.num_elements; ++i) {
      float h = nearbyintf(cf.x[i]/radix);
      lo.x[i] += physical_scale * (cf.x[i]-radix*h);
      hi.x[i] += physical_scale * h;
    }
  }
  wmma::store_matrix_sync(out + row + size_t(col)*(2*m),lo,2*m,wmma::mem_col_major);
  wmma::store_matrix_sync(out + row + m + size_t(col)*(2*m),hi,2*m,wmma::mem_col_major);
}

template<class F> float time_ms(F&& f, int warmup, int iters) {
  for (int i=0; i<warmup; ++i) f();
  CUDA_OK(cudaDeviceSynchronize());
  cudaEvent_t start,stop;
  CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  CUDA_OK(cudaEventRecord(start));
  for (int i=0; i<iters; ++i) f();
  CUDA_OK(cudaEventRecord(stop)); CUDA_OK(cudaEventSynchronize(stop));
  float total; CUDA_OK(cudaEventElapsedTime(&total,start,stop));
  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  return total/iters;
}

const std::vector<int> TERNARY = {-1,0,1};
const std::vector<int> NARROW = {-3,-2,-1,0,1,2,3};
const std::vector<int> FULL = {-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12};

long long encoding_errors(const std::vector<int>& alphabet, int radix) {
  long long errors = 0;
  for (int a0:alphabet) for(int a1:alphabet) {
    int exact = a0 + radix*a1;
    if (__half2float(__float2half_rn(float(exact))) != float(exact)) ++errors;
  }
  return errors;
}

// Probe the actual Volta FP16-input/FP32-accumulator path, not CPU emulation.
void correctness_case(const std::string& label, const std::vector<int>& alphabet,
                      int shift, int k, bool adversarial, unsigned seed) {
  constexpr int m=16,n=16;
  int radix = 1 << shift;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick(0,int(alphabet.size())-1);
  std::vector<int> q0(m*k),q1(m*k),qb(k*n);
  int maxq = alphabet.back();
  for (int col=0; col<k; ++col) for (int row=0; row<m; ++row) {
    size_t i=row+size_t(col)*m;
    // Include positive and negative low-digit carry and cancellation cases.
    q0[i]=adversarial ? (row%2 ? -maxq:maxq) : alphabet[pick(rng)];
    q1[i]=adversarial ? (row%4<2 ? maxq:-maxq) : alphabet[pick(rng)];
  }
  for (int col=0; col<n; ++col) for (int row=0; row<k; ++row)
    qb[row+size_t(col)*k]=adversarial ? (col%2 ? -maxq:maxq) : alphabet[pick(rng)];
  std::vector<half> ha(2*m*k),hb(k*n);
  for (int col=0;col<k;++col) for(int row=0;row<m;++row) {
    ha[row+size_t(col)*(2*m)] = __float2half_rn(float(q0[row+size_t(col)*m]));
    ha[row+m+size_t(col)*(2*m)] = __float2half_rn(float(q1[row+size_t(col)*m]));
  }
  for (size_t i=0;i<hb.size();++i) hb[i]=__float2half_rn(float(qb[i]));
  Device<half> da(ha.size()),db(hb.size()),dp(m*k);
  Device<float> dc(m*n),ddecoded(2*m*n);
  CUDA_OK(cudaMemcpy(da.p,ha.data(),ha.size()*sizeof(half),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db.p,hb.data(),hb.size()*sizeof(half),cudaMemcpyHostToDevice));
  pack_rows<<<(m*k+255)/256,256>>>(da.p,dp.p,m,k,float(radix));
  CUDA_OK(cudaGetLastError());
  wmma_tile<<<1,32>>>(dp.p,db.p,dc.p,k);
  CUDA_OK(cudaGetLastError());
  wmma_decode_chunks<<<dim3(1,1),32>>>(dp.p,db.p,ddecoded.p,m,n,k,float(radix),1.0f);
  CUDA_OK(cudaGetLastError());
  std::vector<half> packed(m*k);
  std::vector<float> z(m*n),chunked(2*m*n);
  CUDA_OK(cudaMemcpy(z.data(),dc.p,z.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(chunked.data(),ddecoded.p,chunked.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(packed.data(),dp.p,packed.size()*sizeof(half),cudaMemcpyDeviceToHost));
  long long input_rounding=0,accum_mismatches=0,raw_ideal_mismatches=0;
  long long direct_pairs_bad=0,chunk_pairs_bad=0;
  double max_direct_abs=0,max_chunk_abs=0;
  for(size_t i=0;i<packed.size();++i)
    if (__half2float(packed[i]) != float(q0[i]+radix*q1[i])) ++input_rounding;
  for (int col=0;col<n;++col) for(int row=0;row<m;++row) {
    int64_t c0=0,c1=0,encoded_exact=0;
    for(int kk=0;kk<k;++kk) {
      c0 += int64_t(q0[row+size_t(kk)*m])*qb[kk+size_t(col)*k];
      c1 += int64_t(q1[row+size_t(kk)*m])*qb[kk+size_t(col)*k];
      encoded_exact += int64_t(__half2float(packed[row+size_t(kk)*m]))*qb[kk+size_t(col)*k];
    }
    float zz=z[row+size_t(col)*m];
    if(double(zz)!=double(encoded_exact)) ++accum_mismatches;
    if(double(zz)!=double(c0+int64_t(radix)*c1)) ++raw_ideal_mismatches;
    double hi=std::nearbyint(double(zz)/radix),lo=double(zz)-radix*hi;
    if(lo!=double(c0) || hi!=double(c1)) ++direct_pairs_bad;
    max_direct_abs=std::max({max_direct_abs,std::abs(lo-double(c0)),std::abs(hi-double(c1))});
    double clo=chunked[row+size_t(col)*(2*m)],chi=chunked[row+m+size_t(col)*(2*m)];
    if(clo!=double(c0) || chi!=double(c1)) ++chunk_pairs_bad;
    max_chunk_abs=std::max({max_chunk_abs,std::abs(clo-double(c0)),std::abs(chi-double(c1))});
  }
  std::cout << "{\"type\":\"correctness\",\"alphabet\":" << quote(label)
    << ",\"shift\":"<<shift<<",\"radix\":"<<radix<<",\"K\":"<<k
    << ",\"distribution\":"<<quote(adversarial ? "all_extrema":"uniform_alphabet")
    << ",\"seed\":"<<seed<<",\"output_pairs\":256"
    << ",\"all_alphabet_pair_encoding_errors\":"<<encoding_errors(alphabet,radix)
    << ",\"sample_input_rounding\":"<<input_rounding
    << ",\"wmma_accum_vs_exact_encoded_dot_errors\":"<<accum_mismatches
    << ",\"wmma_vs_ideal_packed_dot_errors\":"<<raw_ideal_mismatches
    << ",\"direct_decode_bad_pairs\":"<<direct_pairs_bad
    << ",\"direct_decode_max_abs_qproduct_error\":"<<max_direct_abs
    << ",\"chunk16_decode_bad_pairs\":"<<chunk_pairs_bad
    << ",\"chunk16_decode_max_abs_qproduct_error\":"<<max_chunk_abs<<"}\n";
  // Deliberately failing scan cases are informative; supported positive
  // controls must instead fail the process if hardware violates the bound.
  const bool narrow_positive = alphabet == NARROW && shift == 9;
  const bool ternary_positive = alphabet == TERNARY && shift == 8;
  if (narrow_positive || ternary_positive) {
    if (input_rounding != 0 || chunk_pairs_bad != 0)
      throw std::runtime_error("Supported shared-RHS chunk16 positive control failed.");
    const bool direct_supported = (narrow_positive && k == 16)
                                || (ternary_positive && k <= 64);
    if (direct_supported && (direct_pairs_bad != 0 || accum_mismatches != 0
                            || raw_ideal_mismatches != 0))
      throw std::runtime_error("Supported shared-RHS direct-decoding positive control failed.");
  }
}

void two_sided_probe(const std::vector<int>& alphabet, const std::string& name, int shift) {
  // Each WMMA block has K=16 but only k=0 is nonzero. Thus each output is
  // one FP16 multiply through WMMA. Both operands pack two logical values.
  int r=1<<shift;
  const int pairs=int(alphabet.size()*alphabet.size());
  long long bad=0,total=0,rounding=0;
  Device<half> da(256),db(256); Device<float> dc(256);
  for(int base=0;base<pairs;base+=16) {
    std::vector<half> a(256,__float2half_rn(0.0f));
    for(int i=0;i<16 && base+i<pairs;++i) {
      int j=base+i,a0=alphabet[j/alphabet.size()],a1=alphabet[j%alphabet.size()];
      a[i]=__float2half_rn(float(a0+r*a1));
      if(__half2float(a[i])!=float(a0+r*a1)) ++rounding;
    }
    CUDA_OK(cudaMemcpy(da.p,a.data(),a.size()*sizeof(half),cudaMemcpyHostToDevice));
    for(int bbase=0;bbase<pairs;bbase+=16) {
      std::vector<half> b(256,__float2half_rn(0.0f));
      for(int j=0;j<16 && bbase+j<pairs;++j) {
        int p=bbase+j,b0=alphabet[p/alphabet.size()],b1=alphabet[p%alphabet.size()];
        b[j*16]=__float2half_rn(float(b0+r*b1));
      }
      CUDA_OK(cudaMemcpy(db.p,b.data(),b.size()*sizeof(half),cudaMemcpyHostToDevice));
      wmma_tile<<<1,32>>>(da.p,db.p,dc.p,16); CUDA_OK(cudaGetLastError());
      std::vector<float> c(256);
      CUDA_OK(cudaMemcpy(c.data(),dc.p,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int j=0;j<16 && bbase+j<pairs;++j) for(int i=0;i<16 && base+i<pairs;++i) {
        int ap=base+i,bp=bbase+j;
        int a0=alphabet[ap/alphabet.size()],a1=alphabet[ap%alphabet.size()];
        int b0=alphabet[bp/alphabet.size()],b1=alphabet[bp%alphabet.size()];
        double zz=c[i+j*16];
        // Balanced-radix extraction including the middle cross term.
        double h=std::nearbyint(zz/(double(r)*r));
        double rem=zz-double(r)*r*h;
        double cross=std::nearbyint(rem/r);
        double l=rem-r*cross;
        if(h!=a1*b1 || l!=a0*b0) ++bad;
        ++total;
      }
    }
  }
  std::cout<<"{\"type\":\"two_sided_single_product_exhaustive\",\"alphabet\":"<<quote(name)
    <<",\"shift\":"<<shift<<",\"radix\":"<<r<<",\"cases\":"<<total
    <<",\"operand_pair_rounding_cases\":"<<rounding<<",\"bad_product_pairs\":"<<bad
    <<",\"scope\":\"K=1; balanced-radix decoder only; not a universal impossibility test\"}\n";
}

void two_sided_ternary(int k, int r, bool adversarial) {
  // A constructive example: both operands encode two ternary lanes, while
  // the FP32 accumulator holds low product, cross term, and high product.
  std::mt19937 rng(20261122); std::uniform_int_distribution<int> pick(-1,1);
  std::vector<int> a0(16*k),a1(16*k),b0(k*16),b1(k*16);
  std::vector<half> a(16*k),b(k*16);
  for(int kk=0;kk<k;++kk) for(int row=0;row<16;++row) {
    int i=row+kk*16;
    a0[i]=adversarial ? (row%2 ? -1:1):pick(rng);
    a1[i]=adversarial ? (row%4<2 ? 1:-1):pick(rng);
    a[i]=__float2half_rn(float(a0[i]+r*a1[i]));
  }
  for(int col=0;col<16;++col) for(int kk=0;kk<k;++kk) {
    int i=kk+col*k;
    b0[i]=adversarial ? (col%2 ? -1:1):pick(rng);
    b1[i]=adversarial ? (col%4<2 ? -1:1):pick(rng);
    b[i]=__float2half_rn(float(b0[i]+r*b1[i]));
  }
  Device<half> da(a.size()),db(b.size()); Device<float> dc(256);
  CUDA_OK(cudaMemcpy(da.p,a.data(),a.size()*sizeof(half),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db.p,b.data(),b.size()*sizeof(half),cudaMemcpyHostToDevice));
  wmma_tile<<<1,32>>>(da.p,db.p,dc.p,k); CUDA_OK(cudaGetLastError());
  std::vector<float> c(256); CUDA_OK(cudaMemcpy(c.data(),dc.p,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
  int bad=0,accum_bad=0,cross_bad=0;
  for(int col=0;col<16;++col) for(int row=0;row<16;++row) {
    int lo=0,hi=0,cross=0;
    for(int kk=0;kk<k;++kk) {
      int ia=row+kk*16,ib=kk+col*k;
      lo+=a0[ia]*b0[ib]; hi+=a1[ia]*b1[ib];
      cross+=a0[ia]*b1[ib]+a1[ia]*b0[ib];
    }
    double z=c[row+col*16];
    double h=std::nearbyint(z/(r*r)),rem=z-h*r*r;
    double mid=std::nearbyint(rem/r),l=rem-mid*r;
    if(h!=hi || l!=lo) ++bad;
    if(mid!=cross) ++cross_bad;
    if(z!=lo+r*cross+r*r*hi) ++accum_bad;
  }
  std::cout<<"{\"type\":\"two_sided_ternary\",\"K\":"<<k<<",\"radix\":"<<r
    <<",\"distribution\":"<<quote(adversarial ? "all_extrema":"uniform_alphabet")
    <<",\"output_pairs\":256,\"bad_product_pairs\":"<<bad
    <<",\"cross_term_errors\":"<<cross_bad<<",\"raw_accum_errors\":"<<accum_bad<<"}\n";
  if(bad || cross_bad || accum_bad) throw std::runtime_error("The constructive two-sided ternary example failed on this hardware.");
}

void full_e2m1_fixed_radix_collision() {
  // R=128 exactly encodes the full E2M1 integer alphabet, but K>=2 has a
  // decoding ambiguity even with perfect arithmetic. Same B for both rows:
  // row0: A0=[8,0],  A1=[0,0], B=[8,1] -> (64,0), packed dot 64.
  // row1: A0=[-8,0], A1=[0,1], B=[8,1] -> (-64,1), packed dot 64.
  // Zero padding makes it an actual 16-wide WMMA operation.
  std::vector<half> a(256,__float2half_rn(0.0f)),b(256,__float2half_rn(0.0f));
  a[0]=__float2half_rn(8.0f); a[1]=__float2half_rn(-8.0f);
  a[1+16]=__float2half_rn(128.0f);
  for(int col=0;col<16;++col) { b[col*16]=__float2half_rn(8.0f); b[1+col*16]=__float2half_rn(1.0f); }
  Device<half> da(256),db(256); Device<float> dc(256);
  CUDA_OK(cudaMemcpy(da.p,a.data(),a.size()*sizeof(half),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db.p,b.data(),b.size()*sizeof(half),cudaMemcpyHostToDevice));
  wmma_tile<<<1,32>>>(da.p,db.p,dc.p,16); CUDA_OK(cudaGetLastError());
  std::vector<float> c(256); CUDA_OK(cudaMemcpy(c.data(),dc.p,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
  bool observed=c[0]==64.0f && c[1]==64.0f;
  std::cout<<"{\"type\":\"full_E2M1_fixed_radix_collision\",\"radix\":128"
    <<",\"logical_K\":2,\"wmma_K_with_zero_padding\":16"
    <<",\"packed_output_row0\":"<<c[0]<<",\"packed_output_row1\":"<<c[1]
    <<",\"desired_qproduct_pairs\":[[64,0],[-64,1]]"
    <<",\"collision_observed\":"<<(observed ? "true":"false")
    <<",\"scope\":\"Rules out recovery from z alone for this fixed radix; does not rule out other encodings or extra side information.\"}\n";
  if(!observed) throw std::runtime_error("Expected exact arithmetic collision was not reproduced.");
}

void gemm(cublasHandle_t handle,int m,int n,int k,const half* a,const half* b,
          float* c,float alpha) {
  const float beta=0;
  BLAS_OK(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,
    a,CUDA_R_16F,m,b,CUDA_R_16F,k,&beta,c,CUDA_R_32F,m,
    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void benchmark(cublasHandle_t handle,const std::string& name,const std::vector<int>& alphabet,
               int m,int n,int k,int shift,bool chunked,int warmup,int iters) {
  int radix=1<<shift;
  constexpr float physical_scale=0.25f; // q=2*x for E2M1 values.
  std::mt19937 rng(1729);
  std::uniform_int_distribution<int> pick(0,int(alphabet.size())-1);
  std::vector<half> a(size_t(2*m)*k),b(size_t(k)*n);
  for(half& x:a) x=__float2half_rn(float(alphabet[pick(rng)]));
  for(half& x:b) x=__float2half_rn(float(alphabet[pick(rng)]));
  Device<half> da(a.size()),db(b.size()),dp(size_t(m)*k);
  Device<float> dc(size_t(2*m)*n),dz(size_t(m)*n),dout(size_t(2*m)*n);
  CUDA_OK(cudaMemcpy(da.p,a.data(),a.size()*sizeof(half),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db.p,b.data(),b.size()*sizeof(half),cudaMemcpyHostToDevice));
  auto pack=[&]() {
    pack_rows<<<unsigned((size_t(m)*k+255)/256),256>>>(da.p,dp.p,m,k,float(radix));
    CUDA_OK(cudaGetLastError());
  };
  auto baseline=[&]() { gemm(handle,2*m,n,k,da.p,db.p,dc.p,physical_scale); };
  auto compute=[&]() {
    if(chunked) {
      wmma_decode_chunks<<<dim3(m/16,n/16),32>>>(dp.p,db.p,dout.p,m,n,k,float(radix),physical_scale);
      CUDA_OK(cudaGetLastError());
    } else gemm(handle,m,n,k,dp.p,db.p,dz.p,1.0f);
  };
  auto decode=[&]() {
    if(!chunked) {
      unpack_rows<<<unsigned((size_t(m)*n+255)/256),256>>>(dz.p,dout.p,m,n,float(radix),physical_scale);
      CUDA_OK(cudaGetLastError());
    }
  };
  auto end_to_end=[&]() { pack(); compute(); decode(); };
  baseline(); end_to_end(); CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> ref(size_t(2*m)*n),got(ref.size());
  CUDA_OK(cudaMemcpy(ref.data(),dc.p,ref.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(got.data(),dout.p,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
  size_t bad=0; double maxabs=0;
  for(size_t i=0;i<ref.size();++i) {
    if(got[i]!=ref[i]) ++bad;
    maxabs=std::max(maxabs,std::abs(double(got[i])-ref[i]));
  }
  // Check selected cuBLAS outputs against integer reference as an independent
  // guard; full elementwise equality above compares candidate to cuBLAS.
  size_t baseline_samples_bad=0;
  for(int t=0;t<32;++t) {
    int row=(t*131)%(2*m),col=(t*67)%n;
    int64_t exact=0;
    for(int kk=0;kk<k;++kk)
      exact += int64_t(__half2float(a[row+size_t(kk)*(2*m)]))*int64_t(__half2float(b[kk+size_t(col)*k]));
    if(ref[row+size_t(col)*(2*m)] != float(double(exact)*physical_scale)) ++baseline_samples_bad;
  }
  float base_ms=time_ms(baseline,warmup,iters);
  float packed_compute_ms=time_ms(compute,warmup,iters);
  float pack_ms=time_ms(pack,warmup,iters);
  float decode_ms=chunked ? 0.0f:time_ms(decode,warmup,iters);
  float total_ms=time_ms(end_to_end,warmup,iters);
  // Same two MxN outputs in both paths, hence 4*M*N*K useful operations.
  double useful_flops=4.0*m*n*k;
  std::cout<<"{\"type\":\"benchmark\",\"alphabet\":"<<quote(name)
    <<",\"path\":"<<quote(chunked ? "pack_plus_naive_chunk16_wmma_decode":"pack_plus_cublas_plus_decode")
    <<",\"M_per_output\":"<<m<<",\"N\":"<<n<<",\"K\":"<<k<<",\"radix\":"<<radix
    <<",\"warmup\":"<<warmup<<",\"iterations\":"<<iters<<",\"seed\":1729"
    <<",\"baseline\":\"one_stacked_2M_by_K_cublas_gemm\""
    <<",\"baseline_ms\":"<<base_ms<<",\"packed_compute_ms\":"<<packed_compute_ms
    <<",\"pack_ms\":"<<pack_ms<<",\"decode_ms\":"<<decode_ms
    <<",\"candidate_end_to_end_ms\":"<<total_ms
    <<",\"speedup_compute_only\":"<<base_ms/packed_compute_ms
    <<",\"speedup_end_to_end\":"<<base_ms/total_ms
    <<",\"baseline_useful_TFLOPs\":"<<useful_flops/(base_ms*1e9)
    <<",\"candidate_useful_TFLOPs\":"<<useful_flops/(total_ms*1e9)
    <<",\"output_values\":"<<ref.size()<<",\"candidate_mismatches\":"<<bad
    <<",\"max_abs_error\":"<<maxabs<<",\"baseline_exact_samples\":32"
    <<",\"baseline_exact_sample_mismatches\":"<<baseline_samples_bad
    <<",\"shared_scales\":\"all one; no quantization or optimizer included\"}\n";
  if(bad || baseline_samples_bad) throw std::runtime_error("Benchmark correctness failed; speedups must not be used as valid acceleration evidence.");
}

int main(int argc,char** argv) {
  try {
    int device=0,warmup=5,iters=30; bool correctness=true,bench=true,allow_other=false;
    for(int i=1;i<argc;++i) {
      std::string arg=argv[i];
      auto value=[&]() { if(++i>=argc) throw std::runtime_error("Missing value for "+arg); return std::stoi(argv[i]); };
      if(arg=="--device") device=value();
      else if(arg=="--iters") iters=value();
      else if(arg=="--warmup") warmup=value();
      else if(arg=="--correctness-only") bench=false;
      else if(arg=="--benchmark-only") correctness=false;
      else if(arg=="--allow-other-gpu") allow_other=true;
      else if(arg=="--help") {
        std::cout<<"Usage: v100_packing_probe [--device N] [--iters N] [--warmup N] "
          "[--correctness-only|--benchmark-only] [--allow-other-gpu]\n"; return 0;
      } else throw std::runtime_error("Unknown argument: "+arg);
    }
    if(iters<1 || iters>10000 || warmup<0 || warmup>1000) throw std::runtime_error("Invalid iteration count.");
    if(!correctness && !bench) throw std::runtime_error("Choose at most one of --correctness-only and --benchmark-only.");
    int count=0; CUDA_OK(cudaGetDeviceCount(&count));
    if(device<0 || device>=count) throw std::runtime_error("Selected CUDA device does not exist.");
    CUDA_OK(cudaSetDevice(device)); cudaDeviceProp p{}; CUDA_OK(cudaGetDeviceProperties(&p,device));
    int driver=0,runtime=0; CUDA_OK(cudaDriverGetVersion(&driver)); CUDA_OK(cudaRuntimeGetVersion(&runtime));
    bool actual_v100 = p.major==7 && p.minor==0 && std::string(p.name).find("V100")!=std::string::npos;
    std::cout<<std::setprecision(10);
    std::cout<<"{\"type\":\"environment\",\"gpu\":"<<quote(p.name)<<",\"device\":"<<device
      <<",\"compute_capability\":"<<quote(std::to_string(p.major)+"."+std::to_string(p.minor))
      <<",\"actual_V100\":"<<(actual_v100 ? "true":"false")
      <<",\"SMs\":"<<p.multiProcessorCount<<",\"global_memory_bytes\":"<<p.totalGlobalMem
      <<",\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime
      <<",\"build_cuda_version\":"<<CUDART_VERSION
      <<",\"note\":\"Only this program running on actual_V100=true is V100 hardware evidence.\"}\n";
    if(p.major<7) throw std::runtime_error("WMMA requires compute capability >=7.0.");
    if(!actual_v100 && !allow_other) throw std::runtime_error("This is not a V100. Pass --allow-other-gpu only for a separately labelled comparison.");
    if(correctness) {
      unsigned seed=20261122;
      two_sided_ternary(16,128,false);
      two_sided_ternary(16,128,true);
      two_sided_ternary(32,256,false);
      two_sided_ternary(32,256,true);
      full_e2m1_fixed_radix_collision();
      for(const auto& entry: std::vector<std::pair<std::string,const std::vector<int>*>>{
            {"ternary_q",&TERNARY},{"E2M1_abs_le_1.5_q",&NARROW},{"full_E2M1_q",&FULL}})
        for(int shift: {3,4,5,6,7,8,9,10}) for(int k:{16,32,64,128,512})
          for(bool extreme:{false,true}) correctness_case(entry.first,*entry.second,shift,k,extreme,seed++);
      for(int shift:{4,5,6,7,8,9}) {
        two_sided_probe(NARROW,"E2M1_abs_le_1.5_q",shift);
        two_sided_probe(FULL,"full_E2M1_q",shift);
      }
    }
    if(bench) {
      cublasHandle_t handle; BLAS_OK(cublasCreate(&handle));
      BLAS_OK(cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH));
      int version=0; BLAS_OK(cublasGetVersion(handle,&version));
      std::cout<<"{\"type\":\"cublas_environment\",\"version\":"<<version<<"}\n";
      benchmark(handle,"E2M1_abs_le_1.5_q",NARROW,512,512,16,9,false,warmup,iters);
      benchmark(handle,"ternary_q",TERNARY,512,512,64,8,false,warmup,iters);
      benchmark(handle,"E2M1_abs_le_1.5_q",NARROW,512,512,512,9,true,warmup,iters);
      BLAS_OK(cublasDestroy(handle));
    }
    CUDA_OK(cudaDeviceSynchronize());
    std::cout<<"{\"type\":\"completed\",\"success\":true}\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"Probe failed: "<<e.what()<<"\n";
    return 2;
  }
}
