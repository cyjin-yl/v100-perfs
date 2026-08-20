// 微基准: 量化 cudaGetDeviceProperties / cudaGetDevice / cudaDeviceGetAttribute
// 的单次调用开销。
//
// 动机: src/devices/cuda/fastllm-iq4xs-sm70.cu 的 s70_current_device() 在
// **每一次 IQ4_XS 线性层调用**上都执行 cudaGetDeviceProperties, 而且排在
// n/m/k 这些"零成本"的形状判定之前。decode 时 n=1..3 必然被 "n range" 拒绝,
// 也就是说这次昂贵的查询 100% 是白花的。
// Q5_K_M 权重不是 IQ4_XS, 整个 trial 分支根本不进 —— 所以这项开销是
// **IQ4_XS 独有**的, 正好能解释"换 IQ4_XS 反而更慢"。
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    if (cudaSetDevice(0) != cudaSuccess) { printf("no cuda\n"); return 1; }
    cudaFree(0); // 建上下文

    const int N = 20000;
    // 预热
    for (int i = 0; i < 100; i++) { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) { cudaGetDevice(&device); }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) { cudaDeviceProp p; cudaGetDeviceProperties(&p, device); }
    auto t2 = std::chrono::high_resolution_clock::now();
    int major = 0;
    for (int i = 0; i < N; i++) { cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); }
    auto t3 = std::chrono::high_resolution_clock::now();

    auto us = [](auto a, auto b){ return std::chrono::duration<double, std::micro>(b-a).count(); };
    printf("cudaGetDevice            : %8.3f us/call\n", us(t0,t1)/N);
    printf("cudaGetDeviceProperties  : %8.3f us/call\n", us(t1,t2)/N);
    printf("cudaDeviceGetAttribute   : %8.3f us/call\n", us(t2,t3)/N);
    printf("sizeof(cudaDeviceProp)   : %zu bytes\n", sizeof(cudaDeviceProp));
    int rt=0; cudaRuntimeGetVersion(&rt);
    printf("cuda runtime             : %d\n", rt);
    return 0;
}
