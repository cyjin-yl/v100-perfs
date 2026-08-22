// 直接测量 FastllmCudaTrySm70Iq4XsMmq() 在 **decode 形状**(n=1)下的每次调用开销。
//
// 为什么要测这个:
//   这个函数在每一次 IQ4_XS 线性层调用上都会被调一遍。decode 的 n=1..3 必然
//   在 "n range"(n<8) 上被拒 —— 也就是说这次调用什么都没干, 但**它的固定开销
//   要乘以每 token 几百次线性调用**。修复前它内部每次都执行
//   cudaGetDeviceProperties(实测 36.5 us/call), 修复后走 thread_local 缓存。
//
//   Q5_K_M 的权重不是 GGML_TYPE_IQ4_XS, 调用点根本不进这个 trial, 开销恒为 0
//   —— 所以这项开销**只在 IQ4_XS 档位出现**, 正好会被误读成"IQ4_XS 更慢"。
//
// 对照组 old_style_trial() 精确复刻修复前的实现(每次查 cudaGetDeviceProperties
// 再做同样的形状判定), 便于同口径对比。
//
// 编译: 见同目录 build_bench.sh
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

#include "fastllm.h"
#include "fastllm-kernel-route.h"

extern "C" bool FastllmCudaTrySm70Iq4XsMmq(const void *weight, const void *input,
                                           void *output, fastllm::DataType dataType,
                                           int n, int m, int k, void *stream);

// 修复前的实现(逐字复刻 s70_current_device 的旧版本 + 同样的判定顺序)
static bool old_style_trial(int n, int m, int k) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { cudaGetLastError(); return false; }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { cudaGetLastError(); return false; }
    int sm = prop.major * 10 + prop.minor;
    if (sm != 70) return false;
    if (n < 8 || n > 64) return false;
    if (m <= 0 || (m % 256) != 0) return false;
    if (k < 128) return false;
    return true;
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) { printf("no cuda\n"); return 1; }
    cudaFree(0);

    // Qwen3.8-27B 最热的线性层形状之一: ffn_gate 5120 -> 17408
    const int m = 5120, k = 17408;
    const int N = 20000;

    for (int i = 0; i < 200; i++) {
        old_style_trial(1, m, k);
        FastllmCudaTrySm70Iq4XsMmq((const void*)1, (const void*)1, (void*)1,
                                   fastllm::DataType::FLOAT16, 1, m, k, nullptr);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) old_style_trial(1, m, k);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        FastllmCudaTrySm70Iq4XsMmq((const void*)1, (const void*)1, (void*)1,
                                   fastllm::DataType::FLOAT16, 1, m, k, nullptr);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    auto us = [](auto a, auto b){ return std::chrono::duration<double, std::micro>(b-a).count(); };
    double oldUs = us(t0,t1)/N, newUs = us(t1,t2)/N;
    printf("decode 形状 n=1 m=%d k=%d, 每次 trial 调用:\n", m, k);
    printf("  修复前(每次 cudaGetDeviceProperties): %9.4f us/call\n", oldUs);
    printf("  修复后(thread_local 缓存)           : %9.4f us/call\n", newUs);
    printf("  节省                                : %9.4f us/call (%.0fx)\n",
           oldUs - newUs, newUs > 0 ? oldUs / newUs : 0.0);
    for (int calls : {400, 450, 500}) {
        printf("  按每 token %d 次 IQ4_XS 线性调用估算: %6.2f ms/token -> %6.3f ms/token\n",
               calls, oldUs * calls / 1000.0, newUs * calls / 1000.0);
    }
    return 0;
}
