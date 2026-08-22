// 端到端验证算子路由普查: 用真实的 IQ4_XS 权重跑 fastllm::Linear, 看普查表
// 记录下来的路由是不是和预期一致。
//
// 回答的是这个问题: "IQ4_XS 到底有没有走到 DP4A MMQ 快路径, 还是静默退化成
// 反量化 + fp16 GEMM?" —— 用生产的真实张量形状(ffn_gate 5120->17408)和真实的
// n(decode 1..3 / MMQ 区间 16 / prefill 128), 直接把答案打出来。
#include <cstdio>
#include <cstring>
#include <vector>

#include "fastllm.h"
#include "fastllm-kernel-route.h"
#include "gguf.h"

static void FillBlock(block_iq4_xs &blk, uint32_t seed) {
    auto rng = [](uint32_t &s) -> uint8_t {
        s = s * 1103515245u + 12345u;
        return (uint8_t)((s >> 16) & 0xFF);
    };
    uint32_t s = seed;
    uint16_t dBits = 0x0280u | ((uint16_t)rng(s) & 0x007Fu);
    std::memcpy(&blk.d, &dBits, sizeof(dBits));
    blk.scales_h = (uint16_t)(rng(s) | ((uint16_t)rng(s) << 8));
    for (int i = 0; i < 4; i++) blk.scales_l[i] = rng(s);
    for (int i = 0; i < QK_K / 2; i++) blk.qs[i] = rng(s);
}

static void RunOne(int inputDim, int outputDim, int n, const char *label) {
    const int blockSize = (int)ggml_blck_size(GGML_TYPE_IQ4_XS);
    const int blocksPerRow = inputDim / blockSize;
    fastllm::Data weight(fastllm::DataType::DATA_GGUF_FORMAT,
                         (int)GGML_TYPE_IQ4_XS, {outputDim, inputDim});
    weight.isGGUFData = true;
    weight.disableGGUFRepack = true;
    weight.Allocate();
    const size_t bytesPerRow = ggml_row_size(GGML_TYPE_IQ4_XS, inputDim);
    std::vector<block_iq4_xs> rowBlocks(blocksPerRow);
    for (int row = 0; row < outputDim; row++) {
        for (int b = 0; b < blocksPerRow; b++) {
            std::memset(&rowBlocks[b], 0, sizeof(block_iq4_xs));
            FillBlock(rowBlocks[b], (uint32_t)((row * 1000 + b) * 7 + 3));
        }
        std::memcpy(weight.cpuData + (size_t)row * bytesPerRow,
                    rowBlocks.data(),
                    (size_t)blocksPerRow * sizeof(block_iq4_xs));
    }
    weight.name = "probe.ffn_gate";
    weight.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);

    std::vector<float> inputFp32((size_t)n * inputDim);
    for (size_t i = 0; i < inputFp32.size(); i++) {
        inputFp32[i] = (float)((i * 17 + 5) % 101 - 50) / 47.0f;
    }
    // 生产是 --atype float16
    fastllm::Data input(fastllm::DataType::FLOAT16, {n, inputDim}, inputFp32);
    input.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);

    fastllm::ResetKernelRouteCensus();
    fastllm::Data bias, output;
    fastllm::Linear(input, weight, bias, output);
    printf("%-46s n=%-5d %s\n", label, n,
           fastllm::FormatKernelRouteCensus().c_str());
}

int main() {
    fastllm::SetDeviceMap({{"cuda", 1}});
    // Qwen3.8-27B 的 ffn_gate: 5120 -> 17408(为省显存这里用 17408 的 1/8)
    const int m = 5120, k = 2176;
    printf("权重形状 m=%d k=%d, IQ4_XS, 激活 float16\n\n", m, k);
    RunOne(m, k, 1,   "decode(batch1)");
    RunOne(m, k, 3,   "decode(batch1 + MTP2 验证)");
    RunOne(m, k, 8,   "MMQ 区间下界");
    RunOne(m, k, 16,  "MMQ 区间内");
    RunOne(m, k, 64,  "MMQ 区间上界");
    RunOne(m, k, 65,  "刚出 MMQ 区间");
    RunOne(m, k, 1024,"长 prefill chunk(默认 CHUNK_CAP)");
    return 0;
}
