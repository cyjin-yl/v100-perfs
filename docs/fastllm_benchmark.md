# FastLLM on V100 — Build, Patches, and Qwen3.6 Inference Findings

Date: 2026-07-28
Hardware: NVIDIA V100 (SM70, 32GB VRAM, 16GB SM, 64GB system RAM)
Software: Fedora 44, GCC 16.1.1 (system), GCC 12.4.0 (conda), CUDA 12.9

## Outcome

FastLLM **builds** on this machine and **loads** the David Fable Fusion Qwen3.6 27B GGUF (IQ4_XS, with MTP) into the V100. However, **inference deadlocks** because fastllm's `qwen3_5` implementation does not support the SSM (state space) layers that Qwen3.6 introduces on top of Qwen3.5. The llama.cpp backend continues to be the working production path.

## What works

1. **Build** with the patches documented below — produces `main`, `apiserver`, `benchmark`, `quant`, `pyfastllm`, and `libfastllm_tools.so`.
2. **Model load** — all 100 blocks (blk.0..blk.64 plus MTP head) load into GPU in ~3 minutes. Tokenizer initializes successfully.

## What's blocked

- `generate()`, `response_logits()`, `response()`, and `stream_response()` all crash during forward pass with `std::system_error: Resource deadlock avoided` (Linux `EDEADLK`).
- Root cause: the GGUF reports ~7 SSM tensors per block (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`). FastLLM logs these as `unmatched weight` for every layer and has no forward implementation for them, causing the deadlock in the inference thread.
- Workaround would require implementing SSM forward + MTP head forward in fastllm — significant work, out of scope for this iteration.

## Required source patches (2 lines total)

Both patches are minimal and safe to upstream.

### Patch 1 — GGUF architecture name mapping

File: `src/model.cpp`, function `ConvertGGUFTypeToFastllmType`

GGUF metadata uses `general.architecture = "qwen35"` for Qwen3.6, but fastllm's
internal model type is `"qwen3_5"`. The lookup dict was missing the alias.

```diff
 static std::map <std::string, std::string> ggufTypeToFastllmTypeDict = {
     {"qwen2", "qwen2"},
     {"qwen3moe", "qwen3_moe"}, {"qwen3_moe", "qwen3_moe"},
+    {"qwen35", "qwen3_5"}, {"qwen3_5", "qwen3_5"},
     {"glm4_moe", "glm4_moe"},
     {"glm-dsa", "glm_moe_dsa"}, {"glm_moe_dsa", "glm_moe_dsa"},
     {"minimax_m2", "minimax_m2"},
     {"deepseek2", "deepseek_v2"}, {"deepseek_v2", "deepseek_v2"}, {"deepseek_v3", "deepseek_v2"}
 };
```

### Patch 2 — IQ4_XS dequantize function pointer

File: `third_party/gguf/gguf.cpp`, the `type_traits` table for `GGML_TYPE_IQ4_XS`

The `to_float` function pointer was commented out, causing
`WeightImportGGUFTensor: weight ... can't convert to fp32`. The function
`dequantize_row_iq4_xs` exists in `ggml-dequantize.cpp`; just needs to be wired in.

There are two `type_traits` tables in this file (legacy C-style and modern
designated-initializer); both need the same one-line change.

```diff
 {GGML_TYPE_IQ4_XS, {
     .type_name                = "iq4_xs",
     .blck_size                = QK_K,
     .type_size                = sizeof(block_iq4_xs),
     .is_quantized             = true,
-    // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
+    .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
     // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
 }},
```

And the legacy C-style table:

```diff
 {GGML_TYPE_IQ4_XS, {/* type_name */"iq4_xs", /* blck_size */QK_K,
     /* type_size */ sizeof(block_iq4_xs),/* is_quantized */  true,
-    // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
+    /* to_float */ (ggml_to_float_t) dequantize_row_iq4_xs,
     // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
 }},
```

## Build commands (Fedora 44 + CUDA 12.9)

```bash
# 1. Init pybind11 submodule (only needed for PY_API build)
git submodule update --init third_party/pybind11

# 2. CMake configure (non-Python build — produces main, apiserver, benchmark, libfastllm_tools.so)
cmake .. \
  -DUSE_CUDA=ON \
  -DUSE_NUMAS=OFF \
  -DCMAKE_CUDA_COMPILER=/tmp/nvcc-wrapper \
  -DCUDA_ARCH="70" \
  -DCMAKE_CXX_COMPILER=/path/to/conda-gcc-12 \
  -DCMAKE_CXX_FLAGS="-I/path/to/conda/include" \
  -DCMAKE_CUDA_FLAGS="-I/path/to/conda/include"

# 3. Build (use -j2 or -j4, swap pressure makes -j8 unstable)
make -j2 main apiserver benchmark fastllm_tools
```

The conda `include` path is needed because:
- `third_party/pybind11` requires Python headers
- `fastllm` requires `numa.h` (only available via system `numactl-devel`)
- Multicuda requires `nccl.h`

The `nvcc-wrapper` script points nvcc at the conda GCC 12 (system GCC 16 is
incompatible with CUDA 12.9 even with `--allow-unsupported-compiler`).

## Notable build pitfalls encountered

1. **GCC 16 vs nvcc** — system GCC 16.1.1 cannot compile CUDA even with
   `--allow-unsupported-compiler`. Must use conda GCC 12.
2. **Swap thrashing at high parallelism** — `-j8` causes OOM kills on CUDA
   compile due to 48GB swap saturation. Use `-j2` or `-j4`.
3. **NUMAS is incompatible** — system `libnuma.so` references GLIBC 2.38
   symbols that conda's older glibc lacks. Disable with `-DUSE_NUMAS=OFF`.
4. **`std::bad_alloc` on first model load** — caused by `/usr/include` leaking
   into CUDA include path via `CPATH` (math header conflicts). Fix: symlink
   `numa.h` and `nccl.h` into conda's `include/` instead, no `CPATH=/usr/include`.

## KV cache dtype values (gotcha)

`int8` is **not** a valid `kv_cache_dtype`. Valid values:
`auto`, `float32`, `float16`, `bfloat16`, `fp8_e4m3`. Passing `int8` silently
errors after model load completes.

## Token benchmark results

No benchmark numbers were collected because inference deadlocks before any
token is produced. The Qwen3.6 SSM architecture support is the gating work
item for any future fastllm speed comparison.

## Recommendation

Continue running the existing llama.cpp backend for the David Fable Fusion
Qwen3.6 model. The fastllm build artifacts are kept for future use when
Qwen3.6 SSM support lands upstream.
