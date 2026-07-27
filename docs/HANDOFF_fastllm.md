# Handoff — FastLLM Service on V100 for Qwen3.6 Fable Fusion

**Author:** Sisyphus session (2026-07-28)
**Project root:** `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/`
**Status:** ⏸ Blocked at inference — model loads but forward pass deadlocks. Patches ready, upstream PR not yet opened.

This document is self-contained. Anyone picking this up does **not** need to read prior session logs — everything needed to continue is here.

---

## 1. Goal

Deploy **fastllm** as an alternative inference backend for the **Qwen3.6 27B "Fable Fusion"** model (David's variant), running on this V100, alongside the existing `thinking_proxy.py`. The end state is:

1. FastLLM `apiserver` running on a port (e.g. 8002) serving the same Qwen3.6 model as llama.cpp.
2. `thinking_proxy.py` able to route to fastllm as a backend, in parallel with or instead of llama.cpp.
3. Benchmark numbers at 128K / 160K / 256K context comparing fastllm vs llama.cpp (prefill t/s, decode t/s).
4. A clean upstream PR to `ztxz16/fastllm` with the two-line patch that fixes Qwen3.6 GGUF loading (see §7).

The benchmark (step 3) is **blocked** because inference deadlocks. See §6 for the root cause and what work remains.

## 2. Hardware & Software

| | |
|---|---|
| GPU | Tesla V100-PCIE-32GB (SM70, compute capability 7.0), 32768 MiB |
| Driver | 580.159.04 |
| System RAM | 64 GB (often 50+ GB in swap, see §3.4) |
| Swap | 72 GB (frequently near-full under load) |
| OS | Fedora 44 |
| System GCC | 16.1.1 (**incompatible with nvcc — do not use**) |
| Conda env | `tsenv` at `/home/ezra/.conda/envs/tsenv/` — provides GCC 12.4.0 host compiler, CUDA 12.9 toolchain, Python 3.12.11 |
| Conda g++ | `/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++` |
| nvcc | `/home/ezra/.conda/envs/tsenv/bin/nvcc` |
| nvcc wrapper | `/tmp/nvcc-wrapper` (always pass this as `CMAKE_CUDA_COMPILER`) |
| Internet | SSH to github works; HTTPS to github is firewalled (timeout). Always use `git@github.com:...` |

All paths below are absolute. The drive `/run/media/ezra/13D010B6FDBC1A06/` is an external disk; git operations on it are slow, so prefer file-level edits over `git diff`/`git log`.

## 3. Build (already done — for reference if rebuild needed)

### 3.1 Source
Cloned via SSH into `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/`. Upstream HEAD as of this work: `0036333aaaeeb7ca8f12ac771ddfcd40d760ac96` ("修正 Qwen3.5 自动显存与并发预算"). pybind11 submodule was initialized with `git submodule update --init third_party/pybind11`.

### 3.2 nvcc wrapper
`/tmp/nvcc-wrapper` content (recreate if missing — it lives in /tmp and is wiped on reboot):
```bash
#!/bin/bash
exec /home/ezra/.conda/envs/tsenv/bin/nvcc -ccbin=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++ "$@"
```
Without the `-ccbin` flag, nvcc picks up system GCC 16 and every CUDA translation unit fails to compile.

### 3.3 System symlinks (already created)
These exist and must stay:
- `/home/ezra/.conda/envs/tsenv/include/numa.h` → `/usr/include/numa.h`
- `/home/ezra/.conda/envs/tsenv/include/numaif.h` → `/usr/include/numaif.h`
- `/home/ezra/.conda/envs/tsenv/include/numacompat1.h` → `/usr/include/numacompat1.h`
- `/home/ezra/.conda/envs/tsenv/lib/libnuma.so` → `/usr/lib64/libnuma.so`
- `/home/ezra/.conda/envs/tsenv/lib/libnuma.so.1` → `/usr/lib64/libnuma.so.1`

**Do NOT** add `/usr/include` to `CPATH`/`CPLUS_INCLUDE_PATH` — the system `bits/mathcalls.h` declares `cospi`/`sinpi`/`rsqrt` with exception specs that conflict with CUDA's `crt/math_functions.h`, producing ~90 errors in `fastllm-ggml-cuda.cu`. Only the specific `numa*.h` headers are symlinked into conda's include dir.

### 3.4 NUMAS must be OFF
System `libnuma.so` is built against glibc 2.38+ (`__isoc23_strtoull@GLIBC_2.38` etc.). Conda's glibc is older. Even with the symlinks above (which satisfy compilation and the link step), running `ldd` on the resulting binary or the runtime symbol resolution will throw `undefined reference to __isoc23_*`. So **always** configure with `-DUSE_NUMAS=OFF`. Single-V100 machines do not benefit from NUMAS anyway.

### 3.5 Parallelism
Use `-j2` or `-j4` **maximum**. `-j8` reliably hangs at the CUDA phase (~70-80%) because nvcc holds huge resident sets and the 48 GB of swap-in-use triggers the OOM killer on individual nvcc processes. A failed `-j8` run leaves the build in a half-state — just re-run `make -j2` and it resumes.

### 3.6 CMake configure commands

**Variant A — C++ binaries only** (`main`, `apiserver`, `benchmark`, `quant`, `webui`, `libfastllm_tools.so`):
```bash
export PATH="/home/ezra/.conda/envs/tsenv/bin:$PATH"
export LD_LIBRARY_PATH="/home/ezra/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/home/ezra/.conda/envs/tsenv/lib"
export CC=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++
cd /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build
cmake .. \
  -DUSE_CUDA=ON \
  -DUSE_NUMAS=OFF \
  -DCMAKE_CUDA_COMPILER=/tmp/nvcc-wrapper \
  -DCUDA_ARCH="70" \
  -DCMAKE_CXX_COMPILER=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++ \
  -DCMAKE_CXX_FLAGS="-I/home/ezra/.conda/envs/tsenv/include" \
  -DCMAKE_CUDA_FLAGS="-I/home/ezra/.conda/envs/tsenv/include"
make -j2 main apiserver benchmark fastllm_tools
```

**Variant B — add Python bindings** (`pyfastllm.cpython-312-x86_64-linux-gnu.so`):
Same as A, plus `-DPY_API=ON -DPython3_ROOT_DIR=/home/ezra/.conda/envs/tsenv/bin/`. Note that with `PY_API=ON`, the C++ targets `main`/`apiserver`/`benchmark` are **NOT** defined (the CMakeLists uses an `if(PY_API) ... else() ... endif()` block). To get both, run variant B first, copy the `.so` somewhere safe, then re-run variant A.

### 3.7 Post-build layout
After variant A, artifacts are at `build/`:
- `main` — interactive CLI chat (loads model, reads stdin)
- `apiserver` — OpenAI-compatible HTTP server (this is the one to use for service)
- `benchmark` — synthetic prefill/decode bench
- `quant`, `webui` — not used here
- `libfastllm_tools.so` — shared lib used by Python tools

After variant B, additionally:
- `build/pyfastllm.cpython-312-x86_64-linux-gnu.so` — Python module
- Run `build/tools/ftllm/` setup by copying `pyfastllm.*.so`, `libfastllm_tools.so`, and the contents of `tools/fastllm_pytools/` into a single dir. Then `sys.path.insert(0, that_dir)` and `import ftllm.llm`.

### 3.8 Current artifact state (as of handoff)
- `build/main` 55.7 MB, mtime 22:46 ✓ patched
- `build/apiserver` 55.8 MB, mtime 22:51 ✓ patched
- `build/benchmark` 55.7 MB, mtime 22:52 ✓ patched
- `build/libfastllm_tools.so` 57.2 MB ✓ patched
- `build/tools/ftllm/pyfastllm.cpython-312-x86_64-linux-gnu.so` 57.8 MB ✓ patched
- `build/tools/ftllm/libfastllm_tools.so` 57.2 MB ✓ patched
- All include both patches from §7. Verified via `strings ... | grep qwen35` and `strings ... | grep dequantize_row_iq4_xs`.

## 4. Model

| | |
|---|---|
| Path | `models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf` |
| Size | 15 GB |
| Architecture (GGUF `general.architecture`) | `qwen35` |
| Quant | IQ4_XS for most weights, Q5_K for attn_qkv, f32 for norms |
| Layers | 64 transformer blocks + 1 MTP head (`blk.64.nextn.*`) |
| Hybrid | Attention + SSM (state space) layers in every block |
| Multimodal projector | `models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf` (889 MB) |
| Chat template | `chat_templates/qwen3.6_merged.jinja` (14 KB, contains `{%- macro %}` blocks) |

All `models/...` paths are relative to `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/`.

Available variants of the same model (same architecture, different quants — same SSM blocker applies to all):
- `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-IQ3_M.gguf` (14 GB)
- `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-IQ4_XS.gguf` (16 GB, no MTP)
- `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf` (15 GB, **with MTP — the one in use**)

## 5. What works today (verified)

- **Build** — all 6 targets compile cleanly with the patches in §7. See §3.6 for the exact commands.
- **Model load** — both C++ `apiserver` and Python `ftllm.llm.model(...)` load the full 27B model into V100 in ~3 minutes (159s cold, 220s warm). All 100 weight blocks (`Loading 0` → `Loading 100`) succeed. IQ4_XS weights dequantize via the patched `to_float` pointer.
- **Python bindings importable** — `/home/ezra/.conda/envs/tsenv/bin/python3` with `sys.path` pointing at `build/tools/` loads `ftllm` and `ftllm.llm` cleanly. `pyfastllm` C extension initializes and prints CPU instruction info.

## 6. The blocker — inference deadlocks

After successful load, **any** call into the forward pass crashes:
- `model.generate([[tokens]], max_new_tokens=1)` → `terminate called after throwing an instance of 'std::system_error': Resource deadlock avoided` (Linux `EDEADLK`)
- `model.response_logits("...")` → first hits the Jinja parser error (§6.1), then would hit the same deadlock
- `model.response("...")` → same Jinja error
- C++ `apiserver` + `--device cuda` → same `std::bad_alloc` family of errors during first request (same underlying code path)

### 6.1 Root cause #1: SSM layers not implemented

Qwen3.6 is a **hybrid attention + SSM** architecture. Every transformer block contains 7 SSM tensors:
```
blk.N.ssm_a            (48)         f32
blk.N.ssm_alpha.weight (48 × 5120)  iq4_xs
blk.N.ssm_beta.weight  (48 × 5120)  iq4_xs
blk.N.ssm_conv1d.weight(10240 × 4)  f32
blk.N.ssm_dt.bias      (48)         f32
blk.N.ssm_norm.weight  (128)        f32
blk.N.ssm_out.weight   (5120 × 6144) iq4_xs
```
FastLLM's `Qwen3_5Model` (in `src/models/qwen3_5.cpp`) was written for Qwen3.5, which is **attention-only**. It has no forward code for these tensors, so on load it logs `unmatched weight blk.N.ssm_*` for every one of them (hundreds of warnings), and on the first forward pass the inference thread hits a code path that ends in `EDEADLK`. The deadlock is the symptom; the missing implementation is the cause.

### 6.2 Root cause #2: MTP head not implemented

Block 64 is a Multi-Token-Prediction head:
```
blk.64.nextn.eh_proj.weight         (5120 × 10240) iq4_xs
blk.64.nextn.enorm.weight           (5120)         f32
blk.64.nextn.hnorm.weight           (5120)         f32
blk.64.nextn.shared_head_norm.weight(5120)         f32
```
Same story — unmatched on load, would fail at forward time. Even if SSM were implemented, the MTP head needs its own forward pass.

### 6.3 Secondary issue: Jinja chat template parser

The GGUF embeds a chat template starting with `{%- macro render_content(content, do_vision_count, is_system_content=false) %}`. FastLLM's bundled Jinja parser does not support `{% macro %}` blocks, so any API that applies the chat template (`response`, `response_logits`, `stream_response`, `launch_stream_response`) dies with:
```
FastLLM Error: Jinja parse failed (Unknown block type): {%- macro render_content(...) %}
```
**Workaround:** use `model.generate([token_ids], ...)` with pre-encoded token IDs — it bypasses the chat template entirely. This is what any future benchmark/service code should do until the Jinja parser is extended.

### 6.4 Secondary issue: kv_cache_dtype values

`int8` is **rejected** as `kv_cache_dtype`. Valid values are exactly: `auto`, `float32`, `float16`, `bfloat16`, `fp8_e4m3`. V100 has no FP8 hardware so `fp8_e4m3` is software-emulated and slow — use `float16` for benchmarks.

## 7. Patches (already applied to the working tree)

Two minimal edits. Both are safe to upstream — they only add aliases, no behavior change for existing models.

### Patch 1 — `src/model.cpp`, `ConvertGGUFTypeToFastllmType` dict (~line 2110)

Adds `qwen35` → `qwen3_5` alias so GGUF files exported for Qwen3.6 are recognized. Without this, fastllm prints `Warning: Can't convert type "qwen35"` and then `Unsupport graph model type qwen35`.

```diff
         static std::map <std::string, std::string> ggufTypeToFastllmTypeDict = {
             {"qwen2", "qwen2"}, // llama
             {"qwen3moe", "qwen3_moe"}, {"qwen3_moe", "qwen3_moe"}, // qwen3_moe
+            {"qwen35", "qwen3_5"}, {"qwen3_5", "qwen3_5"}, // qwen3_5 (Qwen3.6 GGUF reports as qwen35)
             {"glm4_moe", "glm4_moe"}, // glm4_moe
             {"glm-dsa", "glm_moe_dsa"}, {"glm_moe_dsa", "glm_moe_dsa"}, // glm_moe_dsa
             {"minimax_m2", "minimax_m2"}, // minimax_m2
             {"deepseek2", "deepseek_v2"}, {"deepseek_v2", "deepseek_v2"},  {"deepseek_v3", "deepseek_v2"} // deepseek_v2
         };
```

### Patch 2 — `third_party/gguf/gguf.cpp`, two `type_traits` tables

The `to_float` function pointer for `GGML_TYPE_IQ4_XS` was commented out in **both** tables in this file. The function `dequantize_row_iq4_xs` exists in `ggml-dequantize.cpp` and is declared in `gguf.h` — it just wasn't wired in. Without this patch, IQ4_XS weights fail with `WeightImportGGUFTensor: weight token_embd.weight(type iq4_xs) can't convert to fp32`.

**Designated-initializer table** (~line 543):
```diff
         {GGML_TYPE_IQ4_XS, {
             .type_name                = "iq4_xs",
             .blck_size                = QK_K,
             .type_size                = sizeof(block_iq4_xs),
             .is_quantized             = true,
-            // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
+            .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
             // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
         }},
```

**Legacy C-style table** (~line 206):
```diff
         {GGML_TYPE_IQ4_XS, {/* type_name */"iq4_xs", /* blck_size */QK_K,
             /* type_size */ sizeof(block_iq4_xs),/* is_quantized */  true,
-            // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
+            /* to_float */ (ggml_to_float_t) dequantize_row_iq4_xs,
             // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
         }},
```

A combined patch file is saved at `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/scripts/fastllm_qwen36_iq4xs.patch`.

### 7.1 Upstream PR notes (not yet opened)

- Repo: `git@github.com:ztxz16/fastllm.git` (upstream `master` HEAD `0036333`).
- These two patches are genuinely upstream-safe — they only add an alias and uncomment an existing function pointer. They do not change behavior for any model that already works.
- The **SSM forward implementation** (§6.1, §6.2) is a much larger piece of work and is out of scope for this initial PR. The PR description should explicitly say "loads Qwen3.6 GGUF; full inference still requires SSM support which is being tracked separately".
- Before opening the PR, rebase on latest upstream `master`, verify the patch still applies cleanly, run the existing fastllm test suite if any, and add a short test that loads a small qwen3.6 GGUF and asserts it doesn't throw `Unsupport graph model type`.

## 8. Work remaining (in priority order)

1. **Implement SSM forward in `Qwen3_5Model`** (`src/models/qwen3_5.cpp` + `include/models/qwen3_5.h`). This is the gating item — without it, no benchmark, no service. Reference implementations to crib from:
   - `llama.cpp`'s `llama.cpp/src/llama-graph.cpp` search for `ssm_` tensors (the llama.cpp path runs this model correctly today)
   - Mamba2 papers and the Qwen3.6 technical report for the math
   - The 7 SSM tensors per block and their semantics are listed in §6.1
2. **Implement MTP head forward** (the `blk.64.nextn.*` block). Smaller scope, only 4 tensors. Reference: llama.cpp's `nextn` handling and the original DeepSeek-V3 MTP paper.
3. **Extend Jinja parser** to support `{% macro %}` (or at minimum, strip macro blocks before parsing). Lower priority since `generate()` with raw tokens is a viable workaround.
4. **Once inference works:** run benchmarks at 128K / 160K / 256K context, measure prefill t/s and decode t/s, write up in `v100-perfs/docs/fastllm_benchmark.md`.
5. **Once benchmarks pass:** write the fastllm backend adapter in `thinking_proxy.py` (see §9), stand up `apiserver` on a port, route proxy traffic to it.
6. **Open the upstream PR** (§7.1).

## 9. Thinking-proxy integration spec (for after inference works)

Current `thinking_proxy.py` (line refs as of this session):
- `BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"` where `BACKEND_PORT` defaults to 8001 (line 44, 70)
- `PROXY_PORT` defaults to 8000 (line 45) — this is the public entrypoint
- The llama.cpp backend is started by the proxy itself via `subprocess.Popen` (line 78+) with `llama-server` binary and a long arg list including `-c 262144 -ngl 99 -np 1 -fa on -fit off --cache-type-k turbo4 --cache-type-v turbo4 --spec-type draft-mtp --spec-draft-n-max 9`
- `_pick_backend` (search near line 252) decides routing; `MODEL_HERETIC = "qwen3.6-27b-heretic"` is a model-name substring that triggers a special route
- The model path is read from env `TURBO_MODEL`, defaulting to `models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf`

To add a fastllm backend:
1. Stand up `apiserver` on a new port (suggest `8002`) with `--device cuda --port 8002 -p <model> --model_name qwen3.6-fastllm`. FastLLM's apiserver exposes an OpenAI-compatible `/v1/chat/completions` so it is a drop-in for the existing proxy→backend HTTP contract.
2. In `thinking_proxy.py`, add a `FASTLLM_URL = "http://127.0.0.1:8002"` constant and a routing branch in `_pick_backend`. The simplest initial policy: A/B test a fraction of traffic to fastllm, or route based on a `X-Backend: fastllm` header.
3. The fastllm `apiserver` does **not** need the `-c 262144` flag the llama.cpp path uses — context size is set differently. Check the apiserver `--help` output for the equivalent flag (likely `--tokens`).
4. Spec decoding (`--spec-type draft-mtp`) has no fastllm equivalent yet; fastllm's own MTP support would need to land first (§8 item 2). Until then, fastllm will be slower on decode than llama.cpp's MTP-accelerated path — plan the A/B test accordingly.
5. KV cache dtype: pass `float16` (§6.4).

## 10. Reproduction: how to confirm the current state

To verify the build still works and reproduces the deadlock:
```bash
cd /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build/tools
/home/ezra/.conda/envs/tsenv/bin/python3 -c "
import sys, time
sys.path.insert(0, '.')
import ftllm.llm as llm_mod
llm_mod.set_device_map({'cuda:0': 1})
llm_mod.set_cpu_low_mem('true')
llm_mod.set_cpu_threads(2)
m = '/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf'
t0 = time.time()
model = llm_mod.model(m, dtype='float16', kv_cache_dtype='float16')
print(f'loaded in {time.time()-t0:.1f}s')
ids = model.encode('Hello world')
print(f'got {len(ids)} tokens')
print(model.generate([ids], max_new_tokens=1, do_sample=False))  # will throw EDEADLK
"
```
Expected output: ~3 min load, `got 2 tokens`, then `terminate called after throwing an instance of 'std::system_error': Resource deadlock avoided`. That confirms §6.

## 11. Key files (absolute paths)

| Purpose | Path |
|---|---|
| FastLLM source (patched) | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/` |
| FastLLM build dir | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build/` |
| Python tools dir | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build/tools/ftllm/` |
| apiserver binary | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build/apiserver` |
| Model GGUF (MTP, IQ4_XS) | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf` |
| Multimodal projector | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf` |
| Chat template (jinja) | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/chat_templates/qwen3.6_merged.jinja` |
| Thinking proxy | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/thinking_proxy.py` |
| llama.cpp server (working baseline) | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/llama.cpp-turboquant/build/bin/llama-server` |
| Patch file (combined) | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/scripts/fastllm_qwen36_iq4xs.patch` |
| Build/patch notes (committed) | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/docs/fastllm_benchmark.md` |
| Conda env | `/home/ezra/.conda/envs/tsenv/` |
| nvcc wrapper (recreate after reboot) | `/tmp/nvcc-wrapper` (content in §3.2) |

## 12. Things I did NOT do (so the next person knows)

- **Did not open the upstream PR.** Patches are in the working tree and in `v100-perfs/scripts/fastllm_qwen36_iq4xs.patch`, ready to be cleaned up and pushed.
- **Did not restart the llama.cpp backend.** It was killed earlier in this session to free VRAM for fastllm testing. The proxy was also killed. Whoever picks this up should restart both before relying on them. The original llama-server command line is in §9 (and in the proxy source).
- **Did not collect any benchmark numbers.** Inference is blocked; no t/s data exists for fastllm on this model.
- **Did not implement SSM or MTP forward.** This is the gating work item — see §8 item 1.
- **Did not extend the Jinja parser.** Workaround is `generate()` with raw tokens (§6.3).
