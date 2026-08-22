# Handoff — FastLLM Service on V100 for Qwen3.6 Fable Fusion

**Author:** Sisyphus session (2026-07-28)
**Project root:** `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/`
**Status:** ✅ Trunk + MTP9 inference works. Exact-window 128K and 160K complete with FP16 KV; exact 256K completes with FP8 E4M3 KV. FP16 KV does not fit exact 256K on the 32GB V100. Native SM70 paged XQA is implemented, benchmarked, and pushed to upstream PR #705. ⚠️ No FastLLM/proxy service is currently running, and the independent fifth-backend proxy refactor is not finished.

This document is self-contained. Anyone picking this up does **not** need to read prior session logs — everything needed to continue is here.

---

## 1. Goal

Deploy **fastllm** as an alternative inference backend for the **Qwen3.6 27B "Fable Fusion"** model (David's variant), running on this V100, alongside the existing `thinking_proxy.py`. The end state is:

1. FastLLM `apiserver` running on a port (e.g. 8002) serving the same Qwen3.6 model as llama.cpp.
2. `thinking_proxy.py` able to route to FastLLM as an independent backend alongside llama.cpp.
3. Benchmark numbers at 128K / 160K / 256K context comparing fastllm vs llama.cpp (prefill t/s, decode t/s).
4. A clean upstream PR to `ztxz16/fastllm` covering the Qwen3.5/3.6 GGUF, runtime, MTP, and V100 IQ4_XS changes (see §7).

The runtime implementation, adapter unit tests, MTP9 smoke tests, exact-window context matrix, and native SM70 `qLen=1` XQA optimization now work. The remaining deployment work is to turn the current FastLLM replacement mode into a genuinely independent fifth backend, implement structured streaming tool-call conversion, and run live OpenAI/Anthropic tool-result round trips.

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

Available variants of the same model (same supported Qwen3.5/3.6 GDN architecture, different quantization and MTP contents):
- `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-IQ3_M.gguf` (14 GB)
- `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-IQ4_XS.gguf` (16 GB, no MTP)
- `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf` (15 GB, **with MTP — the one in use**)

## 5. What works today (verified)

- **Build** — all 6 targets compile cleanly with the patches in §7. See §3.6 for the exact commands.
- **Model load** — both C++ `apiserver` and Python `ftllm.llm.model(...)` load the full 27B model into V100 in ~3 minutes (159s cold, 220s warm). All 100 weight blocks (`Loading 0` → `Loading 100`) succeed. IQ4_XS weights dequantize via the patched `to_float` pointer.
- **Python bindings importable** — `/home/ezra/.conda/envs/tsenv/bin/python3` with `sys.path` pointing at `build/tools/` loads `ftllm` and `ftllm.llm` cleanly. `pyfastllm` C extension initializes and prints CPU instruction info.

## 6. Current inference status

The earlier conclusion that FastLLM lacked Qwen3.6 SSM/GDN and MTP execution was incorrect. The current source includes Qwen3.5/3.6 linear-attention/GDN kernels, full-attention paths, and an MTP draft head. The actual GGUF issues were architecture naming, tensor mapping, V-head export layout, MTP tensor routing, and runtime metadata.

Verified current behavior:
- Real 27B trunk inference returns HTTP 200.
- MTP9 executes speculative validation and records partial acceptance, rather than merely accepting an environment flag.
- The SM70 IQ4_XS MMQ path is selected on V100 and retains legacy fallbacks.
- Focused `qwen35_gguf` and `gguf_dequant` regressions pass; the full `regressionOps` suite passes on the isolated latest-upstream branch.
- Exact requests complete at 131,072 and 163,840 total tokens with FP16 KV, and at 262,144 total tokens with FP8 E4M3 KV. All successful runs returned 256 completion tokens and correct continuation output.
- Exact 256K with FP16 KV fails deterministically after reaching 31,642 MiB and requesting an additional 874,086,400-byte `lm_head.weight` allocation. Disabling CUDA embedding repeats the same failure. The successful FP8 run peaks at 31,088 MiB.
- The final benchmark traces were exclusive-GPU runs: the earlier ComfyUI PID was no longer running when they started.
- Full-history/worktree audit shows that FastLLM's baseline native split-KV stack is pre-existing upstream code, not a FlashInfer-SM70 port from this work: `38b4b8b8` added the native fallback on 2026-05-31, `adb30fd0` optimized SM70 bandwidth, and `bd9a24ad` added chunked-cuBLAS prefill.
- No independent FlashInfer-SM70 backend, planner, Volta MMA layer, BM32/`ALL_P` prefill kernel, or benchmark artifact landed in the current FastLLM worktree/PR. Bundled upstream FlashInfer remains disabled on CC 7.0. `b693ad8` is the attention change that did land: it adapts the Flash-Attention-V100 GQA6 K/V-reuse strategy into FastLLM's page-128 `SM70 paged XQA` route.
- Post-`b693ad8`, native SM70 paged XQA is selected for exact FP16 `page128/Q24/KV4/D256/GQA6/qLen1` decode. Its single-layer attention microbenchmark is 2.22× faster at 8K, 3.37× at 32K, and 4.03× at 128K than the original per-Q-head native split route.
- The XQA phase-1 cubin uses 136 registers and 30,912 bytes shared memory with zero local memory/spill. Eager, partial/non-sequential pages, batch 2, CUDA graph replay, dual-PTDS concurrency, and rejection paths pass.
- MTP9 target validation uses `qLen=10` and deliberately remains on the native causal prefill path; ordinary greedy/seed `qLen=1` decode can use XQA.
- The 128K/160K/256K exact-window artifacts predate `b693ad8`; they validate FastLLM's existing native attention stack, not the final XQA increment. The final increment has focused/full regression, kernel-microbenchmark, and short 27B smoke evidence, but the full exact-window matrix has not yet been rerun.
- These are completed benchmark/smoke results. At the latest live audit, ports 8000/8001/8002 were closed and no GPU inference process was active.

The embedded macro-heavy Jinja template must be rendered by the proxy-side Jinja2 adapter and sent as `raw_prompt=true`; no FastLLM Jinja-parser extension is required. The adapter code and unit tests exist locally, but the end-to-end independent-backend deployment is not complete.

## 7. Current upstream contribution

The isolated branch is based on current upstream `origin/master` and is published as [ztxz16/fastllm#705](https://github.com/ztxz16/fastllm/pull/705). The PR is open, non-draft, and reports a clean merge state. Its eight Chinese intent-split commits cover:

1. API server prompt lifetime and disconnected-client handling (`d7fecea`).
2. Low-memory embedding boundary and backing-storage validation (`1aade2c`).
3. MTP CLI validation through nine drafts (`24c7ff9`).
4. End-to-end Qwen3.5/3.6 GGUF support and SM70 IQ4_XS MMQ (`cfebc74`).
5. Exact-window page-budget fixes (`97a10e2`).
6. CPU-only Qwen3.5 GGUF loading (`4dbd114`).
7. Cross-configuration GGUF regression hardening (`6adcef1`).
8. Native SM70 paged XQA decode and its correctness/performance fixtures (`b693ad8`).

Validation on the isolated latest-upstream tree:
- `regressionOps` and `apiserver` build successfully.
- Focused `qwen35_gguf`, `gguf_dequant`, and `sm70_paged_xqa` regressions pass on their applicable configurations.
- Full `./regressionOps` exits 0, including current upstream NUMA regressions; unavailable Triton/two-GPU cases skip as expected.
- Retained 27B smoke summaries record HTTP 200 for greedy and MTP9; the greedy summary records `SM70 paged XQA enabled`, while MTP9 records speculative full/partial acceptance. The raw post-`b693ad8` service log was not retained.

## 8. Work remaining
1. Rerun the exact 128K/160K/256K matrix on branch tip `b693ad8` so the final XQA increment has end-to-end long-context evidence, not only kernel and short-smoke evidence.
2. Finish the independent FastLLM fifth-backend proxy architecture. The current `FASTLLM_MODE` still replaces the existing llama backend instead of coexisting with it.
3. Add OpenAI streaming tool-call deltas and Anthropic `tool_use` / `input_json_delta`, then run live two-turn tool-result round trips.
4. Deploy FastLLM on port 8002 with `--tokens 262144 --kv_cache_dtype fp8_e4m3`, deploy the proxy on 8000, and verify TCP readiness and local-only alias routing.
5. Keep the TurboQuant 256K result as a different-workload reference; do not present it as a direct cold exact-window A/B.

## 9. Thinking-proxy integration

The adapter layer is implemented and unit-tested, but the required independent-backend wiring is **not complete**:
- `fastllm_adapter.py` renders the macro-heavy template with Jinja2, sends `raw_prompt=true`, splits reasoning, and promotes generated XML calls into OpenAI-compatible `message.tool_calls` for non-stream responses.
- Historical tool-call arguments are normalized for multi-turn result injection.
- TCP readiness code exists because the C++ apiserver has no `/health` or `/v1/models` endpoint.
- However, `thinking_proxy.py` still aliases `FASTLLM_MODE` to the single local backend, overwrites `BACKEND_URL`, and disables llama-server spawning. It therefore replaces llama rather than adding FastLLM as the independent fifth backend.
- OpenAI/Anthropic structured streaming and live tool-result round trips remain unverified.

## 10. Reproduction

Focused build and regression:
```bash
cmake --build build --target regressionOps apiserver -j4
FASTLLM_REGRESSION_ONLY=qwen35_gguf build/regressionOps
FASTLLM_REGRESSION_ONLY=gguf_dequant build/regressionOps
FASTLLM_REGRESSION_ONLY=sm70_paged_xqa build/regressionOps
build/regressionOps
```

Production-path service smoke:
```bash
FASTLLM_QWEN35_ENABLE_MTP=9 FASTLLM_QWEN35_MTP_PROFILE=2 \
  build/apiserver -p /path/to/model.gguf -t 2 -l --atype float16 \
  --kv_cache_dtype fp8_e4m3 --batch 1 --tokens 262144 \
  --model_name qwen3.6-fastllm --port 8002 --device cuda --cuda_embedding
```

Use TCP port readiness, then send a pre-rendered prompt with `raw_prompt=true`. The verified exact 256K configuration uses FP8 E4M3 KV; omit `--kv_cache_dtype fp8_e4m3` for the verified FP16 128K/160K runs.

The command above is a verified 256K-capable configuration, not a currently active daemon. The latest audit found no listener on 8002. The proxy on 8000 and the existing llama backend on 8001 were also not running.

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
| Thinking proxy | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/thinking_proxy.py` |
| llama.cpp server (working baseline) | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/llama.cpp-turboquant/build/bin/llama-server` |
| Build and benchmark notes | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/docs/benchmarks/fastllm-qwen36-legacy.md` |
| Conda env | `/home/ezra/.conda/envs/tsenv/` |
| nvcc wrapper (recreate after reboot) | `/tmp/nvcc-wrapper` (content in §3.2) |

## 12. Current caveats

- Exact-window result artifacts and methodology are in `docs/benchmarks/fastllm-qwen36-legacy.md`; only the 160K client captured a reliable first-content TTFT, so no TTFT is claimed for 128K or 256K.
- The official Python `ftllm bench` runner enters a different FP32 GGUF matmul path and aborted during 128K warmup, so it is not used for production-path numbers.
- TurboQuant 256K has a completed historical result and VRAM trace, but its prefix reuse/concurrency/cache workload differs from the cold exact-window FastLLM runs.
- The FastLLM Jinja parser still does not support the embedded macros; proxy-side Jinja2 rendering is the planned workaround, but the independent fifth-backend deployment is not currently active.
