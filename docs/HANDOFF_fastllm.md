# Handoff — FastLLM Service on V100 for Qwen3.6 Fable Fusion

**Author:** Sisyphus session (2026-07-28)
**Project root:** `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/`
**Status:** ✅ Short-context trunk + MTP9 inference works. Long-context capacity initializes through 256K, but exact full-window prefill requests currently stall or time out. An isolated upstream branch is built and regression-tested.

This document is self-contained. Anyone picking this up does **not** need to read prior session logs — everything needed to continue is here.

---

## 1. Goal

Deploy **fastllm** as an alternative inference backend for the **Qwen3.6 27B "Fable Fusion"** model (David's variant), running on this V100, alongside the existing `thinking_proxy.py`. The end state is:

1. FastLLM `apiserver` running on a port (e.g. 8002) serving the same Qwen3.6 model as llama.cpp.
2. `thinking_proxy.py` able to route to fastllm as a backend, in parallel with or instead of llama.cpp.
3. Benchmark numbers at 128K / 160K / 256K context comparing fastllm vs llama.cpp (prefill t/s, decode t/s).
4. A clean upstream PR to `ztxz16/fastllm` with the two-line patch that fixes Qwen3.6 GGUF loading (see §7).

The implementation, proxy adapter, short tool-call roundtrips, and MTP9 smoke tests now work. The remaining performance blocker is long-context prefill completion, not missing SSM/GDN operators.

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

## 6. Current inference status

The earlier conclusion that FastLLM lacked Qwen3.6 SSM/GDN and MTP execution was incorrect. The current source includes Qwen3.5/3.6 linear-attention/GDN kernels, full-attention paths, and an MTP draft head. The actual GGUF issues were architecture naming, tensor mapping, V-head export layout, MTP tensor routing, and runtime metadata.

Verified current behavior:
- Real 27B trunk inference returns HTTP 200.
- MTP9 executes speculative validation and records partial acceptance, rather than merely accepting an environment flag.
- The SM70 IQ4_XS MMQ path is selected on V100 and retains legacy fallbacks.
- Focused `qwen35_gguf` and `gguf_dequant` regressions pass; the full `regressionOps` suite passes on the isolated latest-upstream branch.
- FastLLM capacities of 131,072, 163,840, and 262,144 tokens all initialize while ComfyUI remains resident.
- Exact prompts of 130,816 / 163,584 / 261,888 tokens did not complete within 3,600 / 900 / 900 seconds. This is the current benchmark blocker.

The embedded macro-heavy Jinja template is still rendered by `thinking_proxy.py` with Jinja2 and sent as `raw_prompt=true`; no FastLLM Jinja-parser extension is required for this deployment.

## 7. Current upstream contribution

The isolated branch is based on current upstream `origin/master`, not the old local checkout. It contains four Chinese, intent-split commits:

1. API server prompt lifetime and disconnected-client handling.
2. Low-memory embedding boundary and backing-storage validation.
3. MTP CLI validation through nine drafts.
4. End-to-end Qwen3.5/3.6 GGUF support: architecture validation, metadata/tensor mapping, V-head inverse permutation, MTP layer routing, runtime layout markers, SM70 IQ4_XS MMQ, ROCm fallback stubs, and behavior regressions.

Validation on the isolated latest-upstream tree:
- `regressionOps` and `apiserver` build successfully.
- `FASTLLM_REGRESSION_ONLY=qwen35_gguf ./regressionOps` passes.
- `FASTLLM_REGRESSION_ONLY=gguf_dequant ./regressionOps` passes on V100.
- Full `./regressionOps` exits 0, including current upstream NUMA regressions.
- Real 27B MTP9 smoke returns HTTP 200 and `OK`.

## 8. Work remaining

1. Diagnose the long-context prefill stall observed after successful 128K/160K/256K capacity startup.
2. Repeat the exact-window benchmark after that fix and record TTFT/prefill, decode throughput, MTP acceptance, and continuous VRAM.
3. Complete final review of the isolated upstream branch and publish the upstream contribution.
4. Keep TurboQuant 256K as the working reference: the completed two-slot run sustained 15.65 aggregate tok/s with 256-token outputs.

## 9. Thinking-proxy integration

The integration is implemented rather than planned:
- FastLLM is an independent backend selected by the exact `qwen3.6-fastllm` alias.
- `thinking_proxy.py` renders the macro-heavy template with Jinja2, sends `raw_prompt=true`, and converts FastLLM XML tool calls into OpenAI-compatible `message.tool_calls`.
- Historical tool-call arguments are normalized for multi-turn result injection.
- FastLLM readiness uses TCP because the C++ apiserver has no `/health` or `/v1/models` endpoint.
- Non-FastLLM local and cloud backends remain available; the alias is local-only and cannot silently spill to a cloud model.

## 10. Reproduction

Focused build and regression:
```bash
cmake --build build --target regressionOps apiserver -j4
FASTLLM_REGRESSION_ONLY=qwen35_gguf build/regressionOps
FASTLLM_REGRESSION_ONLY=gguf_dequant build/regressionOps
build/regressionOps
```

Production-path service smoke:
```bash
FASTLLM_QWEN35_ENABLE_MTP=9 FASTLLM_QWEN35_MTP_PROFILE=2 \
  build/apiserver -p /path/to/model.gguf -t 2 -l --atype float16 \
  --batch 1 --tokens 2048 --model_name qwen3.6-fastllm \
  --port 8002 --device cuda
```

Use TCP port readiness, then send a pre-rendered prompt with `raw_prompt=true`. The verified short smoke returns HTTP 200; exact full-window prompts currently reproduce the long-context timeout described in §6.

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
| Build and benchmark notes | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/docs/fastllm_benchmark.md` |
| Conda env | `/home/ezra/.conda/envs/tsenv/` |
| nvcc wrapper (recreate after reboot) | `/tmp/nvcc-wrapper` (content in §3.2) |

## 12. Current caveats

- The published FastLLM throughput table remains empty because all exact-window requests timed out; only capacity startup and failure timing are claimed.
- The official Python `ftllm bench` runner enters a different FP32 GGUF matmul path and aborted during 128K warmup, so it is not used for production-path numbers.
- TurboQuant 256K has a completed historical result and VRAM trace. It was not rerun while ComfyUI remained resident because their combined VRAM requirement exceeds 32GB.
- The FastLLM Jinja parser still does not support the embedded macros; proxy-side Jinja2 rendering is the deployed workaround.
