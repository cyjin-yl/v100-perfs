# MiniMax H3 Video Generation on Tesla V100 32GB

This directory is the video-generation companion to the LLM measurements in the
parent `v100-perfs` repository. It records the H3 model/ComfyUI stack, community
recommendations, V100-specific limits, and reproducible local benchmark artifacts.

The benchmark deliberately separates three kinds of evidence:

1. **Official contract** — facts from MiniMax and ComfyUI documentation.
2. **Community evidence** — measurements published for other GPUs or claims that
   still need a V100 A/B test.
3. **Local evidence** — measurements made on the Tesla V100-PCIE-32GB in this
   repository. Local numbers are the only numbers used for the V100 recommendation.

## Executive recommendation

For this V100-PCIE-32GB (SM70), the current quality/speed baseline is:

```text
ComfyUI 0.30.0
PyTorch 2.10.0+cu128
--listen 0.0.0.0 --port 8188 --reserve-vram 2
--cache-none --disable-pinned-memory --force-fp16 --fp16-text-enc

FL2VA:     minimax_h3_fl2va_pruned_int8_convrot.safetensors
Ref2VA:    minimax_h3_ref2va_pruned_int8_convrot.safetensors
Text:      qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
Video VAE: minimax_h3_video_vae_fp16.safetensors
Audio VAE: minimax_h3_audio_vae_fp32.safetensors

864x480 (about 0.4 MP), 24 fps, 56 frames (2.33 s), 20 steps
res_multistep / simple, seed fixed for A/B tests
Director CFG=1.0, video shift=12, audio shift=3
```

Use the Director T2V/FL2V workflows for explicit control, or Easy for a compact
T2V/I2V/FL2V/Ref2V graph. The native ComfyUI T2V/I2V/R2V templates are the best
compatibility reference. The 0.4 MP / 56-frame preset is an engineering baseline,
not a quality ceiling.

The local baseline already proved a complete 864x480 / 56-frame / 20-step T2V
run with H.264 video and AAC audio. See the parent handoff for the exact output
and the JSON artifacts in `results/` for new measurements.

## Hardware and architecture facts

| Item | Local value / implication |
|---|---|
| GPU | Tesla V100-PCIE-32GB, compute capability 7.0 / SM70 |
| VRAM | 32 GB HBM2; ComfyUI reports about 34.1 GB device total due to allocator/device units |
| PyTorch | 2.10.0+cu128; SM70 kernels are available |
| H3 transformer | Dense 33B Omni Transformer; roughly 13B AdaLN-related parameters can be precomputed for inference |
| Video | 24 fps; video VAE spatial compression 16x and temporal compression 4x |
| Audio | Native stereo, 32 kHz output; audio VAE latent rate 40 Hz |
| Frame grid | `n % 17 == 5`; ComfyUI rounds requested length upward |
| Official output | 768p short edge by default; up to 2K through the hosted regeneration stage; about 4–15 s official output |

### SM70 hard limits

V100 has no FP8 Tensor Core execution path. The ComfyUI `fp8_scaled`/NVFP4 and
other low-bit files can still be useful as storage formats, but kernels that
require newer architectures must fall back to eager dequantization or fail.
Similarly, Marlin-style INT4 kernels target Ampere or newer (SM80+); on V100,
INT4 is primarily a memory-saving experiment and must not be assumed to be faster.

V100 also has no native BF16 Tensor Core path. A BF16 model is therefore not the
same speed choice as FP16 on this card. For the local stack, pruned INT8 ConvRot
with ComfyUI's eager/dequant fallback plus `--force-fp16` is the tested route.

Do not force generic FP16 on every H3 checkpoint. ComfyUI issue [#15262](https://github.com/Comfy-Org/ComfyUI/issues/15262)
reports NaN/Inf on a Tesla V100 16GB when a pruned FP8-scaled H3 model is forced
to FP16. The local 32GB INT8 ConvRot path produced valid video, but this does not
prove that FP8-scaled or BF16 variants are numerically safe on SM70. Black frames,
NaN/Inf, and audio corruption are hard failures, not acceptable speed wins.

## Checkpoint, pruning, and quantization survey

The official/comfy-repacked variants currently seen in the community index are:

| Family | Approx. file size | What it changes | V100 interpretation |
|---|---:|---|---|
| Full BF16 | 61.73 GB | Original task transformer | Does not fit as a resident V100 model; BF16 is not native on SM70 |
| Full INT8 ConvRot | 31.70 GB | INT8 weight representation | Still large; kernel support matters |
| Pruned BF16 | 37.46 GB | Removes/precomputes AdaLN-related inference work | Smaller but still too large for a resident 32GB path |
| Pruned FP8 scaled | 19.52 GB | Pruning + FP8 scaled weights | Storage-friendly; FP8 execution is not native on V100; numerical risk under forced FP16 |
| **Pruned INT8 ConvRot** | **19.53 GB** | **Pruning + ConvRot INT8 + AdaLN tables** | **Best tested V100 quality/compatibility baseline** |

The pruning is not an arbitrary layer deletion: the official architecture notes
that about 13B AdaLN-branch parameters can be precomputed/cached for inference.
The pruned checkpoint therefore removes the need to carry those inference-only
parameters while preserving the generation path.

Community quantization families found in `wildminder/awesome-minimax-H3` include:

- **GGUF Q2/Q3/Q4/Q5/Q6/Q8**: approximately 14.5 GB at Q3_K_M through 33.6 GB
  at Q8_0 for a transformer. Requires ComfyUI-GGUF and is attractive when CPU/RAM
  offload is more important than native GPU execution.
- **NVFP4 and mixed INT4/INT8 ConvRot**: approximately 11–19 GB for pruned
  variants. V100 cannot use the intended native FP4/FP8 kernels; benchmark only
  with the actual eager fallback and inspect quality.
- **NF4/bitsandbytes**: approximately 16 GB transformer and 14 GB text encoder
  components in the listed DiffSynth release. This is a different runtime path,
  not a drop-in replacement for the current native ComfyUI stack.
- **W4/W8W4 ConvRot and W4A8 ConvRot**: lower storage, but some releases require
  ComfyUI core/comfy-kitchen branches and SM80+ or newer optimized kernels.
- **OrbitQuant W4A4**: native packed W4A4 route with its own custom node; not
  validated on SM70 and not installed in the production ComfyUI instance.
- **INT8 Lean ConvRot / dynamic-time separate-QKV**: quality-oriented mixed
  precision and experimental patched-core variants. The latter requires a core
  patch and is not a safe first choice for a shared service.

For this V100, the practical order is:

```text
pruned INT8 ConvRot (tested)
  > pruned FP8 scaled (storage only; SM70 numerical/kernel risk)
  > community GGUF Q4/Q5 (requires GGUF path; benchmark separately)
  > mixed INT4/W4/NVFP4 (memory experiments; no presumed speed benefit)
  > full/pruned BF16 (capacity and BF16 execution disadvantage)
```

This is a compatibility/quality ranking, not a universal quality ranking. A
Q4/GGUF or mixed INT4 model can be preferable when the alternative is OOM.

The Qwen3-VL text encoder is itself a major memory consumer. The ComfyUI NVFP4
AWQ encoder is the practical 32GB choice, but it depends on matching ComfyUI and
`comfy-kitchen` quant support. See [ComfyUI #15400](https://github.com/Comfy-Org/ComfyUI/issues/15400)
and [#15241](https://github.com/Comfy-Org/ComfyUI/issues/15241) before changing
versions. Do not upgrade Torch/Transformers casually in the shared environment.

## Workflow recommendations

### Official native templates

ComfyUI ships T2V, I2V, and R2V templates. They are the compatibility baseline:

- T2V/I2V/FL2VA use `fl2va`.
- R2V uses `ref2va`.
- Both use the Qwen3-VL MiniMax CLIP type, video VAE, and audio VAE.
- Resolution is set by megapixels, rounded to multiples of 32.
- Duration is converted to the `17k+5` frame grid.

The official prompt structure is scene overview first, then timed shots, camera
movement, and one integrated audio description (dialogue, SFX, ambience, music).
For R2V, label each reference and assign it a job: identity, style, motion,
camera, or voice. These points affect quality much more than minor scheduler
changes.

### Director and Easy community workflows

The installed, Chinese-localized community pair is complementary:

- [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
  provides a storyboard/timeline and explicit T2V, I2V, FL2V, R2V, V2V, and RV2V
  modes. The [huangserva mirror](https://github.com/huangserva/ComfyUI_MiniMaxH3_Director)
  provides Chinese workflow copies.
- [nkxx188/ComfyUI-MiniMaxH3-Easy](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy)
  provides one compact media input, indexed `@` references, and dialogue blocks.

These are workflow/UI choices, not different H3 weights. Keep the node version,
model filenames, prompt, seed, resolution, frames, steps, sampler, scheduler,
and acceleration settings identical for A/B comparisons.

### Acceleration choices

| Technique | Community evidence | V100 decision |
|---|---|---|
| SageAttention | Official ComfyUI docs say roughly 2x on compatible GPUs; requires FP16/BF16 tensors | Test only with an SM70-compatible build; do not install a random wheel into the shared service |
| Turbo LoRA | 4-step prototype; 6–8 steps recommended for quality; strength 1.0, scheduler simple | Strong candidate for a separate quality/speed experiment once the matching LoRA is available; not part of baseline |
| FirstBlockCache | 5090 report: 1.44x SageAttention2 / 1.49x native, 30–33% lower warm wall time | Approximate trajectory; install only for isolated A/B, never combine with another block cache |
| TE-Speed | README reports about 45% at its reference 30-step/8s setup | Requires patching `comfy_extras` core; preserve backups and validate audio/video on V100 |
| H3 Cache / EasyCache | Patches ComfyUI core and reuses block computations | High-risk shared-service change; benchmark isolated from FirstBlockCache/TE-Speed |
| Spectrum | Forecasts hidden features; v0.2.1 defaults to offline replay and `audio_blend_weight=0` after audio regressions | Interesting research path, but its own README says it is approximate and not lossless; no local V100 claim yet |
| Sol/INT8 attention ports | Some community entries target newer SM89–SM120 | Not presumed compatible with SM70 |

A cache speedup is meaningful only if the same seed output is reviewed for black
frames, audio clipping/stutter, motion drift, and reference identity. Pixel SSIM
alone is not a quality metric for a moving video.

## Parameter guidance for V100

| Goal | Resolution | Frames | Steps | Notes |
|---|---:|---:|---:|---|
| Fast storyboard preview | 416x224–608x352 (0.1–0.2 MP) | 39–56 | 4–8 or 20 | Below official trained duration; use only for composition checks |
| **Balanced V100 default** | **864x480 (~0.4 MP)** | **56 (~2.33 s)** | **20** | Local, successful, audio+video baseline |
| Quality preview | 960x544–1152x640 (0.5–0.7 MP) | 124 (~5.17 s) | 20 | Expect roughly linear pixel cost plus memory pressure |
| Native-size attempt | 1344x768 (~1.0 MP) | 124 | 20 | Quality target; likely too slow for routine V100 use |
| Long-form attempt | 864x480 | 124–362 (~5–15 s) | 20 | Duration cost is close to linear until memory/offload changes |

Use 24 FPS. Do not lower FPS as a speed trick: H3 was trained for 24 FPS and
lowering the output only changes playback timing while the latent work still
uses the generated frame count. `cfg=1.0` is appropriate for the distilled H3
Director path; the native H3 templates use a distilled guider rather than a
traditional high-CFG workflow.

For 32GB V100, reduce resolution before reducing steps when visual coherence is
important. Use 20 steps as the reference baseline; 4–8 steps are only a speed
profile unless a compatible Turbo LoRA is installed. Keep video/audio shifts at
12/3 on the native path. Turn off source-image export and avoid keeping multiple
transformer variants resident.

## Benchmark design

`bench_h3.py` submits the successful API-format native T2V graph, one job at a
time. It fixes the prompt and seed, changes only one independent variable, and
writes a JSON row after every completed job so a stopped run remains useful.

Primary metrics:

```text
execution_wall_s = time(queue_running) -> time(history success)
queue_wait_s     = time(queue_running) - time(POST /prompt)
end_to_end_s     = time(history success) - time(POST /prompt)
```

`execution_wall_s` is the primary user-generation metric. It includes text encoder
loading/encoding, H3 sampling, VAE decode, audio VAE decode, and MP4 muxing, but
not time spent behind an unrelated job already occupying ComfyUI's queue. The
harness records `queue_wait_s` separately and waits for an idle queue before each
case, so a shared workstation cannot silently turn queue contention into a model
speed result. It also samples ComfyUI's torch VRAM free counter every polling
interval and records minimum sampled free VRAM. It does not use NVML because this
workstation currently has a known driver/userspace mismatch (`nvidia-smi` reports
NVML 580.173 while the loaded kernel module is 580.159.04).

### Required matrix

1. **Resolution sweep**: 0.1, 0.2, ..., 1.0 MP at 16:9, fixed 2 seconds and
   fixed seed. The first pass uses 1 step to map scaling economically; key points
   are then repeated at 4/8/20 steps.
2. **Aspect-ratio sweep**: 16:9, 4:3, 1:1, 3:4, and 9:16 at approximately
   0.4 MP, fixed 2 seconds and fixed seed. Dimensions stay on the 32-pixel grid.
3. **Duration sweep**: requested 1, 2, ..., 15 seconds at fixed 0.4 MP. Every
   duration is recorded as both requested and actual aligned duration. The first
   pass uses 1 step; key durations are repeated at 4/8/20 steps.
4. **Quality anchor**: 0.4 MP, 56 frames, 20 steps, fixed seed. Compare against
   any acceleration node with the exact same graph and inspect both streams.

The model's frame rule means a request is not an exact duration: 1 second becomes
39 frames / 1.625 seconds, 2 becomes 56 / 2.333 seconds, and 15 becomes 362 /
15.083 seconds. This is expected, not a benchmark bug.

Warm/cold handling: record first-run and subsequent-run separately when testing
cache behavior. For ordinary native scaling, report the wall clock as-is and do
not silently discard model-loading time. Never run two matrix jobs concurrently.

## Local V100 checkpoint: safe resolution pass (2026-08-10)

The first pass was intentionally paused for a coordinated FastLLM GPU handoff after
two successful points. It used the native API graph, fixed seed, 56 frames (2.33 s),
and one denoise step; `execution_wall_s` includes model staging, text encoding,
sampling, VAE decode, audio decode, and MP4 muxing.

| Target MP | Dimensions | Execution wall | Minimum sampled free VRAM | Result |
|---:|---:|---:|---:|---|
| 0.1 | 416x224 | 527.969 s | 1.90 GiB | success |
| 0.2 | 608x352 | 467.562 s | 1.71 GiB | success |
| 0.3 | 736x416 | 87.930 s | 24.71 GiB* | targeted interrupt |

`mp_0.3` was interrupted through `/interrupt` followed by deletion of only its
prompt ID when the shared GPU had to be handed to FastLLM. The harness wrote the
interrupted row and submitted no later cases. The result record is in
`video-gen/results/resolution_1step.json`; its successful rows reference the two
MP4s retained under the ComfyUI output directory. This is a checkpoint, not a
completed 0.1--1.0 MP scaling claim. The measured post-cleanup state was 32,012
MiB free in NVML (483 MiB used) and an empty ComfyUI queue.

\*The interrupted point had not reached the peak sampling phase, so its free-VRAM
sample is not comparable with the two completed points.

## Source ledger

- [MiniMax H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3)
- [MiniMax H3 official README/model architecture](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.md)
- [ComfyUI MiniMax H3 tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [Official T2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)
- [Awesome MiniMax H3 model/quant index](https://github.com/wildminder/awesome-minimax-H3)
- [Official V100 FP16 NaN issue](https://github.com/Comfy-Org/ComfyUI/issues/15262)
- [ComfyUI NVFP4 loader issue](https://github.com/Comfy-Org/ComfyUI/issues/15400)
- [Turbo LoRA custom node](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)
- [FirstBlockCache](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache)
- [Spectrum](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)
- [TE-Speed](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS)
- [H3 Cache](https://github.com/lihaoyun6/ComfyUI-MiniMaxH3-Cache)

Community star counts and README claims change over time; local result JSON is
the dated authority for this machine.
