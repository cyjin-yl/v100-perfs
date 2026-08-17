#!/usr/bin/env python3
"""生成 模型 × KV量化 × MTP × 算子/缓存策略 的扫描 profile。

一块 V100 只能串行跑, 所以每个组合 = 一次后端重启 = 一个 env 文件。
手写十几份 env 必然写错(今天已经因为挂错 profile 白测了一轮), 所以用表驱动生成。

用法:
  python sweep_profiles.py --list          # 只打印计划
  python sweep_profiles.py --write         # 写入 runtime/fastllm-native-profiles/sweep/
"""
from __future__ import annotations

import argparse
import os

ROOT = "/run/media/ezra/13D010B6FDBC1A06/1CatVLLM"
PERF = f"{ROOT}/v100-perfs"
OUT = f"{PERF}/runtime/fastllm-native-profiles/sweep"
# 官方模板(与 Qwen/Qwen3.8-27B 逐字一致, md5 519239a4908bb1f805bbce5fa8c8a242)
TEMPLATE = f"{ROOT}/models/_official_qwen38_refs/Qwen3.8-27B/chat_template.jinja"

CYBER_GGUF = f"{ROOT}/models/Qwen3.8-27B-Uncensored-Cyber-GGUF"

# key -> (gguf 路径, mmproj 路径 或 None, 权重 GiB(约), 是否解限版)
MODELS = {
    "n-q4kxl": (f"{ROOT}/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                f"{ROOT}/models/Qwen3.8-27B-UD-Q4_K_XL/mmproj-F16.gguf", 17.9, False),
    "n-q5km": (f"{ROOT}/models/Qwen3.8-27B-Q5KM/Qwen3.8-27B-Q5_K_M.gguf",
               f"{ROOT}/models/Qwen3.8-27B-UD-Q4_K_XL/mmproj-F16.gguf", 19.8, False),
    "n-q6k": (f"{ROOT}/models/Qwen3.8-27B-Q6K/Qwen3.8-27B-Q6_K.gguf",
              f"{ROOT}/models/Qwen3.8-27B-UD-Q4_K_XL/mmproj-F16.gguf", 22.9, False),
    "n-awq": (f"{ROOT}/models/Qwen3.8-27B-W4A16-AWQ", None, 27.0, False),
    "c-q4km": (f"{CYBER_GGUF}/Qwen3.8-27B-Uncensored-Cyber-Q4_K_M-plus-mtp.gguf",
               f"{CYBER_GGUF}/mmproj-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", 19.4, True),
    "c-q5km": (f"{CYBER_GGUF}/Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-plus-mtp.gguf",
               f"{CYBER_GGUF}/mmproj-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", 21.7, True),
    "c-q6k": (f"{CYBER_GGUF}/Qwen3.8-27B-Uncensored-Cyber-Q6_K-plus-mtp.gguf",
              f"{CYBER_GGUF}/mmproj-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", 24.2, True),
    "c-awq": (f"{ROOT}/models/Qwen3.8-27B-Uncensored-Cyber-W4A16-AWQ", None, 27.0, True),
}

# 缓存策略: fastllm 里三级缓存与代价模型早就存在, 但默认值是给"少数超长会话"设计的:
#   CPU_TIER 默认 false(RAM 层根本没启用), MIN_TOKENS 默认 65536,
#   DISK_MIN_TOKENS 默认 65536 —— agent 每轮前缀才 20~30K, 永远够不着门槛,
#   所以实测 L2disk=0.0MB / hits=0, 上一轮前缀被逐出后只能全量重 prefill。
CACHE_TUNED = {
    "FASTLLM_PREFIX_CACHE_CPU_TIER": "1",
    "FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES": str(16 * 1024**3),
    "FASTLLM_PREFIX_CACHE_MIN_HITS": "1",
    "FASTLLM_PREFIX_CACHE_MIN_TOKENS": "4096",
    "FASTLLM_PREFIX_CACHE_DISK_MIN_HITS": "1",
    "FASTLLM_PREFIX_CACHE_DISK_MIN_TOKENS": "4096",
    "FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES": str(200 * 1024**3),
    "FASTLLM_PREFIX_CACHE_ZSTD": "1",
    "FASTLLM_PREFIX_CACHE_ZSTD_LEVEL": "3",
    "FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES": "8",
}
CACHE_BASE = {
    "FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES": str(2 * 1024**3),
    "FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES": "16",
    "FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES": str(3 * 1024**3),
}
SM70_ON = {
    "FASTLLM_CUDA_SM70_FLASH_ATTN": "1",
    "FASTLLM_CUDA_SM70_PAGED_XQA": "1",
}
SM70_OFF = {
    "FASTLLM_CUDA_SM70_FLASH_ATTN": "0",
    "FASTLLM_CUDA_SM70_PAGED_XQA": "0",
}

# id, model, kv_dtype, mtp(0/2), ctx, batch, cache, sm70, 说明
MATRIX = [
    ("n1", "n-q4kxl", "turbo4", 2, 262144, 2, "base", "off", "普通版基线(当前生产)"),
    ("n2", "n-q4kxl", "turbo3", 2, 262144, 2, "base", "off", "KV turbo3 对照"),
    ("n3", "n-q4kxl", "fp8_e4m3", 2, 262144, 2, "base", "off", "KV fp8 对照"),
    ("n4", "n-q4kxl", "turbo4", 0, 262144, 2, "base", "off", "MTP 关(量化 MTP 收益)"),
    ("n5", "n-q4kxl", "turbo4", 2, 262144, 2, "tuned", "off", "缓存策略调优"),
    ("n6", "n-q4kxl", "turbo4", 2, 262144, 2, "tuned", "on", "缓存调优 + SM70 算子"),
    ("n7", "n-q5km", "turbo4", 2, 262144, 2, "tuned", "off", "普通版 Q5(智力更高?)"),
    ("n8", "n-q6k", "turbo4", 2, 131072, 2, "tuned", "off", "普通版 Q6(显存紧, 降 ctx)"),
    ("n9", "n-awq", "turbo3", 2, 262144, 2, "tuned", "off", "普通版 AWQ W4A16"),
    ("c1", "c-q5km", "turbo4", 2, 262144, 2, "tuned", "off", "解限版 Q5(主推候选)"),
    ("c2", "c-q5km", "turbo3", 2, 262144, 2, "tuned", "off", "解限版 Q5 + turbo3"),
    ("c3", "c-q4km", "turbo4", 2, 262144, 2, "tuned", "off", "解限版 Q4_K_M(已知该档有隐患)"),
    ("c4", "c-q6k", "turbo4", 2, 131072, 2, "tuned", "off", "解限版 Q6(显存紧, 降 ctx)"),
    ("c5", "c-awq", "turbo3", 2, 262144, 2, "tuned", "off", "解限版 AWQ W4A16(27G 权重)"),
    ("c6", "c-q5km", "turbo4", 2, 262144, 2, "tuned", "on", "解限版 Q5 + SM70 算子"),
]


def render(entry) -> tuple[str, str]:
    pid, mkey, kv, mtp, ctx, batch, cache, sm70, note = entry
    path, mmproj, gb, uncensored = MODELS[mkey]
    tag = f"q38-{pid}-{mkey}-{kv}-mtp{mtp}"
    if cache == "tuned":
        tag += "-cachetuned"
    if sm70 == "on":
        tag += "-sm70"
    cmd = [f"{ROOT}/fastllm/build-rw/apiserver", "--path", path]
    if mmproj:
        cmd += ["--mmproj", mmproj]
    cmd += ["--threads", "2", "--atype", "float16",
            "--kv_cache_dtype", kv, "--batch", str(batch),
            "--tokens", str(ctx), "--chunked_prefill_size", "512",
            "--default_max_tokens", "16384",
            "--model_name", "qwen3.8-fastllm", "--port", "8002",
            "--device", "cuda"]
    env = {
        "PROJECT_DIR": PERF,
        "PROXY_HOST": "0.0.0.0",
        "PROXY_PORT": "8000",
        "PROXY_LOG_FILE": f"{PERF}/runtime/fastllm-native-profiles/logs/proxy-{tag}.log",
        "FASTLLM_BACKEND_URL": "http://127.0.0.1:8002",
        "FASTLLM_CHAT_TEMPLATE": TEMPLATE,
        "FASTLLM_MODEL_SLUG": "qwen3.8-fastllm",
        "FASTLLM_PUBLIC_ALIASES": "qwen3.8-27b,qwen3.8-27b-q4kxl,qwen3.8-27b-heretic",
        "FASTLLM_OWNED": "1",
        "FASTLLM_BACKEND_COMMAND": "'" + " ".join(cmd) + "'",
        "FASTLLM_BACKEND_CWD": PERF,
        "FASTLLM_BACKEND_LOG": f"{PERF}/runtime/fastllm-native-profiles/logs/backend-{tag}.log",
        "FASTLLM_START_TIMEOUT": "1800",
        "FASTLLM_STOP_TIMEOUT": "30",
        "FASTLLM_IDLE_TIMEOUT": "0",
        "FASTLLM_LIFECYCLE_INTERVAL": "5",
        "FASTLLM_VRAM_MIN_FREE_GIB": "0.5",
        "FASTLLM_VRAM_RESUME_FREE_GIB": "1.0",
        "QUEUE_TIMEOUT": "1800",
        # 实验环境禁止静默云端 fallback: 路由错配必须显性失败
        "FALLBACK_ENABLED": "0",
        "FASTLLM_PREFIX_CACHE_PERSIST": "1",
        "FASTLLM_PREFIX_CACHE_DISK_DIR": f"{PERF}/runtime/prefix-cache-{tag}",
        "FASTLLM_PREFIX_CACHE_PERSIST_KEY": tag,
        "FASTLLM_PREFIX_CACHE_DISK_MIN_FREE_BYTES": str(8 * 1024**3),
        "FASTLLM_CPU_REQUEST_SWAP": "1",
        "FASTLLM_CPU_REQUEST_SWAP_ZSTD": "1",
        "FASTLLM_CPU_REQUEST_SWAP_ZSTD_LEVEL": "3",
        "FASTLLM_CPU_REQUEST_SWAP_DISK_DIR": f"{PERF}/runtime/swap-{tag}",
        "FASTLLM_CPU_REQUEST_SWAP_DISK_MAX_BYTES": str(4 * 1024**3),
        "FASTLLM_CPU_REQUEST_SWAP_DISK_MIN_BYTES": "67108864",
        "FASTLLM_CPU_REQUEST_SWAP_DISK_MIN_FREE_BYTES": str(4 * 1024**3),
        "FASTLLM_QWEN35_ENABLE_MTP": str(mtp),
        "FASTLLM_QWEN35_INTERLEAVE_LONG_PREFILL": "1",
        "FASTLLM_HOST_SUSPEND_CACHE": "1",
        "FASTLLM_HOST_SUSPEND_CACHE_MAX_BYTES": str(16 * 1024**3),
        "FASTLLM_HOST_SUSPEND_MIN_FREE_BYTES": str(4 * 1024**3),
        "FASTLLM_SKIP_WARMUP": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    # KV dtype 的 CLI 参数不够: turbo3/turbo4 还要各自的 env 开关, 否则后端启动即
    # 抛 "Qwen3.5 turboN KV cache requires FASTLLM_QWEN35_TURBON_KV=1."
    if kv == "turbo4":
        env["FASTLLM_QWEN35_TURBO4_KV"] = "1"
    elif kv == "turbo3":
        env["FASTLLM_QWEN35_TURBO3_KV"] = "1"
    env.update(CACHE_TUNED if cache == "tuned" else CACHE_BASE)
    env.update(SM70_ON if sm70 == "on" else SM70_OFF)
    header = (f"# sweep {pid}: {note}\n"
              f"# model={mkey} weights~{gb}GiB kv={kv} mtp={mtp} ctx={ctx} "
              f"batch={batch} cache={cache} sm70={sm70} "
              f"uncensored={'yes' if uncensored else 'no'}\n")
    body = "\n".join(f"{k}={v}" for k, v in env.items())
    return tag, header + body + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", default="", help="逗号分隔的 id 过滤, 如 n1,n5,c1")
    args = ap.parse_args()
    keep = [s for s in args.only.split(",") if s]
    if args.write:
        os.makedirs(OUT, exist_ok=True)
    for entry in MATRIX:
        if keep and entry[0] not in keep:
            continue
        tag, text = render(entry)
        path = os.path.join(OUT, f"{tag}.env")
        model_path = MODELS[entry[1]][0]
        exists = os.path.exists(model_path)
        print(f"{entry[0]:>3}  {tag:<52} model_ready={'Y' if exists else 'N'}  {entry[8]}")
        if args.write:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
    if args.write:
        print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
