#!/usr/bin/env python3
"""Measure prefill and decode throughput of the current production FastLLM
backend (IQ6 abliterated 262K turbo3 MTP2) and record a structured result."""
import json
import pathlib
import re
import time
import urllib.request

ENV = pathlib.Path('/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.env').read_text()
AUTH = re.search(r'^\s*AUTH_TOKEN\s*=\s*["\']?([^"\'\n]+)', ENV, re.M).group(1)
TOKEN = open('/proc/1130557/environ', 'rb').read().split(
    b'FASTLLM_PREFIX_CACHE_CONTROL_TOKEN=')[1].split(b'\0')[0].decode()

URL = 'http://127.0.0.1:8002/v1/chat/completions'

def post(body, timeout=1800):
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0

# --- 1. decode: short prompt, long generation -----------------------------
prompt_short = '写一段关于机器学习的科普文字。'
out, wall = post({
    'model': 'qwen3.6-fastllm', 'max_tokens': 512, 'temperature': 0,
    'stream': False, 'raw_prompt': True,
    'prompt': prompt_short})
usage = out['usage']
completion = usage.get('completion_tokens', 0)
decode_tok_s = completion / max(wall - 0.2, 0.1)
print(f'decode: {completion} tokens in {wall:.2f}s -> {decode_tok_s:.1f} tok/s '
      f'(prompt_tokens={usage.get("prompt_tokens")})')

# --- 2. prefill: 32K-token prompt, tiny generation ------------------------
unit = ' a'
prefix = '<|im_start|>user\nresident-bench\nContinue the text.\n'
suffix = '<|im_end|>\n<|im_start|>assistant\n'
count = 32000
prompt = prefix + unit * count + suffix
out, wall = post({
    'model': 'qwen3.6-fastllm', 'max_tokens': 8, 'temperature': 0,
    'stream': False, 'raw_prompt': True, 'prompt': prompt})
usage = out['usage']
prompt_tokens = usage.get('prompt_tokens', 0)
evaluated = usage.get('prompt_tokens_evaluated', prompt_tokens)
prefill_tok_s = evaluated / max(wall - 8 / max(decode_tok_s, 1), 0.1)
print(f'prefill: {evaluated} tokens in {wall:.2f}s -> {prefill_tok_s:.1f} tok/s '
      f'(cache_hit={prompt_tokens - evaluated})')

result = {
    'schema_version': 1,
    'benchmark': 'fastllm_iq6_prefill_decode_throughput',
    'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'hardware': {'gpu': 'Tesla V100-PCIE-32GB', 'memory_total_mib': 32768},
    'runtime': {
        'backend': 'FastLLM native apiserver',
        'model': 'Huihui-ThinkingCap-Qwen3.6-27B-abliterated i1-Q6_K + MTP mmproj',
        'binary': '/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build/apiserver',
        'activation_dtype': 'float16', 'kv_cache_dtype': 'turbo3',
        'token_pool': 262144, 'mtp': 2, 'chunked_prefill_size': 512,
    },
    'decode': {
        'max_tokens': 512, 'completion_tokens': completion,
        'tokens_per_second': round(decode_tok_s, 2),
    },
    'prefill': {
        'prompt_tokens': prompt_tokens, 'prompt_tokens_evaluated': evaluated,
        'tokens_per_second': round(prefill_tok_s, 2),
    },
}
print(json.dumps(result, ensure_ascii=False))
