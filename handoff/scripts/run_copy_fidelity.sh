#!/usr/bin/env bash
# 单独跑抄写保真, 不与任何回滚逻辑耦合。
# 生产 batch=1 且有真实 agent 在跑时, 探针会排在几十个请求后面, 所以要放 tmux 里。
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
A=$(grep -m1 -oE '^AUTH_TOKEN=.*' "$ROOT/.env" | cut -d= -f2- | tr -d "\"' ")
exec python3 /home/ezra/projects/EzraVastLLM/scripts/copy_fidelity_http.py \
  --token "$A" --json "$ROOT/imatrix/logs/copy-fidelity-fused-xqa.json"
