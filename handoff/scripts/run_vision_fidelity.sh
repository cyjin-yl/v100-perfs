#!/usr/bin/env bash
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
A=$(grep -m1 -oE '^AUTH_TOKEN=.*' "$ROOT/.env" | cut -d= -f2- | tr -d "\"' ")
echo "=== 带图 ==="
python3 /home/ezra/projects/EzraVastLLM/scripts/copy_fidelity_vision.py \
  --token "$A" --json "$ROOT/imatrix/logs/vision-fidelity-${TAG:-base}.json"
echo
echo "=== 纯文本对照(同一批用例) ==="
python3 /home/ezra/projects/EzraVastLLM/scripts/copy_fidelity_vision.py \
  --token "$A" --no-image --json "$ROOT/imatrix/logs/text-fidelity-${TAG:-base}.json"
