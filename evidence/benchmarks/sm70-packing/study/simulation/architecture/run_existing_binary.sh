#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
./build/accel-sim.out \
  -trace ./official-lud64/kernelslist.g \
  -config ./configs/qv100-upstream.config \
  -config ./configs/qv100-upstream-trace.config \
  > logs/official-lud64-qv100.log 2>&1
python3 make_synthetic_traces.py > synthetic-generation.json
python3 run_synthetic.py > synthetic-run-console.log
python3 summarize_evidence.py
