#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yaozhenyang/dev/sglang-v100
RECORD="$REPO/experiments/expert_prefetch/records/trace-analysis/EXP-0004__20260909T143537+0800__random-k-cross-request-unique-experts-per-layer"
DATASET="$REPO/experiments/expert_prefetch/records/trace-analysis/EXP-0001__20260908T170025+0800__v100-random-1024x256-expert-activation-dataset/artifacts/EXP-0001.expert-activation-dataset.npz"
PYTHON=/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python

cd "$REPO"
"$PYTHON" "$RECORD/analyze_unique_experts.py" \
  --dataset "$DATASET" \
  --output-dir "$RECORD/results" \
  --k-values 1 2 4 8 16 32 64 128 \
  --trials 500 \
  --seed 0 \
  2>&1 | tee "$RECORD/logs/analysis.log"
