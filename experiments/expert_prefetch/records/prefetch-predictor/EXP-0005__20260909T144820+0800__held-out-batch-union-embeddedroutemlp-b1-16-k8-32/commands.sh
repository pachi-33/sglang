#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yaozhenyang/dev/sglang-v100
RECORD="$REPO/experiments/expert_prefetch/records/prefetch-predictor/EXP-0005__20260909T144820+0800__held-out-batch-union-embeddedroutemlp-b1-16-k8-32"
EXP3="$REPO/experiments/expert_prefetch/records/prefetch-predictor/EXP-0003__20260909T115133+0800__v100-frozen-embeddedroutemlp-top-k-8-16-24-32"
TEST="$REPO/experiments/expert_prefetch/records/trace-analysis/EXP-0001__20260908T170025+0800__v100-random-1024x256-expert-activation-dataset/artifacts/splits/request-seed0-96-16-16/test.npz"
PYTHON=/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python

cd "$REPO"
PYTHONPATH=python:. "$PYTHON" "$RECORD/evaluate_batch_union.py" \
  --test-npz "$TEST" \
  --predictions-npz "$EXP3/artifacts/top32_predictions.npz" \
  --output-results "$RECORD/results" \
  --output-artifacts "$RECORD/artifacts" \
  --batch-sizes 1 2 4 8 16 \
  --candidate-counts 8 16 24 32 \
  --history-tokens 8 --subset-chunk-size 16 \
  2>&1 | tee "$RECORD/logs/evaluate.log"

# Export the authoritative metric using the clarified notation:
# M = correctly predicted unique experts, N = actual unique experts.
"$PYTHON" "$RECORD/export_batch_recall.py" \
  --source "$RECORD/results/batch_union_summary.json" \
  --output-json "$RECORD/results/batch_recall_summary.json" \
  --output-csv "$RECORD/results/batch_recall.csv" \
  2>&1 | tee "$RECORD/logs/export_batch_recall.log"
