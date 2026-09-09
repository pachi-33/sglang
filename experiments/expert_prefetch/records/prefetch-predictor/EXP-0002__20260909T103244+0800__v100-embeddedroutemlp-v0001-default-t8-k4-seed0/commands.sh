#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yaozhenyang/dev/sglang-v100
RECORD="$REPO/experiments/expert_prefetch/records/prefetch-predictor/EXP-0002__20260909T103244+0800__v100-embeddedroutemlp-v0001-default-t8-k4-seed0"
VERSION="$REPO/experiments/expert_prefetch/algorithms/embedded-route-mlp/versions/v0001"
SPLITS="$REPO/experiments/expert_prefetch/records/trace-analysis/EXP-0001__20260908T170025+0800__v100-random-1024x256-expert-activation-dataset/artifacts/splits/request-seed0-96-16-16"
PYTHON=/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python
GPU=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96

cd "$REPO"
nvidia-smi --query-gpu=index,uuid,name,memory.used \
  --format=csv,noheader > "$RECORD/results/gpu_before.csv"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=python:. "$PYTHON" "$VERSION/train.py" \
  --train-npz "$SPLITS/train.npz" \
  --validation-npz "$SPLITS/validation.npz" \
  --test-npz "$SPLITS/test.npz" \
  --results-json "$RECORD/results/predictor_metrics.json" \
  --epoch-log-jsonl "$RECORD/results/epochs.jsonl" \
  --checkpoint "$RECORD/artifacts/best_model.pt" \
  --history-tokens 8 --current-previous-layers 4 --lead-layers 0 \
  --route-embedding-dim 32 --hidden-dim 512 --dropout 0.1 \
  --batch-size 512 --eval-batch-size 1024 \
  --max-epochs 20 --early-stopping-patience 3 \
  --learning-rate 0.001 --weight-decay 0.0001 \
  --gradient-clip-norm 1.0 --seed 0 --device cuda \
  --require-compute-capability 7.0 \
  2>&1 | tee "$RECORD/logs/train.log"

nvidia-smi --query-gpu=index,uuid,name,memory.used \
  --format=csv,noheader > "$RECORD/results/gpu_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader > "$RECORD/results/compute_processes_after.csv"
