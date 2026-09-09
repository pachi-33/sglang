#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yaozhenyang/dev/sglang-v100
RECORD="$REPO/experiments/expert_prefetch/records/prefetch-predictor/EXP-0003__20260909T115133+0800__v100-frozen-embeddedroutemlp-top-k-8-16-24-32"
EXP2="$REPO/experiments/expert_prefetch/records/prefetch-predictor/EXP-0002__20260909T103244+0800__v100-embeddedroutemlp-v0001-default-t8-k4-seed0"
VERSION="$REPO/experiments/expert_prefetch/algorithms/embedded-route-mlp/versions/v0001"
SPLITS="$REPO/experiments/expert_prefetch/records/trace-analysis/EXP-0001__20260908T170025+0800__v100-random-1024x256-expert-activation-dataset/artifacts/splits/request-seed0-96-16-16"
PYTHON=/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python
GPU=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96

cd "$REPO"
sha256sum "$EXP2/artifacts/best_model.pt" > "$RECORD/results/checkpoint_sha256.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used \
  --format=csv,noheader > "$RECORD/results/gpu_before.csv"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=python:. "$PYTHON" \
  experiments/expert_prefetch/evaluate_embedded_route_mlp_v0001.py \
  --version-dir "$VERSION" \
  --checkpoint "$EXP2/artifacts/best_model.pt" \
  --train-npz "$SPLITS/train.npz" \
  --validation-npz "$SPLITS/validation.npz" \
  --test-npz "$SPLITS/test.npz" \
  --candidate-counts 8 16 24 32 --batch-size 1024 --device cuda \
  --require-compute-capability 7.0 \
  --output-json "$RECORD/results/topk_metrics.json" \
  --predictions-npz "$RECORD/artifacts/top32_predictions.npz" \
  2>&1 | tee "$RECORD/logs/evaluate.log"

nvidia-smi --query-gpu=index,uuid,name,memory.used \
  --format=csv,noheader > "$RECORD/results/gpu_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader > "$RECORD/results/compute_processes_after.csv"
