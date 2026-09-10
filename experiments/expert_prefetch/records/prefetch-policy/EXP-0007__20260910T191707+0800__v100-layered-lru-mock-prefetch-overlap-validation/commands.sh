#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/home/yaozhenyang/dev/sglang-v100/experiments/expert_prefetch/records/prefetch-policy/EXP-0007__20260910T191707+0800__v100-layered-lru-mock-prefetch-overlap-validation
PY=/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python
MODEL=/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16
PACK=/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json
SHAREGPT=/home/yaozhenyang/dev/sglang-v100/ShareGPT_V3_unfiltered_cleaned_split.json
V100=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96

CUDA_VISIBLE_DEVICES="$V100" PYTHONPATH=python:. "$PY" \
  -m sglang.srt.layers.qwen3_5.single_gpu_api \
  --model-dir "$MODEL" --served-model-name "$MODEL" \
  --expert-pack-manifest "$PACK" \
  --expert-source pinned-memory --expert-cache-policy layer-lru \
  --expert-cache-ratio 0.40 --expert-cache-mib 7168 \
  --expert-stage-slots 16 --expert-io-workers 2 --capacity 2048 \
  --enable-mock-expert-prefetch \
  --mock-prefetch-log "$EXP_DIR/artifacts/prefetch_timing.jsonl" \
  --stats-path "$EXP_DIR/results/store_stats.json" \
  --host 127.0.0.1 --port 8818

PYTHONPATH=python:. "$PY" -m sglang.bench_serving \
  --dataset-name random --backend sglang --model "$MODEL" \
  --dataset-path "$SHAREGPT" --host 127.0.0.1 --port 8818 \
  --max-concurrency 1 --request-rate inf --random-input-len 1024 \
  --random-output-len 16 --num-prompts 1 --random-range-ratio 1 \
  --random-input-token-ids --seed 0 --mock-expert-prefetch \
  --mock-prefetch-recall 0 --mock-prefetch-top-k 0 \
  --mock-prefetch-lead-layers 2 --mock-prefetch-seed 0 \
  --output-file "$EXP_DIR/results/control_bench.jsonl"

PYTHONPATH=python:. "$PY" -m sglang.bench_serving \
  --dataset-name random --backend sglang --model "$MODEL" \
  --dataset-path "$SHAREGPT" --host 127.0.0.1 --port 8818 \
  --max-concurrency 1 --request-rate inf --random-input-len 1024 \
  --random-output-len 16 --num-prompts 1 --random-range-ratio 1 \
  --random-input-token-ids --seed 0 --mock-expert-prefetch \
  --mock-prefetch-recall 0.5 --mock-prefetch-top-k 8 \
  --mock-prefetch-lead-layers 2 --mock-prefetch-seed 0 \
  --output-file "$EXP_DIR/results/treatment_bench.jsonl"
