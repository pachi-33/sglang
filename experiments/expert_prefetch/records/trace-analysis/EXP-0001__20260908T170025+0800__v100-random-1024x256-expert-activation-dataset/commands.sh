#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yaozhenyang/dev/sglang-v100
RECORD=/home/yaozhenyang/dev/sglang-v100/experiments/expert_prefetch/records/trace-analysis/EXP-0001__20260908T170025+0800__v100-random-1024x256-expert-activation-dataset
PYTHON=/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python
MODEL=/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16
PACK=/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json
DATASET="$REPO/ShareGPT_V3_unfiltered_cleaned_split.json"
GPU=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96

# Terminal 1: API service.
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=python:. "$PYTHON" \
  -m sglang.srt.layers.qwen3_5.single_gpu_api \
  --model-dir "$MODEL" --served-model-name "$MODEL" \
  --expert-pack-manifest "$PACK" \
  --expert-cache-mib 7168 --expert-stage-slots 16 --expert-io-workers 2 \
  --capacity 2048 --host 127.0.0.1 --port 8818 \
  --stats-path "$RECORD/results/store_stats_final.json" \
  --expert-trace-dir "$RECORD/artifacts/traces" \
  2>&1 | tee "$RECORD/logs/server.log"

# Terminal 2: benchmark after /health reports ok.
cd "$REPO"
PYTHONPATH=python:. "$PYTHON" -m sglang.bench_serving \
  --dataset-name random --backend sglang --model "$MODEL" \
  --dataset-path "$DATASET" --host 127.0.0.1 --port 8818 \
  --max-concurrency 1 --random-input-len 1024 --random-output-len 256 \
  --num-prompts 128 --random-range-ratio 1 --request-rate inf --seed 0 \
  --expert-trace --random-input-token-ids \
  --output-file "$RECORD/results/bench.jsonl" \
  2>&1 | tee "$RECORD/logs/bench.log"

# After stopping the service, validate every trace and build the dataset index.
cd "$REPO"
PYTHONPATH=python:. "$PYTHON" experiments/expert_prefetch/validate_trace_dataset.py \
  --bench-jsonl "$RECORD/results/bench.jsonl" \
  --trace-dir "$RECORD/artifacts/traces" \
  --output-manifest "$RECORD/results/dataset_manifest.jsonl" \
  --output-summary "$RECORD/results/dataset_summary.json" \
  --expected-count 128 --prompt-tokens 1024 --completion-tokens 256 \
  --gpu-uuid "$GPU" --source-dataset "$DATASET"

# Auxiliary cold/warm characterization, terminal 1. This run deliberately omits
# --expert-trace-dir, and its two requests are not dataset members.
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=python:. "$PYTHON" \
  -m sglang.srt.layers.qwen3_5.single_gpu_api \
  --model-dir "$MODEL" --served-model-name "$MODEL" \
  --expert-pack-manifest "$PACK" \
  --expert-cache-mib 7168 --expert-stage-slots 16 --expert-io-workers 2 \
  --capacity 2048 --host 127.0.0.1 --port 8818 \
  --stats-path "$RECORD/results/cold_warm_store_stats.json" \
  2>&1 | tee "$RECORD/logs/cold_warm_server.log"

# Auxiliary cold/warm characterization, terminal 2. Stop terminal 1 with
# SIGINT after this client succeeds.
cd "$REPO"
PYTHONPATH=python:. "$PYTHON" "$RECORD/cold_warm_client.py" \
  2>&1 | tee "$RECORD/logs/cold_warm_client.log"
