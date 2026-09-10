#!/usr/bin/env bash
set -euo pipefail

cd /home/yaozhenyang/dev/sglang-v100

EXP_DIR=/home/yaozhenyang/dev/sglang-v100/experiments/expert_prefetch/records/io-pipeline/EXP-0006__20260910T144252+0800__v100-resident-pp-decode-window-and-pinned-h2d-capacity
PYTHON_BIN=/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python
MODEL_DIR=/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16
MANIFEST=/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json
DATASET=/home/yaozhenyang/dev/sglang-v100/ShareGPT_V3_unfiltered_cleaned_split.json
ORACLE=/home/yaozhenyang/dev/sglang-v100/experiments/expert_prefetch/records/trace-analysis/EXP-0001__20260908T170025+0800__v100-random-1024x256-expert-activation-dataset/artifacts/EXP-0001.expert-activation-dataset.npz
V100_UUID=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96
RTX4070_UUID=GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4

nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,temperature.gpu,pstate --format=csv > "$EXP_DIR/logs/device_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$EXP_DIR/logs/processes_before.csv"

PYTHONPATH=python:. "$PYTHON_BIN" experiments/expert_prefetch/profile_resident_decode_window.py \
  --model-dir "$MODEL_DIR" \
  --dataset-path "$DATASET" \
  --oracle-npz "$ORACLE" \
  --output-jsonl "$EXP_DIR/artifacts/decode_timing.jsonl" \
  --output-summary "$EXP_DIR/results/pp_run_summary.json" \
  --front-uuid "$RTX4070_UUID" \
  --back-uuid "$V100_UUID" \
  --split-layer 17 \
  --capacity 2048 \
  --seed 0 \
  --request-index 0 \
  --input-tokens 1024 \
  --oracle-tokens 256 \
  --warmup-decode-steps 16 \
  --decode-steps 256 \
  --expected-prompt-sha256 4e7bf8d35d419417f9274499ef041fe6cc5de9210c9e1ffc6f05bc8f9af823e8 \
  > "$EXP_DIR/logs/pp_profile.log" 2>&1

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$EXP_DIR/logs/processes_after_pp.csv"

CUDA_VISIBLE_DEVICES="$V100_UUID" PYTHONPATH=python:. "$PYTHON_BIN" experiments/expert_prefetch/benchmark_pinned_h2d.py \
  --manifest-path "$MANIFEST" \
  --output-npz "$EXP_DIR/artifacts/h2d_samples.npz" \
  --output-summary "$EXP_DIR/results/h2d_run_summary.json" \
  --expected-gpu-uuid "$V100_UUID" \
  --candidates 1 2 4 8 16 24 32 \
  --warmup 20 \
  --samples 200 \
  > "$EXP_DIR/logs/h2d_benchmark.log" 2>&1

PYTHONPATH=python:. "$PYTHON_BIN" experiments/expert_prefetch/analyze_decode_window.py \
  --decode-jsonl "$EXP_DIR/artifacts/decode_timing.jsonl" \
  --pp-summary "$EXP_DIR/results/pp_run_summary.json" \
  --h2d-npz "$EXP_DIR/artifacts/h2d_samples.npz" \
  --layer-summary "$EXP_DIR/results/layer_timing_summary.csv" \
  --window-summary "$EXP_DIR/results/router_window_summary.csv" \
  --h2d-summary "$EXP_DIR/results/h2d_summary.csv" \
  --capacity-summary "$EXP_DIR/results/prefetch_capacity.csv" \
  --experiment-summary "$EXP_DIR/results/experiment_summary.json" \
  --layer-start 17 \
  --layer-end 40 \
  --expected-steps 256 \
  --expected-h2d-samples 200 \
  > "$EXP_DIR/logs/analysis.log" 2>&1

nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,temperature.gpu,pstate --format=csv > "$EXP_DIR/logs/device_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$EXP_DIR/logs/processes_after.csv"
