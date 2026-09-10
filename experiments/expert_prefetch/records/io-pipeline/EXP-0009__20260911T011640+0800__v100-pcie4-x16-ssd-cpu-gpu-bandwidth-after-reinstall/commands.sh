#!/usr/bin/env bash
set -euo pipefail

# Repository root: /home/yaozhenyang/dev/sglang-v100
RECORD="experiments/expert_prefetch/records/io-pipeline/EXP-0009__20260911T011640+0800__v100-pcie4-x16-ssd-cpu-gpu-bandwidth-after-reinstall"
PYTHON="/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python"
GPU_UUID="GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"
MANIFEST="/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json"

nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,pstate,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv > "$RECORD/logs/gpu_before.csv"
nvidia-smi topo -m > "$RECORD/logs/topology.txt"
lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN > "$RECORD/logs/storage.txt"
findmnt -T "$RECORD" -o SOURCE,FSTYPE,TARGET,OPTIONS > "$RECORD/logs/filesystem.txt"
lscpu > "$RECORD/logs/lscpu.txt"

dd if=/dev/zero of="$RECORD/artifacts/ssd-test-8g.bin" bs=64M count=128 oflag=direct conv=fsync status=progress
stat -c 'logical_bytes=%s allocated_blocks_512=%b' "$RECORD/artifacts/ssd-test-8g.bin" > "$RECORD/logs/test_file_allocation.txt"

CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" experiments/expert_prefetch/benchmark_ssd_cpu_gpu.py \
  --file "$RECORD/artifacts/ssd-test-8g.bin" \
  --expected-gpu-uuid "$GPU_UUID" \
  --output-raw "$RECORD/artifacts/bandwidth_samples.npz" \
  --output-summary "$RECORD/results/bandwidth_summary.json" \
  --ssd-chunk-mib 16 --h2d-buffer-mib 256 --pipeline-buffers 3 \
  --rounds 3 --bytes-per-round-gib 8 \
  > "$RECORD/logs/benchmark.stdout.log" 2> "$RECORD/logs/benchmark.stderr.log"

# Rerun after moving peak-memory sampling before the untimed correctness
# reduction. The first-run raw data and summary remain preserved.
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" experiments/expert_prefetch/benchmark_ssd_cpu_gpu.py \
  --file "$RECORD/artifacts/ssd-test-8g.bin" \
  --expected-gpu-uuid "$GPU_UUID" \
  --output-raw "$RECORD/artifacts/bandwidth_samples_rerun.npz" \
  --output-summary "$RECORD/results/bandwidth_summary_rerun.json" \
  --ssd-chunk-mib 16 --h2d-buffer-mib 256 --pipeline-buffers 3 \
  --rounds 3 --bytes-per-round-gib 8

# Same payload/components/candidate counts as EXP-0006 for an apples-to-apples
# pre/post reinstall H2D comparison.
CUDA_VISIBLE_DEVICES="$GPU_UUID" PYTHONPATH=python:. "$PYTHON" experiments/expert_prefetch/benchmark_pinned_h2d.py \
  --manifest-path "$MANIFEST" \
  --output-npz "$RECORD/artifacts/expertpack_h2d_samples.npz" \
  --output-summary "$RECORD/results/expertpack_h2d_summary.json" \
  --expected-gpu-uuid "$GPU_UUID" \
  --candidates 1 2 4 8 16 24 32 \
  --warmup 20 --samples 200

rm -- "$RECORD/artifacts/ssd-test-8g.bin"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$RECORD/logs/compute_processes_after.csv"
nvidia-smi --query-gpu=index,name,uuid,memory.used,pstate,pcie.link.gen.current,pcie.link.width.current --format=csv > "$RECORD/logs/gpu_after.csv"
