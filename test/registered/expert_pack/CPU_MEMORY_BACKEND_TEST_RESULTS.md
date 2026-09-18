# CPU-memory expert backend test results

This file records manual validation of the CPU-memory expert offloader
implementation. Each section states the scope of the corresponding run; the
targeted unit-test section by itself is not an end-to-end performance claim.

## 2026-09-18 targeted validation

- Start time: 2026-09-18 13:43:19 +0800 (Asia/Singapore)
- Repository: `/home/yaozhenyang/dev/sglang`
- Branch: `feat/expertpack/cpu-memory-backend`
- Base commit: `4b186cfea59371cc8ec38597f750a5597f7147a0`
- Worktree state: uncommitted implementation and tests
- SHA-256 over the 14 implementation/test files under review, in the order
  listed by the command below:
  `4f622b3b872882e9fc773c901e5d8be70001535b457b791ecf1da0fa095b8060`

### Environment

- Conda environment: `sgl-main`
- Python: 3.12.0
- PyTorch: 2.13.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- GPU: NVIDIA GeForce RTX 4070 SUPER, compute capability 8.9
- GPU UUID: `GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4`
- Driver: 615.71.09
- GPU memory: 12282 MiB
- `pytest` was not installed in this environment. The files use
  `unittest`, so they were executed directly.
- `SGLANG_CACHE_DIR` was redirected to writable `/tmp/codex-sglang-cache`
  to allow FlashInfer import-time logging in the restricted runner.

### Commands

```bash
env SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  test/registered/unit/layers/moe/test_expert_offload.py -v

env SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  test/registered/unit/layers/moe/test_cpu_memory_expert_backend.py -v

env SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  test/registered/expert_pack/test_cpu_memory_backend.py -v
```

Code fingerprint command:

```bash
sha256sum \
  python/sglang/srt/arg_groups/expert_pack_hook.py \
  python/sglang/srt/layers/attention/linear/gdn_backend.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py \
  python/sglang/srt/layers/moe/cpu_memory_expert_backend.py \
  python/sglang/srt/layers/moe/expert_offload.py \
  python/sglang/srt/layers/quantization/awq/awq.py \
  python/sglang/srt/layers/quantization/base_config.py \
  python/sglang/srt/layers/quantization/unquant.py \
  python/sglang/srt/model_loader/expert_pack_loader.py \
  python/sglang/srt/model_loader/loader.py \
  python/sglang/srt/server_args.py \
  test/registered/expert_pack/test_cpu_memory_backend.py \
  test/registered/unit/layers/moe/test_cpu_memory_expert_backend.py \
  test/registered/unit/layers/moe/test_expert_offload.py \
  | sha256sum
```

### Results

| Test file | Passed | Failed | Skipped | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `test_expert_offload.py` | 26 | 0 | 0 | 0.106 s |
| `test_cpu_memory_expert_backend.py` | 11 | 0 | 0 | 0.158 s |
| `expert_pack/test_cpu_memory_backend.py` | 9 | 0 | 0 | 0.003 s |
| **Total** | **46** | **0** | **0** | **0.267 s** |

The GPU-dependent tests that passed were:

- `test_cuda_cache_copy_and_remap_smoke`
- `test_cuda_staging_ring_uses_pinned_memory`
- `test_cuda_unquantized_final_layout_cache_matches_baseline_across_microbatches`
- `test_cuda_awq_final_layout_payload_is_bit_exact_in_compact_slots`

The test GPU had no remaining compute process after the run.

### What this run establishes

- Capability declarations, coverage tracking, wrapper delegation, runtime
  layer views, logical-to-physical expert remapping, and lease release paths
  pass their targeted tests.
- Fake-event tests cover staging reuse, cache admission/eviction, active lease
  protection, rollback, pool poisoning, budget allocation, timing counters,
  and stats flushing.
- Real CUDA tests cover pinned staging, asynchronous cache copies, compact slot
  remapping, unquantized numerical parity for the toy case, and bit-exact AWQ
  final-layout payload copies.
- Public loader/config guard tests pass.

### Not covered by this run

- Loading and serving an actual Qwen3.5 AWQ checkpoint end to end.
- Numerical parity for a complete model over prefill and multi-token decode.
- Cache hit-rate, H2D bandwidth, TTFT, ITL, throughput, or peak-memory
  measurements.
- Multi-request, TP, EP, CUDA graph, EPLB, LoRA, shared-expert, hot weight
  reload, and shutdown-lifecycle behavior.
- The repository's full regression suite.

## 2026-09-18 Qwen3.5-MoE AWQ serving benchmark

This run validates end-to-end loading and serving of a real AWQ checkpoint and
measures how the GPU expert-cache budget affects single-request performance.
The local checkpoint directory is named `Qwen3.6-35B-A3B-AWQ`, while its
Transformers architecture is `Qwen3_5MoeForConditionalGeneration` (40 MoE
layers, 256 routed experts, top-8 routing). The checkpoint is approximately
24 GB and uses AWQ 4-bit weights with group size 128; SGLang converted it to
AWQ Marlin layout at load time.

### Code and environment

- Date: 2026-09-18 (Asia/Singapore)
- Repository: `/home/yaozhenyang/dev/sglang`
- Branch: `feat/expertpack/cpu-memory-backend`
- Commit under test: `3a583e96f2167152bf139b2537218c18fa0ad5d6`
- Conda environment: `sgl-main`
- Python: 3.12.0
- PyTorch: 2.13.0+cu130
- GPU: NVIDIA GeForce RTX 4070 SUPER, compute capability 8.9
- GPU UUID: `GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4`
- Driver: 615.71.09
- GPU memory: 12282 MiB
- Parallelism: TP=1, EP=1, DP=1
- `max_running_requests=1` (resolved by the `expert_pack` hook)
- CUDA graph disabled (resolved by the `expert_pack` hook)
- `max_total_tokens=4096`, `mem_fraction_static=0.88`

The Conda environment's `bin` directory must be in `PATH`, because AWQ Marlin's
runtime compilation invokes `ninja`.

### Recommended server command

The 3584 MiB cache was the best tested configuration that retained more than
1 GiB of free device memory after serving. It is therefore preferred over
pushing the cache to the card's absolute capacity.

```bash
env \
  PATH=/home/yaozhenyang/downloads/yes/envs/sgl-main/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  -m sglang.launch_server \
  --model-path /home/yaozhenyang/huggingface/Qwen3.6-35B-A3B-AWQ \
  --host 127.0.0.1 \
  --port 31000 \
  --load-format expert_pack \
  --model-loader-extra-config '{"source_backend":"cpu_memory","cache_vram_mib":3584,"cache_vram_reserve_mib":256,"stage_slots":32,"stats_flush_interval":2000,"stats_path":"/tmp/cpu-expert-bench/cache3584-stats.json"}' \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.88
```

The cache sweep used the following offloader settings; all other server
arguments were held fixed.

| Requested cache | Reserve | Stage slots | Actual cache | Slots/layer |
| ---: | ---: | ---: | ---: | ---: |
| 1024 MiB | 1536 MiB | 16 | 1023.4 MiB | 15-16 |
| 2048 MiB | 1024 MiB | 32 | 2047.0 MiB | 30-31 |
| 3072 MiB | 512 MiB | 32 | 3070.6 MiB | 45-46 |
| 3584 MiB | 256 MiB | 32 | 3583.0 MiB | 53-54 |

### Benchmark command

The sweep used two measured requests per configuration. The recommended
configuration was then checked with four measured requests. Every invocation
used one warmup request, deterministic sampling, fixed 128-token inputs and
64-token outputs, and concurrency one.

```bash
env \
  SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:31000 \
  --dataset-name random \
  --model /home/yaozhenyang/huggingface/Qwen3.6-35B-A3B-AWQ \
  --tokenizer /home/yaozhenyang/huggingface/Qwen3.6-35B-A3B-AWQ \
  --num-prompts 4 \
  --random-input-len 128 \
  --random-output-len 64 \
  --random-range-ratio 1 \
  --max-concurrency 1 \
  --request-rate inf \
  --warmup-requests 1 \
  --output-file /tmp/cpu-expert-bench/serving-cache3584-i128-o64-n4.jsonl
```

For the two-request sweep, only `--num-prompts` and `--output-file` changed.
The raw JSONL and stats files were retained under `/tmp/cpu-expert-bench/` for
the duration of this run.

### Serving results

| Cache | Requests | Duration | Output tok/s | Total tok/s | Mean E2E | Mean TTFT | Mean TPOT | Mean ITL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 MiB | 2 | 21.57 s | 5.93 | 17.80 | 10781.0 ms | 5690.7 ms | 80.80 ms | 80.80 ms |
| 2048 MiB | 2 | 16.31 s | 7.85 | 23.54 | 8140.6 ms | 3777.2 ms | 69.26 ms | 69.26 ms |
| 3072 MiB | 2 | 12.74 s | 10.05 | 30.14 | 6353.5 ms | 2552.7 ms | 60.33 ms | 60.33 ms |
| 3584 MiB | 2 | 11.14 s | 11.49 | 34.48 | 5552.2 ms | 2049.3 ms | 55.60 ms | 55.60 ms |
| **3584 MiB** | **4** | **23.20 s** | **11.04** | **33.11** | **5796.5 ms** | **2053.1 ms** | **59.42 ms** | **59.42 ms** |

All measured requests completed successfully and produced the requested 64
tokens. On the matched two-request sweep, increasing the cache from 1024 MiB
to 3584 MiB increased output throughput by 93.8%, reduced mean TTFT by 64.0%,
and reduced mean TPOT by 31.2%. The four-request confirmation corresponds to
approximately 16.83 decode tokens/s when expressed as `1 / mean_TPOT`; output
throughput is lower because it also includes prefill and request serialization.

### Cache and memory observations

These counters are snapshots accumulated since each server started. They
include SGLang's automatic warmup and the measured requests, and the number of
requests accumulated by each server was not identical. Consequently the hit
rate and transfer counters establish the trend, but absolute H2D byte totals
must not be interpreted as a per-request comparison.

| Cache | Cumulative hit rate | Cumulative H2D | H2D CUDA time | Ready wait | Peak allocated | `nvidia-smi` used/free |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 MiB | 34.45% | 192.77 GiB | 10.047 s | 5.291 s | 8.66 GiB | 9731 / 2251 MiB |
| 2048 MiB | 45.45% | 116.93 GiB | 5.948 s | 3.365 s | 9.16 GiB | 9941 / 2015 MiB |
| 3072 MiB | 59.17% | 84.09 GiB | 3.986 s | 2.422 s | 9.69 GiB | 10499 / 1457 MiB |
| 3584 MiB, after 2-request sweep | 60.26% | 67.34 GiB | 3.248 s | 1.997 s | 9.96 GiB | 10785 / 1171 MiB |
| 3584 MiB, final cumulative snapshot | 65.23% | 161.42 GiB | 7.776 s | 4.901 s | 9.97 GiB | 10845 / 1111 MiB |

The backend retained 17,927,503,872 bytes (16.70 GiB) of expert payloads in
host memory. The recommended configuration allocated 3,757,068,288 bytes for
GPU cache slots and 201,326,592 bytes for staging. The final four-request
serving result is the more stable headline measurement; the small sweep should
still be treated as directional rather than a statistically rigorous study.

### Known limitations and warnings

- This is a single-GPU, single-concurrency serving test. TP, EP, concurrent
  requests, CUDA graphs, EPLB, and LoRA remain outside the validated scope.
- CUDA graphs and shared-expert fusion are intentionally disabled by the
  `expert_pack` configuration guard.
- No RTX 4070 SUPER-specific Triton MoE tuning file existed for the tested
  physical expert counts (`E=16`, `31`, `46`, and `54`), so SGLang warned that
  it was using default, potentially suboptimal kernel configurations.
- Fused GDN decode projection/Conv1D fell back because its participating dtypes
  did not match.
- The benchmark client printed a `multiprocess.resource_tracker` exception
  during interpreter shutdown. It occurred after successful responses and did
  not change the benchmark result.
- The temporary SGLang services were stopped after the run. A final
  `nvidia-smi --query-compute-apps=...` returned no process, and port 31000 had
  no listener.
