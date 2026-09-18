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

### Narrow CPU/GPU profiling

The recommended 3584 MiB configuration was also profiled with one complete
request. The throughput and latency reported by profiler-enabled requests are
not benchmark results: trace collection and export perturb execution. The
analysis below uses the activity spans inside the traces.

Capturing an entire request with both CPU and GPU activities exceeded profiler
memory while exporting and the scheduler was killed with exit code -9. No
trace was produced by that failed attempt. The successful runs instead used
`--profile-by-stage --profile-num-steps 1`, which captures one prefill step and
one decode step, and collected GPU and CPU activities separately:

```bash
# Common serving arguments are identical to the benchmark command above.
# GPU activity trace:
... -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:31000 \
  --dataset-name random \
  --model /home/yaozhenyang/huggingface/Qwen3.6-35B-A3B-AWQ \
  --tokenizer /home/yaozhenyang/huggingface/Qwen3.6-35B-A3B-AWQ \
  --num-prompts 1 \
  --random-input-len 128 \
  --random-output-len 64 \
  --random-range-ratio 1 \
  --max-concurrency 1 \
  --request-rate inf \
  --warmup-requests 1 \
  --profile \
  --profile-activities GPU \
  --profile-by-stage \
  --profile-num-steps 1 \
  --profile-output-dir /tmp/cpu-expert-bench/profile3584-narrow \
  --profile-prefix cpu-expert-3584-narrow

# CPU activity trace: use the same arguments and replace GPU/output settings:
... --profile-activities CPU \
  --profile-output-dir /tmp/cpu-expert-bench/profile3584-cpu-narrow \
  --profile-prefix cpu-expert-3584-cpu-narrow
```

The GPU prefill trace covered 128 new tokens. The later CPU prefill trace saw
64 new and 64 prefix-cached tokens, so only its qualitative call-path evidence
is used. Decode remains one token per step in both traces.

| GPU trace phase | Trace span | Pinned H2D | H2D activity | Kernel activity | No kernel/copy activity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill, 128 new tokens | 2371.28 ms | 14887.69 MiB / 43288 copies | 662.52 ms (27.94%) | 70.46 ms (2.97%) | 1638.13 ms (69.08%) |
| Decode, one token | 66.95 ms | 113.46 MiB / 271 copies | 4.92 ms (7.35%) | 11.51 ms (17.19%) | 50.45 ms (75.36%) |

Pinned H2D achieved 23.56 GB/s in prefill and 24.10 GB/s in decode. The PCIe
copy engine is therefore transferring efficiently when active. H2D and kernels
were almost completely serialized: the union of all GPU kernel/copy intervals
was 733.15 ms in prefill and 16.50 ms in decode, almost exactly the sum of the
two activity classes.

Within the decode kernel time, dense GEMV kernels accounted for 8.84 ms and
AWQ Marlin MoE kernels for 1.39 ms. The MoE kernel itself is not the primary
decode bottleneck at this cache size. The larger issue is the approximately
50.45 ms per decode step during which the GPU executes neither a kernel nor a
copy.

The CPU trace directly identifies the offload control path as hot. Although
CPU profiling inflated the decode step to 108.88 ms and nested durations must
not be added together, it recorded:

- 40 calls to `expert_offload.py:apply`, totaling 54.80 ms;
- 40 cache `acquire` calls, totaling 30.13 ms;
- 50 `_load_expert` calls, totaling 22.58 ms;
- 50 pinned staging calls, totaling 16.94 ms.

For the partially prefix-cached prefill step, the CPU trace recorded 129
offload/apply microbatches and 2564 `_load_expert` calls. This confirms that
contiguous token microbatching and per-expert admission generate substantial
Python, event, and copy-launch overhead.

The profiling evidence points to the following optimization order:

1. Remove the per-layer logical-ID GPU-to-CPU-to-GPU round trip on cache hits.
   A GPU-resident logical-to-slot table could remap hits without host
   intervention and return only misses to the CPU policy.
2. Reduce prefill fragmentation. The current contiguous-range planner must
   repeatedly dispatch small MoE microbatches when the selected-expert union
   exceeds the 53-54 slots available per layer. An expert-major or grouped
   prefill path could reuse each admitted expert across more tokens.
3. Coalesce AWQ expert payload transfers. Thousands of small copies achieve
   good aggregate bandwidth but consume CPU launch/event work. A contiguous
   expert payload or batched copy path would reduce copy count.
4. Overlap transfers with useful work. The current dependency chain makes
   copies and MoE kernels nearly serial. Computing cache-hit experts while
   misses arrive, or accurate next-layer/token prefetch, could hide part of the
   H2D time.
5. After reducing control-plane gaps, tune the batch-one dense GEMV/GDN path.
   Dense GEMV already dominates measured decode kernel time, while Marlin MoE
   is comparatively small.

The successful compressed traces were retained under
`/tmp/cpu-expert-bench/profile3584-narrow/` and
`/tmp/cpu-expert-bench/profile3584-cpu-narrow/` during this run. They are not
committed because profiler traces are machine-specific and substantially
larger than the textual summary.

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

## 2026-09-18 all-pinned host-expert experiment

This experiment tests whether keeping the complete final-layout expert payload
in CUDA pinned host memory removes enough pageable-to-pinned staging work to
reduce the large GPU-idle gaps measured above. It uses the same model, 3584 MiB
GPU cache, request shape, concurrency, and benchmark seed as the four-request
pageable-source baseline.

### Experimental implementation

The opt-in loader setting was:

```json
{"source_backend":"cpu_memory","pin_host_experts":true,"cache_vram_mib":3584,"cache_vram_reserve_mib":256,"stage_slots":32,"stats_flush_interval":2000,"stats_path":"/tmp/cpu-expert-pinned-bench/cache3584-pinned-stats.json"}
```

After each layer's device-staged AWQ post-load transform, every nonempty expert
tensor was copied to pinned storage and its registered Parameter/buffer object
was kept while replacing its storage. This bounds transient host memory to one
tensor rather than duplicating the complete 16.70 GiB payload. Cache misses
then copy the immutable pinned expert slices directly to GPU cache slots. The
pageable staging ring remains the default when `pin_host_experts` is false.

AWQ Marlin has zero-byte `w13_g_idx_sort_indices` and
`w2_g_idx_sort_indices` placeholders. PyTorch reports an empty tensor as
unpinned even after `pin_memory()`. The first startup attempt exposed this edge
case and stopped before serving; empty tensors were then correctly treated as
having no payload to stage or protect for DMA. The failed attempt is retained
here because it was part of the experiment.

Before the model run, a 1 GiB CUDA pinned allocation succeeded in 0.43 seconds
despite the process reporting an 8 MiB `RLIMIT_MEMLOCK`. The running backend
reported all 40 layers as `source_memory=pinned` and the following allocation
totals:

| Allocation | Bytes | GiB |
| --- | ---: | ---: |
| Host expert payload | 17,927,503,872 | 16.70 |
| Pinned host expert payload | 17,927,503,872 | 16.70 |
| GPU expert cache | 3,757,068,288 | 3.50 |
| Pinned staging ring | 201,326,592 | 0.19 |

### Matched serving benchmark

The server and benchmark commands were the recommended commands above with
`"pin_host_experts":true`, `/tmp/cpu-expert-pinned-bench/` output paths, and
otherwise identical arguments. The measured workload remained four requests,
one warmup request, 128 input tokens, 64 output tokens, and concurrency one.

| Metric | Pageable source | All-pinned source | Change |
| --- | ---: | ---: | ---: |
| Benchmark duration | 23.20 s | 15.47 s | -33.3% |
| Output throughput | 11.04 tok/s | 16.55 tok/s | +49.9% |
| Total throughput | 33.11 tok/s | 49.64 tok/s | +49.9% |
| Mean E2E | 5796.5 ms | 3864.5 ms | -33.3% |
| Mean TTFT | 2053.1 ms | 987.4 ms | -51.9% |
| Mean TPOT / ITL | 59.42 ms | 45.67 ms | -23.1% |

All four measured requests completed and generated 64 tokens. Peak GPU tensor
allocation remained effectively unchanged at 9.97 GiB. During the later
profile runs the observed available host RAM reached 7.2 GiB, and swap use
temporarily rose from approximately 7.4 GiB to 9.2 GiB while the service was
alive. This memory pressure is the principal cost of the result.

### Matched narrow profiling

The GPU and CPU commands were the narrow profiling commands above with pinned
output directories and prefixes. Profiler latency is not a serving benchmark.
The decode traces are directly comparable: both contain 40 MoE applies and 50
expert misses. Transfer byte counts differ slightly with the routed experts.

| Decode GPU trace | Pageable source | All-pinned source |
| --- | ---: | ---: |
| GPU activity span | 66.55 ms | 54.97 ms |
| Pinned H2D | 113.46 MiB / 271 copies | 115.02 MiB / 277 copies |
| H2D activity | 4.94 ms | 5.13 ms |
| Kernel activity | 11.51 ms | 11.47 ms |
| No kernel/copy activity | 50.05 ms (75.2%) | 38.32 ms (69.7%) |

| Decode CPU trace | Calls | Pageable source | All-pinned source | Change |
| --- | ---: | ---: | ---: | ---: |
| `expert_offload.apply` | 40 | 54.80 ms | 40.03 ms | -27.0% |
| `ExpertCachePool.acquire` | 40 | 30.13 ms | 15.30 ms | -49.2% |
| `_load_expert` | 50 | 22.58 ms | 8.07 ms | -64.3% |
| `PinnedStagingRing.stage` | 50 | 16.94 ms | 3.12 ms | -81.6% |

The CPU prefill traces also had nearly identical work counts (129 applies and
2564 versus 2565 misses). Their cumulative `stage` time fell from 726.31 ms to
144.87 ms, while cumulative `apply` time fell from 1281.41 ms to 686.07 ms.
The separate GPU prefill traces are not used for a direct comparison because
the pinned run profiled a 64-new/64-prefix-cached step, whereas the earlier GPU
baseline profiled 128 new tokens.

### Conclusion

Pinning the complete host expert payload is effective on this machine. It
removes most pageable staging work and improves the matched headline output
throughput by 49.9%. It does not remove PCIe traffic: decode H2D and kernel
times are essentially unchanged. Even after pinning, approximately 38.3 ms of
the 55.0 ms decode GPU span contains neither a kernel nor a copy. The remaining
bottleneck is therefore the serial host-side routing, logical-to-slot remap,
cache-policy/event bookkeeping, and many small copy launches, not PCIe payload
bandwidth alone.

The raw benchmark, stats, and compressed traces were retained under
`/tmp/cpu-expert-pinned-bench/` for this run. After profiling, the service was
stopped; the GPU had no compute process, port 31000 had no listener, and
available host RAM returned to approximately 29 GiB.
